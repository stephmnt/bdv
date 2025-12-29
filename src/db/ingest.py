from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.constants import CANDIDATE_CATEGORIES
from src.data import preprocess as preprocess_module
from src.db.schema import (
    bureaux,
    categories,
    communes,
    create_schema,
    elections,
    get_engine,
    results_local,
    results_national,
)
from src.features import build_features

LOGGER = logging.getLogger(__name__)
TARGET_COLS = [f"target_share_{c}" for c in CANDIDATE_CATEGORIES]
ID_COLS = ["commune_code", "code_bv", "election_type", "election_year", "round", "date_scrutin"]


def load_panel(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset panel introuvable : {input_path}")
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path, sep=";")


def ensure_panel_exists(panel_path: Path, elections_long_path: Path, mapping_path: Path) -> pd.DataFrame:
    if panel_path.exists():
        return load_panel(panel_path)
    LOGGER.info("Panel manquant, tentative de reconstruction via preprocess + build_features.")
    if not elections_long_path.exists():
        preprocess_module.preprocess_all(Path("data/raw"), elections_long_path.parent, preprocess_module.DEFAULT_META_CONFIG)
    build_features.build_panel(elections_long_path, mapping_path, panel_path, csv_output=None)
    return load_panel(panel_path)


def check_mass(panel: pd.DataFrame, tolerance: float = 0.05) -> None:
    sums = panel[TARGET_COLS].sum(axis=1)
    bad = panel[(sums < (1 - tolerance)) | (sums > (1 + tolerance))]
    if not bad.empty:
        LOGGER.warning("Somme des parts hors intervalle attendu pour %s lignes (tol=%s).", len(bad), tolerance)


def melt_panel(panel: pd.DataFrame) -> pd.DataFrame:
    long_df = panel.melt(id_vars=ID_COLS + ["turnout_pct"], value_vars=TARGET_COLS, var_name="category", value_name="share")
    long_df["category"] = long_df["category"].str.replace("target_share_", "", regex=False)
    return long_df


def _upsert_simple(conn, table, rows: Iterable[dict], index_elements: Iterable[str]) -> None:
    stmt = insert(table).values(list(rows))
    stmt = stmt.on_conflict_do_nothing(index_elements=list(index_elements))
    if rows:
        conn.execute(stmt)


def ingest(panel: pd.DataFrame, engine) -> None:
    check_mass(panel)
    panel = panel.copy()
    panel["round"] = panel["round"].fillna(1).astype(int)
    panel["date_scrutin"] = pd.to_datetime(panel["date_scrutin"]).dt.date

    long_df = melt_panel(panel)
    long_df = long_df[long_df["category"].isin(CANDIDATE_CATEGORIES)]
    long_df["share_pct"] = (long_df["share"].astype(float) * 100).round(6)

    with engine.begin() as conn:
        create_schema(conn)
        LOGGER.info("Schéma vérifié.")

        _upsert_simple(conn, categories, [{"name": cat} for cat in CANDIDATE_CATEGORIES], ["name"])
        cat_map = dict(conn.execute(sa.select(categories.c.name, categories.c.id)))

        commune_rows = [
            {"name_normalized": code, "insee_code": code}
            for code in sorted(long_df["commune_code"].dropna().unique())
        ]
        _upsert_simple(conn, communes, commune_rows, ["insee_code"])
        commune_map = dict(conn.execute(sa.select(communes.c.insee_code, communes.c.id)))

        def bureau_code_only(code_bv: str) -> str:
            if "-" in str(code_bv):
                parts = str(code_bv).split("-", 1)
                return parts[1]
            return str(code_bv)

        bureau_rows = []
        for _, row in long_df.drop_duplicates(subset=["commune_code", "code_bv"]).iterrows():
            commune_id = commune_map.get(row["commune_code"])
            if commune_id is None:
                continue
            bureau_rows.append(
                {
                    "commune_id": commune_id,
                    "bureau_code": bureau_code_only(row["code_bv"]),
                    "bureau_label": None,
                }
            )
        _upsert_simple(conn, bureaux, bureau_rows, ["commune_id", "bureau_code"])
        bureau_map = {
            (commune_id, bureau_code): bureau_id
            for bureau_id, commune_id, bureau_code in conn.execute(
                sa.select(bureaux.c.id, bureaux.c.commune_id, bureaux.c.bureau_code)
            )
        }

        election_rows = []
        for _, row in panel.drop_duplicates(subset=["election_type", "election_year", "round"]).iterrows():
            election_rows.append(
                {
                    "election_type": row["election_type"],
                    "election_year": int(row["election_year"]),
                    "round": int(row["round"]) if not pd.isna(row["round"]) else None,
                    "date": row["date_scrutin"],
                }
            )
        _upsert_simple(conn, elections, election_rows, ["election_type", "election_year", "round"])
        election_map: Dict[Tuple[str, int, int], int] = {
            (etype, year, int(round_) if round_ is not None else 1): eid
            for eid, etype, year, round_ in conn.execute(
                sa.select(elections.c.id, elections.c.election_type, elections.c.election_year, elections.c.round)
            )
        }

        local_rows = []
        for row in long_df.itertuples(index=False):
            commune_id = commune_map.get(row.commune_code)
            if commune_id is None:
                continue
            bureau_id = bureau_map.get((commune_id, bureau_code_only(row.code_bv)))
            election_id = election_map.get((row.election_type, int(row.election_year), int(row.round)))
            category_id = cat_map.get(row.category)
            if None in (bureau_id, election_id, category_id):
                continue
            turnout_pct = None if pd.isna(row.turnout_pct) else float(row.turnout_pct) * 100
            local_rows.append(
                {
                    "bureau_id": bureau_id,
                    "election_id": election_id,
                    "category_id": category_id,
                    "share_pct": None if pd.isna(row.share_pct) else float(row.share_pct),
                    "votes": None,
                    "expressed": None,
                    "turnout_pct": turnout_pct,
                }
            )
        if local_rows:
            stmt = insert(results_local).values(local_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["bureau_id", "election_id", "category_id"],
                set_={
                    "share_pct": stmt.excluded.share_pct,
                    "votes": stmt.excluded.votes,
                    "expressed": stmt.excluded.expressed,
                    "turnout_pct": stmt.excluded.turnout_pct,
                },
            )
            conn.execute(stmt)
        LOGGER.info("Résultats locaux insérés/mis à jour : %s lignes", len(local_rows))

        nat_rows = []
        nat = (
            long_df.groupby(["election_type", "election_year", "round", "category"], as_index=False)
            .agg(share=("share_pct", "mean"))
            .rename(columns={"share": "share_pct"})
        )
        # Participation moyenne par scrutin
        turnout_nat = panel.groupby(["election_type", "election_year", "round"], as_index=False)["turnout_pct"].mean()
        nat = nat.merge(turnout_nat, on=["election_type", "election_year", "round"], how="left")

        for row in nat.itertuples(index=False):
            election_id = election_map.get((row.election_type, int(row.election_year), int(row.round)))
            category_id = cat_map.get(row.category)
            if None in (election_id, category_id):
                continue
            nat_rows.append(
                {
                    "election_id": election_id,
                    "category_id": category_id,
                    "share_pct": None if pd.isna(row.share_pct) else float(row.share_pct),
                    "votes": None,
                    "expressed": None,
                    "turnout_pct": None if pd.isna(row.turnout_pct) else float(row.turnout_pct * 100),
                }
            )
        if nat_rows:
            stmt = insert(results_national).values(nat_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["election_id", "category_id"],
                set_={
                    "share_pct": stmt.excluded.share_pct,
                    "votes": stmt.excluded.votes,
                    "expressed": stmt.excluded.expressed,
                    "turnout_pct": stmt.excluded.turnout_pct,
                },
            )
            conn.execute(stmt)
        LOGGER.info("Référentiels nationaux insérés/mis à jour : %s lignes", len(nat_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion du panel harmonisé dans PostgreSQL.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/panel.parquet"), help="Chemin vers le panel parquet.")
    parser.add_argument(
        "--elections-long",
        type=Path,
        default=Path("data/interim/elections_long.parquet"),
        help="Format long (fallback pour reconstruire le panel).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/mapping_candidats_blocs.csv"),
        help="Mapping nuance -> catégorie (fallback).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    panel = ensure_panel_exists(args.input, args.elections_long, args.mapping)
    engine = get_engine()
    ingest(panel, engine)


if __name__ == "__main__":
    main()
