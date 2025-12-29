from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Column, Date, Float, Integer, MetaData, String, Table
from sqlalchemy.engine import Engine

from .constants import NUMERIC_COLUMNS
from .pipeline import normalize_bloc


def get_engine(url: Optional[str] = None) -> Engine:
    db_url = url or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Example: postgresql+psycopg2://user:pass@localhost:5432/elections")
    return sa.create_engine(db_url)


def define_schema(metadata: MetaData) -> Table:
    return Table(
        "election_results",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("code_bv", String(32), index=True, nullable=False),
        Column("nom_bv", String(255)),
        Column("date_scrutin", Date, index=True, nullable=False),
        Column("annee", Integer, index=True, nullable=False),
        Column("type_scrutin", String(32), index=True, nullable=False),
        Column("tour", Integer, nullable=False),
        Column("bloc", String(64), index=True, nullable=False),
        Column("voix_bloc", Float),
        Column("exprimes", Float),
        Column("inscrits", Float),
        Column("votants", Float),
        Column("blancs", Float),
        Column("nuls", Float),
        Column("part_bloc", Float),
        Column("part_bloc_national", Float),
        Column("taux_participation_national", Float),
        Column("taux_participation_bv", Float),
        Column("taux_blancs_bv", Float),
        Column("taux_nuls_bv", Float),
        Column("ecart_bloc_vs_national", Float),
        Column("ecart_participation_vs_nat", Float),
        Column("croissance_inscrits_depuis_base", Float),
        Column("part_bloc_lag1", Float),
        Column("ecart_bloc_vs_national_lag1", Float),
        Column("taux_participation_bv_lag1", Float),
        Column("annee_centre", Float),
    )


def create_schema(engine: Engine) -> None:
    metadata = MetaData()
    define_schema(metadata)
    metadata.create_all(engine)


def _coerce_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_processed_to_db(
    processed_path: Path = Path("data/processed/elections_blocs.csv"),
    *,
    engine: Optional[Engine] = None,
    if_exists: str = "replace",
    chunksize: int = 1000,
) -> int:
    """
    Load the processed bloc-level dataset into PostgreSQL.

    Returns the number of rows written.
    """
    engine = engine or get_engine()
    create_schema(engine)

    df = pd.read_csv(processed_path, sep=";")
    df["date_scrutin"] = pd.to_datetime(df["date_scrutin"]).dt.date
    if "bloc" in df.columns:
        df["bloc"] = df["bloc"].apply(normalize_bloc)
    df = _coerce_numeric(df, NUMERIC_COLUMNS)

    df.to_sql(
        "election_results",
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=chunksize,
    )
    return len(df)


def list_bureaux(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        result = conn.execute(sa.text("select distinct code_bv from election_results order by code_bv"))
        return [row[0] for row in result.fetchall()]


def fetch_history(engine: Engine, code_bv: str) -> pd.DataFrame:
    query = sa.text(
        """
        select *
        from election_results
        where code_bv = :code_bv
        order by date_scrutin asc, bloc asc
        """
    )
    return pd.read_sql(query, engine, params={"code_bv": code_bv})


__all__ = [
    "create_schema",
    "define_schema",
    "fetch_history",
    "get_engine",
    "list_bureaux",
    "load_processed_to_db",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialise la base et charge les résultats.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Charger data/processed/elections_blocs.csv dans la base (remplace la table).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/processed/elections_blocs.csv"),
        help="Chemin vers le fichier processe (CSV ; par defaut data/processed/elections_blocs.csv).",
    )
    args = parser.parse_args()

    engine = get_engine()
    create_schema(engine)
    if args.load:
        rows = load_processed_to_db(args.path, engine=engine)
        print(f"{rows} lignes inserees dans election_results.")
    else:
        print("Schema cree. Utilisez --load pour charger les donnees.")
