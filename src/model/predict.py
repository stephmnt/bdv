from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.constants import CANDIDATE_CATEGORIES
from src.features.build_features import (
    aggregate_by_event,
    compute_national_reference,
    expand_by_category,
    load_elections_long,
    load_mapping,
)

LOGGER = logging.getLogger(__name__)


def filter_history(df: pd.DataFrame, target_year: int, commune_code: str | None) -> pd.DataFrame:
    df = df[df["annee"] < target_year]
    if commune_code:
        df = df[df["code_commune"] == commune_code]
    return df


def build_feature_matrix(
    elections_long: pd.DataFrame,
    mapping: pd.DataFrame,
    target_type: str,
    target_year: int,
) -> pd.DataFrame:
    expanded = expand_by_category(elections_long, mapping)
    local = aggregate_by_event(expanded)
    nat = compute_national_reference(local)
    local = local.merge(nat, on=["election_type", "election_year", "round", "category"], how="left")
    local["dev_to_nat"] = local["share"] - local["share_nat"]
    local = local.sort_values("date_scrutin")

    last_any_share = (
        local.sort_values("date_scrutin").groupby(["code_bv", "category"])["share"].last()
    )
    last_any_dev = (
        local.sort_values("date_scrutin").groupby(["code_bv", "category"])["dev_to_nat"].last()
    )
    last_type_share = (
        local[local["election_type"] == target_type]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])["share"]
        .last()
    )
    last_type_dev = (
        local[local["election_type"] == target_type]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])["dev_to_nat"]
        .last()
    )
    # Swing entre les deux derniers scrutins tous types
    swing_any = (
        local.groupby(["code_bv", "category"])["share"]
        .apply(lambda s: s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan)
        .rename("swing_any")
    )

    turnout_any = local.groupby("code_bv")["turnout_pct"].last()
    turnout_type = (
        local[local["election_type"] == target_type]
        .sort_values("date_scrutin")
        .groupby("code_bv")["turnout_pct"]
        .last()
    )

    bureaux = sorted(local["code_bv"].dropna().unique())
    records: List[dict] = []
    for code_bv in bureaux:
        record = {
            "commune_code": str(code_bv).split("-")[0],
            "code_bv": code_bv,
            "election_type": target_type,
            "election_year": target_year,
            "round": 1,
            "date_scrutin": f"{target_year}-01-01",
            "prev_turnout_any_lag1": turnout_any.get(code_bv, np.nan),
            "prev_turnout_same_type_lag1": turnout_type.get(code_bv, np.nan),
        }
        for cat in CANDIDATE_CATEGORIES:
            record[f"prev_share_any_lag1_{cat}"] = last_any_share.get((code_bv, cat), np.nan)
            record[f"prev_share_type_lag1_{cat}"] = last_type_share.get((code_bv, cat), np.nan)
            record[f"prev_dev_to_national_any_lag1_{cat}"] = last_any_dev.get((code_bv, cat), np.nan)
            record[f"prev_dev_to_national_type_lag1_{cat}"] = last_type_dev.get((code_bv, cat), np.nan)
            record[f"swing_any_{cat}"] = swing_any.get((code_bv, cat), np.nan)
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_references(local: pd.DataFrame, target_year: int) -> Dict[str, Dict[str, float]]:
    refs: Dict[str, Dict[str, float]] = {}
    leg = (
        local[(local["election_type"] == "legislatives") & (local["election_year"] < target_year)]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])
        .last()
    )
    mun2020 = (
        local[(local["election_type"] == "municipales") & (local["election_year"] == 2020)]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])
        .last()
    )
    refs["leg"] = {(code_bv, cat): row["share"] for (code_bv, cat), row in leg.iterrows()}
    refs["mun2020"] = {(code_bv, cat): row["share"] for (code_bv, cat), row in mun2020.iterrows()}
    return refs


def load_feature_columns(path: Path, df: pd.DataFrame) -> List[str]:
    if path.exists():
        return json.loads(path.read_text())
    # fallback: use all non-target columns except identifiers
    exclude = {"commune_code", "code_bv", "election_type", "election_year", "round", "date_scrutin"}
    return [c for c in df.columns if c not in exclude]


def predict(
    model_path: Path,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    refs: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    model = joblib.load(model_path)
    # Align feature set with trained columns (add missing as NaN)
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    for col in missing_cols:
        feature_df[col] = np.nan
    preds = model.predict(feature_df[feature_cols])
    preds = np.clip(preds, 0, 1)
    sums = preds.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    preds = preds / sums
    preds_pct = preds * 100

    rows = []
    for idx, row in feature_df.iterrows():
        code_bv = row["code_bv"]
        record = {
            "commune_code": row["commune_code"],
            "code_bv": code_bv,
        }
        for cat_idx, cat in enumerate(CANDIDATE_CATEGORIES):
            pred_val = preds_pct[idx, cat_idx]
            record[f"predicted_share_{cat}"] = round(float(pred_val), 2)
            leg_ref = refs["leg"].get((code_bv, cat))
            mun_ref = refs["mun2020"].get((code_bv, cat))
            record[f"delta_leg_{cat}"] = "N/A" if leg_ref is None else round(float(pred_val - leg_ref * 100), 2)
            record[f"delta_mun2020_{cat}"] = "N/A" if mun_ref is None else round(float(pred_val - mun_ref * 100), 2)
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédictions bureau par bureau pour une échéance cible.")
    parser.add_argument("--model-path", type=Path, default=Path("models/hist_gradient_boosting.joblib"), help="Modèle entraîné.")
    parser.add_argument("--feature-columns", type=Path, default=Path("models/feature_columns.json"), help="Colonnes de features attendues.")
    parser.add_argument("--elections-long", type=Path, default=Path("data/interim/elections_long.parquet"), help="Historique long.")
    parser.add_argument("--mapping", type=Path, default=Path("config/nuances.yaml"), help="Mapping nuances->catégories.")
    parser.add_argument("--target-election-type", type=str, default="municipales", help="Type d'élection cible.")
    parser.add_argument("--target-year", type=int, default=2026, help="Année cible.")
    parser.add_argument("--commune-code", type=str, default="34301", help="Code commune à filtrer (Sete=34301).")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"), help="Répertoire de sortie.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    elections_long = load_elections_long(args.elections_long)
    elections_long = filter_history(elections_long, args.target_year, args.commune_code)
    mapping = load_mapping(args.mapping)

    feature_df = build_feature_matrix(elections_long, mapping, args.target_election_type, args.target_year)
    if feature_df.empty:
        raise RuntimeError("Aucune donnée historique disponible pour construire les features.")
    feature_cols = load_feature_columns(args.feature_columns, feature_df)
    refs = compute_references(
        aggregate_by_event(expand_by_category(elections_long, mapping)).assign(
            election_type=lambda d: d["election_type"]
        ),
        args.target_year,
    )
    preds_df = predict(args.model_path, feature_df, feature_cols, refs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"pred_{args.target_election_type}_{args.target_year}_sete.csv"
    preds_df.to_csv(output_path, index=False)
    LOGGER.info("Prédictions écrites dans %s", output_path)


if __name__ == "__main__":
    main()
