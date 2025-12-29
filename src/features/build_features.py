from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.constants import CANDIDATE_CATEGORIES

LOGGER = logging.getLogger(__name__)

INDEX_COLS = [
    "commune_code",
    "code_bv",
    "election_type",
    "election_year",
    "round",
    "date_scrutin",
]

PRESIDENTIAL_NAME_TO_CATEGORY = {
    "arthaud": "extreme_gauche",
    "poutou": "extreme_gauche",
    "melenchon": "gauche_dure",
    "roussel": "gauche_dure",
    "hidalgo": "gauche_modere",
    "jadot": "gauche_modere",
    "hamon": "gauche_modere",
    "macron": "centre",
    "lassalle": "centre",
    "cheminade": "centre",
    "pecresse": "droite_modere",
    "fillon": "droite_modere",
    "dupontaignan": "droite_dure",
    "asselineau": "droite_dure",
    "lepen": "extreme_droite",
    "zemmour": "extreme_droite",
}

EUROPEAN_LIST_KEYWORDS: list[tuple[str, str]] = [
    ("rassemblementnational", "extreme_droite"),
    ("lepen", "extreme_droite"),
    ("republiqueenmarche", "centre"),
    ("renaissance", "centre"),
    ("modem", "centre"),
    ("franceinsoumise", "gauche_dure"),
    ("lutteouvriere", "extreme_gauche"),
    ("revolutionnairecommunistes", "extreme_gauche"),
    ("communiste", "gauche_dure"),
    ("deboutlafrance", "droite_dure"),
    ("dupontaignan", "droite_dure"),
    ("frexit", "droite_dure"),
    ("patriotes", "droite_dure"),
    ("uniondeladroite", "droite_modere"),
    ("droiteetducentre", "droite_modere"),
    ("printempseuropeen", "gauche_modere"),
    ("generation", "gauche_modere"),
    ("animaliste", "gauche_modere"),
    ("ecolog", "gauche_modere"),
    ("federaliste", "centre"),
    ("pirate", "centre"),
    ("citoyenseuropeens", "centre"),
    ("leseuropeens", "centre"),
    ("lesoubliesdeleurope", "centre"),
    ("initiativecitoyenne", "centre"),
    ("esperanto", "centre"),
    ("europeauservicedespeuples", "droite_dure"),
    ("franceroyale", "extreme_droite"),
    ("pourleuropedesgens", "gauche_dure"),
    ("allonsenfants", "droite_modere"),
    ("alliancejaune", "centre"),
    ("giletsjaunes", "centre"),
]


def normalize_category(label: str | None) -> str | None:
    if label is None:
        return None
    norm = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    synonyms = {
        "doite_dure": "droite_dure",
        "droite_moderee": "droite_modere",
        "gauche_moderee": "gauche_modere",
        "extreme_gauche": "extreme_gauche",
        "extreme_droite": "extreme_droite",
        "divers": None,
        "gauche": "gauche_modere",
        "droite": "droite_modere",
    }
    mapped = synonyms.get(norm, norm)
    if mapped in CANDIDATE_CATEGORIES:
        return mapped
    return None


def _normalize_code_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA, "<NA>": pd.NA})
    )


def _normalize_person_name(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z]", "", text)


def _category_from_name(name: str | None) -> str | None:
    norm = _normalize_person_name(name)
    if not norm:
        return None
    for key, category in PRESIDENTIAL_NAME_TO_CATEGORY.items():
        if key in norm:
            return category
    return None


def _category_from_list_name(name: str | None) -> str | None:
    norm = _normalize_person_name(name)
    if not norm:
        return None
    for key, category in EUROPEAN_LIST_KEYWORDS:
        if key in norm:
            return category
    return None


def load_elections_long(path: Path, commune_code: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier long introuvable : {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";")
    df["date_scrutin"] = pd.to_datetime(df["date_scrutin"])
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce").fillna(df["date_scrutin"].dt.year)
    df["election_year"] = df["annee"]
    df["tour"] = pd.to_numeric(df["tour"], errors="coerce")
    df["round"] = df["tour"]
    for col in ["exprimes", "votants", "inscrits", "voix", "blancs", "nuls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "code_candidature" in df.columns:
        df["code_candidature"] = _normalize_code_series(df["code_candidature"])
    if "code_commune" in df.columns:
        df["code_commune"] = (
            df["code_commune"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    else:
        df["code_commune"] = df["code_bv"].astype(str).str.split("-").str[0]
    if commune_code is not None:
        df = df[df["code_commune"].astype(str) == str(commune_code)].copy()
    df = _unpivot_wide_candidates(df)
    if "code_candidature" in df.columns:
        df["code_candidature"] = _normalize_code_series(df["code_candidature"])
    df["type_scrutin"] = df["type_scrutin"].str.lower()
    df["election_type"] = df["type_scrutin"]
    return df


def _unpivot_wide_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    voix_cols = [c for c in df.columns if re.match(r"^Voix \d+$", str(c))]
    if not voix_cols:
        return df
    wide_mask = df[voix_cols].notna().any(axis=1)

    def _fill_unsuffixed_rows(local: pd.DataFrame) -> pd.DataFrame:
        # Some datasets only expose unsuffixed columns (Voix, Code Nuance).
        if "voix" in local.columns and "Voix" in local.columns:
            missing_voix = local["voix"].isna() | (local["voix"] == 0)
            local.loc[missing_voix, "voix"] = pd.to_numeric(
                local.loc[missing_voix, "Voix"],
                errors="coerce",
            )
        if "code_candidature" in local.columns:
            if "Code Nuance" in local.columns:
                local["code_candidature"] = local["code_candidature"].fillna(local["Code Nuance"])
            if "Nuance" in local.columns:
                local["code_candidature"] = local["code_candidature"].fillna(local["Nuance"])
        if "nom_candidature" in local.columns:
            if "Nom" in local.columns and "Prénom" in local.columns:
                prenom = local["Prénom"].fillna("").astype(str).str.strip()
                nom = local["Nom"].fillna("").astype(str).str.strip()
                combined = (prenom + " " + nom).str.strip().replace("", pd.NA)
                local["nom_candidature"] = local["nom_candidature"].fillna(combined)
            elif "Nom" in local.columns:
                local["nom_candidature"] = local["nom_candidature"].fillna(local["Nom"])
        return local

    if not wide_mask.any():
        return _fill_unsuffixed_rows(df)

    def _indexed_cols(pattern: str) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for col in df.columns:
            match = re.match(pattern, str(col))
            if match:
                mapping[int(match.group(1))] = col
        return mapping

    voice_map = _indexed_cols(r"^Voix (\d+)$")
    code_map = _indexed_cols(r"^Code Nuance (\d+)$")
    nuance_map = _indexed_cols(r"^Nuance (\d+)$")
    for idx, col in nuance_map.items():
        code_map.setdefault(idx, col)
    if "voix" in df.columns:
        voice_map.setdefault(1, "voix")
    if "code_candidature" in df.columns:
        code_map.setdefault(1, "code_candidature")

    if not any(idx > 1 for idx in voice_map):
        return df

    drop_cols = {c for c in df.columns if re.search(r"\s\d+$", str(c))}
    drop_cols.update({"voix", "code_candidature", "nom_candidature"})
    base_cols = [c for c in df.columns if c not in drop_cols]

    df_long = _fill_unsuffixed_rows(df[~wide_mask].copy())
    df_wide = df[wide_mask].copy()
    frames = []

    def _compose_nom(idx: int) -> pd.Series | None:
        series = pd.Series(pd.NA, index=df_wide.index, dtype="string")
        etendu_col = f"Libellé Etendu Liste {idx}"
        abrege_col = f"Libellé Abrégé Liste {idx}"
        nom_col = f"Nom {idx}"
        prenom_col = f"Prénom {idx}"

        if etendu_col in df_wide.columns:
            series = series.fillna(df_wide[etendu_col].astype("string"))
        if abrege_col in df_wide.columns:
            series = series.fillna(df_wide[abrege_col].astype("string"))
        if nom_col in df_wide.columns and prenom_col in df_wide.columns:
            prenom = df_wide[prenom_col].fillna("").astype(str).str.strip()
            nom = df_wide[nom_col].fillna("").astype(str).str.strip()
            combined = (prenom + " " + nom).str.strip().replace("", pd.NA)
            series = series.fillna(combined)
        elif nom_col in df_wide.columns:
            series = series.fillna(df_wide[nom_col].astype("string"))
        elif prenom_col in df_wide.columns:
            series = series.fillna(df_wide[prenom_col].astype("string"))
        if idx == 1 and "nom_candidature" in df_wide.columns:
            series = series.fillna(df_wide["nom_candidature"].astype("string"))
        if series.isna().all():
            return None
        return series

    for idx in sorted(voice_map):
        voix_col = voice_map[idx]
        if voix_col not in df_wide.columns:
            continue
        temp = df_wide[base_cols].copy()
        temp["voix"] = df_wide[voix_col]
        code_candidates = []
        if idx in code_map:
            code_candidates.append(code_map[idx])
        if idx in nuance_map and nuance_map[idx] not in code_candidates:
            code_candidates.append(nuance_map[idx])
        code_series = pd.Series(pd.NA, index=df_wide.index, dtype="string")
        for candidate in code_candidates:
            if candidate in df_wide.columns:
                code_series = code_series.fillna(df_wide[candidate])
        temp["code_candidature"] = code_series
        nom_series = _compose_nom(idx)
        if nom_series is not None:
            temp["nom_candidature"] = nom_series
        frames.append(temp)

    if not frames:
        return df
    wide_long = pd.concat(frames, ignore_index=True)
    wide_long["voix"] = pd.to_numeric(wide_long["voix"], errors="coerce")
    wide_long = wide_long[wide_long["voix"].notna() & (wide_long["voix"] > 0)]
    return pd.concat([df_long, wide_long], ignore_index=True)


def _mapping_from_yaml(mapping_path: Path) -> pd.DataFrame:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML est requis pour charger un mapping YAML.") from exc
    raw = yaml.safe_load(mapping_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Mapping YAML invalide: attendu un dictionnaire.")

    base_mapping = raw.get("base_mapping")
    mapping_entries = raw.get("mapping")
    overrides = raw.get("overrides", [])

    mapping = pd.DataFrame()
    if mapping_entries:
        mapping = pd.DataFrame(mapping_entries)
    elif base_mapping:
        base_path = Path(base_mapping)
        if not base_path.is_absolute():
            base_path = mapping_path.parent / base_path
        mapping = pd.read_csv(base_path, sep=";")
    else:
        mapping = pd.DataFrame(columns=["code_candidature", "nom_candidature", "bloc_1", "bloc_2", "bloc_3"])

    if overrides:
        override_df = pd.DataFrame(overrides)
        if not override_df.empty:
            if "blocs" in override_df.columns:
                blocs = override_df["blocs"].apply(lambda v: v if isinstance(v, list) else [])
                override_df["bloc_1"] = blocs.apply(lambda v: v[0] if len(v) > 0 else None)
                override_df["bloc_2"] = blocs.apply(lambda v: v[1] if len(v) > 1 else None)
                override_df["bloc_3"] = blocs.apply(lambda v: v[2] if len(v) > 2 else None)
                override_df = override_df.drop(columns=["blocs"])
            if "code_candidature" not in override_df.columns and "code" in override_df.columns:
                override_df = override_df.rename(columns={"code": "code_candidature"})
            if "nom_candidature" not in override_df.columns and "nom" in override_df.columns:
                override_df = override_df.rename(columns={"nom": "nom_candidature"})

            if "code_candidature" in mapping.columns:
                mapping["code_candidature"] = _normalize_code_series(mapping["code_candidature"])
            if "code_candidature" in override_df.columns:
                override_df["code_candidature"] = _normalize_code_series(override_df["code_candidature"])

            mapping = mapping.copy()
            for _, row in override_df.iterrows():
                code = row.get("code_candidature")
                if code is None:
                    continue
                mask = mapping["code_candidature"] == code
                if mask.any():
                    for col in ["nom_candidature", "bloc_1", "bloc_2", "bloc_3"]:
                        if col in row and pd.notna(row[col]):
                            mapping.loc[mask, col] = row[col]
                else:
                    mapping = pd.concat([mapping, pd.DataFrame([row])], ignore_index=True)
    return mapping


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping candidats/blocs manquant : {mapping_path}")
    if mapping_path.suffix in {".yml", ".yaml"}:
        mapping = _mapping_from_yaml(mapping_path)
    else:
        mapping = pd.read_csv(mapping_path, sep=";")
    if "code_candidature" in mapping.columns:
        mapping["code_candidature"] = _normalize_code_series(mapping["code_candidature"])
    bloc_cols = [c for c in mapping.columns if c.startswith("bloc")]
    for col in bloc_cols:
        mapping[col] = mapping[col].apply(normalize_category)
    return mapping


def expand_by_category(elections_long: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    df = elections_long.merge(mapping, on="code_candidature", how="left", suffixes=("", "_map"))
    records: list[dict] = []
    for row in df.itertuples(index=False):
        blocs = [getattr(row, col, None) for col in ["bloc_1", "bloc_2", "bloc_3"]]
        blocs = [normalize_category(b) for b in blocs if isinstance(b, str) or b is not None]
        blocs = [b for b in blocs if b is not None]
        voix = getattr(row, "voix", 0) or 0
        exprimes = getattr(row, "exprimes", np.nan)
        votants = getattr(row, "votants", np.nan)
        inscrits = getattr(row, "inscrits", np.nan)
        blancs = getattr(row, "blancs", np.nan)
        nuls = getattr(row, "nuls", np.nan)
        if not blocs:
            election_type = getattr(row, "election_type", None)
            if election_type == "presidentielles":
                nom = getattr(row, "nom_candidature", None)
                mapped = _category_from_name(nom)
                if mapped:
                    blocs = [mapped]
            elif election_type == "europeennes":
                nom = getattr(row, "nom_candidature", None)
                mapped = _category_from_list_name(nom)
                if mapped:
                    blocs = [mapped]
        if not blocs:
            # Fallback explicite : non mappé -> centre (évite un panel vide)
            blocs = ["centre"]
        part = voix / len(blocs) if len(blocs) > 0 else 0
        for bloc in blocs:
            records.append(
                {
                    "commune_code": getattr(row, "code_commune"),
                    "code_bv": getattr(row, "code_bv"),
                    "election_type": getattr(row, "election_type"),
                    "election_year": int(getattr(row, "election_year")),
                    "round": int(getattr(row, "round")) if not pd.isna(getattr(row, "round")) else None,
                    "date_scrutin": getattr(row, "date_scrutin"),
                    "category": bloc,
                    "voix_cat": part,
                    "exprimes": exprimes,
                    "votants": votants,
                    "inscrits": inscrits,
                    "blancs": blancs,
                    "nuls": nuls,
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate_by_event(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = INDEX_COLS + ["category"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            voix_cat=("voix_cat", "sum"),
            exprimes=("exprimes", "max"),
            votants=("votants", "max"),
            inscrits=("inscrits", "max"),
            blancs=("blancs", "max"),
            nuls=("nuls", "max"),
        )
    )
    agg["share"] = agg["voix_cat"] / agg["exprimes"].replace(0, np.nan)
    base_inscrits = agg["inscrits"].replace(0, np.nan)
    agg["turnout_pct"] = agg["votants"] / base_inscrits
    agg["blancs_pct"] = agg["blancs"] / base_inscrits
    agg["nuls_pct"] = agg["nuls"] / base_inscrits
    return agg


def compute_national_reference(local: pd.DataFrame) -> pd.DataFrame:
    nat_group_cols = ["election_type", "election_year", "round", "category"]
    nat = (
        local.groupby(nat_group_cols, as_index=False)
        .agg(
            voix_cat=("voix_cat", "sum"),
            exprimes=("exprimes", "sum"),
            votants=("votants", "sum"),
            inscrits=("inscrits", "sum"),
        )
    )
    nat["share_nat"] = nat["voix_cat"] / nat["exprimes"].replace(0, np.nan)
    nat["turnout_nat"] = nat["votants"] / nat["inscrits"].replace(0, np.nan)
    return nat[nat_group_cols + ["share_nat", "turnout_nat"]]


def add_lags(local: pd.DataFrame) -> pd.DataFrame:
    df = local.sort_values("date_scrutin").copy()
    df["share_lag_any"] = df.groupby(["code_bv", "category"])["share"].shift(1)
    df["share_lag2_any"] = df.groupby(["code_bv", "category"])["share"].shift(2)
    df["share_lag_same_type"] = df.groupby(["code_bv", "category", "election_type"])["share"].shift(1)
    df["dev_to_nat"] = df["share"] - df["share_nat"]
    df["dev_to_nat_lag_any"] = df.groupby(["code_bv", "category"])["dev_to_nat"].shift(1)
    df["dev_to_nat_lag_same_type"] = df.groupby(["code_bv", "category", "election_type"])["dev_to_nat"].shift(1)
    df["swing_any"] = df["share_lag_any"] - df["share_lag2_any"]
    return df


def _pivot_feature(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=INDEX_COLS, columns="category", values=value_col)
    pivot = pivot[[c for c in pivot.columns if c in CANDIDATE_CATEGORIES]]
    pivot.columns = [f"{prefix}{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def build_panel(
    elections_long_path: Path,
    mapping_path: Path,
    output_path: Path,
    *,
    csv_output: Path | None = None,
) -> pd.DataFrame:
    elections_long = load_elections_long(elections_long_path)
    mapping = load_mapping(mapping_path)
    expanded = expand_by_category(elections_long, mapping)
    local = aggregate_by_event(expanded)

    nat = compute_national_reference(local)
    local = local.merge(nat, on=["election_type", "election_year", "round", "category"], how="left")
    local = add_lags(local)

    turnout_event = (
        local.groupby(INDEX_COLS, as_index=False)["turnout_pct"].max().sort_values("date_scrutin")
    )
    turnout_event["prev_turnout_any_lag1"] = turnout_event.groupby("code_bv")["turnout_pct"].shift(1)
    turnout_event["prev_turnout_same_type_lag1"] = turnout_event.groupby(["code_bv", "election_type"])[
        "turnout_pct"
    ].shift(1)

    datasets: List[pd.DataFrame] = [
        _pivot_feature(local, "share", "target_share_"),
        _pivot_feature(local, "share_lag_any", "prev_share_any_lag1_"),
        _pivot_feature(local, "share_lag_same_type", "prev_share_type_lag1_"),
        _pivot_feature(local, "dev_to_nat_lag_any", "prev_dev_to_national_any_lag1_"),
        _pivot_feature(local, "dev_to_nat_lag_same_type", "prev_dev_to_national_type_lag1_"),
        _pivot_feature(local, "swing_any", "swing_any_"),
    ]
    panel = reduce(lambda left, right: left.merge(right, on=INDEX_COLS, how="left"), datasets)
    panel = panel.merge(
        turnout_event[INDEX_COLS + ["turnout_pct", "prev_turnout_any_lag1", "prev_turnout_same_type_lag1"]],
        on=INDEX_COLS,
        how="left",
    )

    target_cols = [f"target_share_{c}" for c in CANDIDATE_CATEGORIES]
    for col in target_cols:
        if col not in panel.columns:
            panel[col] = 0.0
    panel[target_cols] = panel[target_cols].fillna(0).clip(lower=0, upper=1)
    panel["target_sum_before_renorm"] = panel[target_cols].sum(axis=1)
    has_mass = panel["target_sum_before_renorm"] > 0
    panel.loc[has_mass, target_cols] = panel.loc[has_mass, target_cols].div(
        panel.loc[has_mass, "target_sum_before_renorm"], axis=0
    )
    panel["target_sum_after_renorm"] = panel[target_cols].sum(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    if csv_output:
        panel.to_csv(csv_output, sep=";", index=False)
    LOGGER.info("Panel enregistré dans %s (%s lignes)", output_path, len(panel))
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construction du dataset panel features+cibles sans fuite temporelle.")
    parser.add_argument(
        "--elections-long",
        type=Path,
        default=Path("data/interim/elections_long.parquet"),
        help="Chemin du format long harmonisé.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/nuances.yaml"),
        help="Mapping nuance -> catégorie.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/panel.parquet"),
        help="Destination du parquet panel.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/panel.csv"),
        help="Destination CSV optionnelle.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    build_panel(args.elections_long, args.mapping, args.output, csv_output=args.output_csv)


if __name__ == "__main__":
    main()
