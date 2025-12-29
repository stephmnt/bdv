from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
import re
import yaml

from .constants import CANDIDATE_CATEGORIES


def normalize_bloc(bloc: str | None) -> str:
    """
    Map bloc labels to the canonical categories used across the project.
    """
    if bloc is None:
        return "centre"
    norm = str(bloc).strip().lower().replace(" ", "_").replace("-", "_")
    synonyms = {
        "droite_moderee": "droite_modere",
        "gauche_moderee": "gauche_modere",
        "doite_dure": "droite_dure",
        "gauche": "gauche_modere",
        "droite": "droite_modere",
        "divers": "centre",
        "divers_droite": "droite_modere",
        "divers_gauche": "gauche_modere",
        "divers_centre": "centre",
        "extreme_gauche": "extreme_gauche",
        "extreme_droite": "extreme_droite",
    }
    norm = synonyms.get(norm, norm)
    if norm not in CANDIDATE_CATEGORIES:
        return "centre"
    return norm


DEFAULT_COMMUNES_PATH = (Path(__file__).resolve().parents[1] / "config" / "communes.yaml")


def _normalize_insee_code(value: str | int | None) -> str:
    if value is None:
        return ""
    cleaned = (
        str(value)
        .strip()
        .replace(".0", "")
    )
    cleaned = re.sub(r"\D", "", cleaned)
    if not cleaned:
        return ""
    if len(cleaned) >= 5:
        return cleaned[:5]
    return cleaned.zfill(5)


def load_target_communes(path: Path = DEFAULT_COMMUNES_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier communes introuvable: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("communes", raw) if isinstance(raw, dict) else raw
    communes: dict[str, str] = {}

    if isinstance(entries, dict):
        for code, name in entries.items():
            norm = _normalize_insee_code(code)
            if norm:
                communes[norm] = str(name) if name is not None else ""
        return communes

    if not isinstance(entries, list):
        raise ValueError("Format YAML invalide: attendu une liste ou un mapping sous 'communes'.")

    for entry in entries:
        if isinstance(entry, str):
            norm = _normalize_insee_code(entry)
            if norm:
                communes[norm] = ""
            continue
        if isinstance(entry, dict):
            code = entry.get("code_insee") or entry.get("code") or entry.get("insee")
            name = entry.get("nom") or entry.get("name") or ""
            norm = _normalize_insee_code(code)
            if norm:
                communes[norm] = str(name) if name is not None else ""
            continue
    return communes


def load_elections_long(path: Path) -> pd.DataFrame:
    """
    Load the harmonised long format dataset (output of notebook 01_pretraitement).
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";")
    df["date_scrutin"] = pd.to_datetime(df["date_scrutin"])
    numeric_cols = ["exprimes", "inscrits", "votants", "voix", "blancs", "nuls"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["voix"] = df["voix"].fillna(0)
    return df


def _mapping_from_yaml(path: Path) -> pd.DataFrame:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML est requis pour charger un mapping YAML.") from exc
    raw = yaml.safe_load(path.read_text()) or {}
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
            base_path = path.parent / base_path
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

            mapping = mapping.copy()
            if "code_candidature" in mapping.columns:
                mapping["code_candidature"] = mapping["code_candidature"].astype(str)
            if "code_candidature" in override_df.columns:
                override_df["code_candidature"] = override_df["code_candidature"].astype(str)

            for _, row in override_df.iterrows():
                code = row.get("code_candidature")
                if code is None:
                    continue
                if "code_candidature" in mapping.columns:
                    mask = mapping["code_candidature"] == code
                else:
                    mask = pd.Series([False] * len(mapping))
                if mask.any():
                    for col in ["nom_candidature", "bloc_1", "bloc_2", "bloc_3"]:
                        if col in row and pd.notna(row[col]):
                            mapping.loc[mask, col] = row[col]
                else:
                    mapping = pd.concat([mapping, pd.DataFrame([row])], ignore_index=True)
    return mapping


def load_bloc_mapping(path: Path) -> pd.DataFrame:
    if path.suffix in {".yml", ".yaml"}:
        mapping = _mapping_from_yaml(path)
    else:
        mapping = pd.read_csv(path, sep=";")
    # normalise bloc labels once to avoid surprises downstream
    for col in ["bloc_1", "bloc_2", "bloc_3"]:
        if col in mapping.columns:
            mapping[col] = mapping[col].apply(normalize_bloc)
    return mapping


def expand_voix_by_bloc(elections_long: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute voix of each candidature across its mapped blocs.
    """
    df = elections_long.merge(mapping, on="code_candidature", how="left")
    records: list[dict] = []
    for _, row in df.iterrows():
        blocs = [row.get("bloc_1"), row.get("bloc_2"), row.get("bloc_3")]
        blocs = [b for b in blocs if isinstance(b, str) and b]
        blocs = [normalize_bloc(b) for b in blocs]
        if not blocs:
            blocs = ["centre"]
        voix = row.get("voix", 0) or 0
        repartition = voix / len(blocs)
        for bloc in blocs:
            records.append(
                {
                    "code_bv": row.get("code_bv"),
                    "nom_bv": row.get("nom_bv"),
                    "date_scrutin": row.get("date_scrutin"),
                    "annee": row.get("annee"),
                    "type_scrutin": row.get("type_scrutin"),
                    "tour": row.get("tour"),
                    "bloc": bloc,
                    "voix_bloc": repartition,
                    "exprimes": row.get("exprimes"),
                    "inscrits": row.get("inscrits"),
                    "votants": row.get("votants"),
                    "blancs": row.get("blancs"),
                    "nuls": row.get("nuls"),
                }
            )
    result = pd.DataFrame.from_records(records)
    result["date_scrutin"] = pd.to_datetime(result["date_scrutin"])
    for col in ["voix_bloc", "exprimes", "inscrits", "votants", "blancs", "nuls"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["part_bloc"] = result["voix_bloc"] / result["exprimes"]
    base_inscrits = result["inscrits"].replace(0, pd.NA)
    result["taux_participation_bv"] = result["votants"] / base_inscrits
    result["taux_blancs_bv"] = result["blancs"] / base_inscrits
    result["taux_nuls_bv"] = result["nuls"] / base_inscrits
    return result


def compute_national_reference(elections_blocs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate national part/participation per date & bloc if no external national file is provided.
    """
    grouped = (
        elections_blocs.groupby(["date_scrutin", "bloc"], as_index=False)[["voix_bloc", "exprimes", "votants", "inscrits"]]
        .sum()
        .rename(columns={"voix_bloc": "voix_bloc_nat", "exprimes": "exprimes_nat", "votants": "votants_nat", "inscrits": "inscrits_nat"})
    )
    grouped["part_bloc_national"] = grouped["voix_bloc_nat"] / grouped["exprimes_nat"].replace(0, pd.NA)
    grouped["taux_participation_national"] = grouped["votants_nat"] / grouped["inscrits_nat"].replace(0, pd.NA)
    return grouped[["date_scrutin", "bloc", "part_bloc_national", "taux_participation_national"]]


def attach_national_results(
    elections_blocs: pd.DataFrame,
    resultats_nationaux: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge national reference scores if provided; otherwise, compute them from the full dataset.
    """
    if resultats_nationaux is None:
        df_nat = compute_national_reference(elections_blocs)
    else:
        df_nat = resultats_nationaux.copy()
        df_nat["date_scrutin"] = pd.to_datetime(df_nat["date_scrutin"])

    elections_blocs = elections_blocs.merge(df_nat, on=["date_scrutin", "bloc"], how="left")
    elections_blocs["ecart_bloc_vs_national"] = (
        elections_blocs["part_bloc"] - elections_blocs["part_bloc_national"]
    )
    elections_blocs["ecart_participation_vs_nat"] = (
        elections_blocs["taux_participation_bv"] - elections_blocs["taux_participation_national"]
    )
    return elections_blocs


def compute_population_growth(elections_blocs: pd.DataFrame, base_year: int = 2014) -> pd.DataFrame:
    bv_pop = elections_blocs.groupby(["code_bv", "annee"], as_index=False)["inscrits"].mean()
    bv_base = (
        bv_pop[bv_pop["annee"] == base_year][["code_bv", "inscrits"]]
        .rename(columns={"inscrits": "inscrits_base"})
    )
    bv_pop = bv_pop.merge(bv_base, on="code_bv", how="left")
    bv_pop["croissance_inscrits_depuis_base"] = (
        bv_pop["inscrits"] - bv_pop["inscrits_base"]
    ) / bv_pop["inscrits_base"]

    elections_blocs = elections_blocs.merge(
        bv_pop[["code_bv", "annee", "croissance_inscrits_depuis_base"]],
        on=["code_bv", "annee"],
        how="left",
    )
    return elections_blocs


def add_lag_features(elections_blocs: pd.DataFrame) -> pd.DataFrame:
    df = elections_blocs.sort_values(["code_bv", "bloc", "date_scrutin"])
    df["part_bloc_lag1"] = df.groupby(["code_bv", "bloc"])["part_bloc"].shift(1)
    df["ecart_bloc_vs_national_lag1"] = df.groupby(["code_bv", "bloc"])[
        "ecart_bloc_vs_national"
    ].shift(1)
    df["taux_participation_bv_lag1"] = df.groupby(["code_bv", "bloc"])[
        "taux_participation_bv"
    ].shift(1)
    df["annee_centre"] = df["annee"] - df["annee"].median()
    return df


def filter_target_communes(elections_blocs: pd.DataFrame, target_communes: Mapping[str, str]) -> pd.DataFrame:
    """
    Keep only bureaux belonging to the target communes list.
    """
    df = elections_blocs.copy()
    if "code_commune" in df.columns:
        code_series = df["code_commune"].astype(str)
    else:
        code_series = df["code_bv"].astype(str).str.split("-").str[0]
    code_series = code_series.str.replace(r"\D", "", regex=True).str.zfill(5).str.slice(0, 5)
    df["code_commune"] = code_series
    df["nom_commune"] = df["code_commune"].map(target_communes)
    return df[df["code_commune"].isin(target_communes.keys())]


def compute_commune_event_stats(
    elections_long: pd.DataFrame,
    target_communes: Mapping[str, str],
) -> pd.DataFrame:
    df = elections_long.copy()
    if "code_commune" in df.columns:
        code_series = df["code_commune"].astype(str)
    else:
        code_series = df["code_bv"].astype(str).str.split("-").str[0]
    code_series = code_series.str.replace(r"\D", "", regex=True).str.zfill(5).str.slice(0, 5)
    df["code_commune"] = code_series
    df = df[df["code_commune"].isin(target_communes.keys())]
    df["nom_commune"] = df["code_commune"].map(target_communes)
    if "date_scrutin" in df.columns:
        df["date_scrutin"] = pd.to_datetime(df["date_scrutin"], errors="coerce")
    for col in ["exprimes", "inscrits", "votants", "blancs", "nuls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    bv_cols = [c for c in ["code_commune", "code_bv", "type_scrutin", "annee", "tour", "date_scrutin"] if c in df.columns]
    bv_event = (
        df.groupby(bv_cols, as_index=False)
        .agg(
            exprimes=("exprimes", "max"),
            inscrits=("inscrits", "max"),
            votants=("votants", "max"),
            blancs=("blancs", "max"),
            nuls=("nuls", "max"),
        )
    )
    commune_cols = [c for c in ["code_commune", "type_scrutin", "annee", "tour", "date_scrutin"] if c in bv_event.columns]
    commune = (
        bv_event.groupby(commune_cols, as_index=False)
        .agg(
            exprimes=("exprimes", "sum"),
            inscrits=("inscrits", "sum"),
            votants=("votants", "sum"),
            blancs=("blancs", "sum"),
            nuls=("nuls", "sum"),
        )
    )
    base_inscrits = commune["inscrits"].replace(0, pd.NA)
    commune["turnout_pct"] = commune["votants"] / base_inscrits
    commune["blancs_pct"] = commune["blancs"] / base_inscrits
    commune["nuls_pct"] = commune["nuls"] / base_inscrits
    commune["nom_commune"] = commune["code_commune"].map(target_communes)
    return commune


def build_elections_blocs(
    elections_long_path: Path,
    mapping_path: Path,
    *,
    national_results_path: Optional[Path] = None,
    base_year: int = 2014,
    target_communes_path: Path = DEFAULT_COMMUNES_PATH,
) -> pd.DataFrame:
    elections_long = load_elections_long(elections_long_path)
    mapping = load_bloc_mapping(mapping_path)

    elections_blocs = expand_voix_by_bloc(elections_long, mapping)

    national_df = None
    if national_results_path and national_results_path.exists():
        if national_results_path.suffix == ".parquet":
            national_df = pd.read_parquet(national_results_path)
        else:
            national_df = pd.read_csv(national_results_path, sep=";")
    # Always attach national reference (computed from full data if no external source)
    elections_blocs = attach_national_results(elections_blocs, national_df)
    # Restreindre aux communes cibles via le fichier YAML
    target_communes = load_target_communes(target_communes_path)
    elections_blocs = filter_target_communes(elections_blocs, target_communes)

    elections_blocs = compute_population_growth(elections_blocs, base_year=base_year)
    elections_blocs = add_lag_features(elections_blocs)
    return elections_blocs


def save_processed(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "elections_blocs.parquet"
    csv_path = output_dir / "elections_blocs.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, sep=";", index=False)


def save_commune_event_stats(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "commune_event_stats.parquet"
    csv_path = output_dir / "commune_event_stats.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, sep=";", index=False)


def run_full_pipeline(
    elections_long_path: Path = Path("data/interim/elections_long.parquet"),
    mapping_path: Path = Path("config/nuances.yaml"),
    output_dir: Path = Path("data/processed"),
    national_results_path: Optional[Path] = None,
    target_communes_path: Path = DEFAULT_COMMUNES_PATH,
) -> pd.DataFrame:
    df = build_elections_blocs(
        elections_long_path=elections_long_path,
        mapping_path=mapping_path,
        national_results_path=national_results_path,
        target_communes_path=target_communes_path,
    )
    save_processed(df, output_dir)
    elections_long = load_elections_long(elections_long_path)
    target_communes = load_target_communes(target_communes_path)
    commune_stats = compute_commune_event_stats(elections_long, target_communes)
    save_commune_event_stats(commune_stats, output_dir)
    return df


__all__ = [
    "build_elections_blocs",
    "run_full_pipeline",
    "save_processed",
    "normalize_bloc",
    "load_target_communes",
    "compute_commune_event_stats",
    "save_commune_event_stats",
]
