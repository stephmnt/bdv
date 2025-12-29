from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from src import data_prep

LOGGER = logging.getLogger(__name__)


DEFAULT_META_CONFIG: Dict[str, Dict[str, Any]] = {
    "14_EU.csv": {
        "type_scrutin": "europeennes",
        "date_scrutin": "2014-05-25",
        "tour_column": "N° tour",
        "code_bv_cols": ["Code de la commune", "N° de bureau de vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Votants": "votants",
            "Exprimés": "exprimes",
            "ExprimÃ©s": "exprimes",
            "Nombre de voix du candidat": "voix",
            "Voix": "voix",
            "Nom du candidat": "nom_candidature",
            "Prénom du candidat": "nom_candidature",
            "Code nuance du candidat": "code_candidature",
        },
    },
    "14_MN14_T1T2.csv": {
        "type_scrutin": "municipales",
        "date_scrutin": "2014-03-23",
        "tour_column": "N° tour",
        "code_bv_cols": ["Code commune", "N° de bureau de vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Votants": "votants",
            "Exprimés": "exprimes",
            "Nombre de voix": "voix",
            "Nom du candidat tête de liste": "nom_candidature",
            "Prénom du candidat  tête de liste": "nom_candidature",
            "Code nuance de la liste": "code_candidature",
        },
    },
    "17_L_T1.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2017-06-11",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Nom": "nom_candidature",
        },
    },
    "17_L_T2.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2017-06-18",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Nom": "nom_candidature",
        },
    },
    "17_PR_T1.csv": {
        "type_scrutin": "presidentielles",
        "date_scrutin": "2017-04-23",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Code nuance du candidat": "code_candidature",
        },
    },
    "17_PR_T2.csv": {
        "type_scrutin": "presidentielles",
        "date_scrutin": "2017-05-07",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Code nuance du candidat": "code_candidature",
        },
    },
    "19_EU.csv": {
        "type_scrutin": "europeennes",
        "date_scrutin": "2019-05-26",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom Tête de Liste": "nom_candidature",
            "Nuance Liste": "code_candidature",
        },
    },
    "20_MN_T1.csv": {
        "type_scrutin": "municipales",
        "date_scrutin": "2020-03-15",
        "tour": 1,
        "sep": ";",
        "code_bv_cols": ["Code de la commune", "Code B.Vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Liste": "nom_candidature",
            "Code Nuance": "code_candidature",
        },
    },
    "20_MN_T2.csv": {
        "type_scrutin": "municipales",
        "date_scrutin": "2020-06-28",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code B.Vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Liste": "nom_candidature",
            "Code Nuance": "code_candidature",
        },
    },
    "21_DEP_T1.csv": {
        "type_scrutin": "departementales",
        "date_scrutin": "2021-06-20",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Binôme": "nom_candidature",
        },
    },
    "21_DEP_T2.csv": {
        "type_scrutin": "departementales",
        "date_scrutin": "2021-06-27",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Binôme": "nom_candidature",
        },
    },
    "21_REG_T1.csv": {
        "type_scrutin": "regionales",
        "date_scrutin": "2021-06-20",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance Liste": "code_candidature",
            "Libellé Abrégé Liste": "nom_candidature",
        },
    },
    "21_REG_T2.csv": {
        "type_scrutin": "regionales",
        "date_scrutin": "2021-06-27",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance Liste": "code_candidature",
            "Libellé Abrégé Liste": "nom_candidature",
        },
    },
    "22_L_T1.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2022-06-12",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Nom": "nom_candidature",
        },
    },
    "22_L_T2.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2022-06-19",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance": "code_candidature",
            "Nom": "nom_candidature",
        },
    },
    "22_PR_T1.csv": {
        "type_scrutin": "presidentielles",
        "date_scrutin": "2022-04-10",
        "tour": 1,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Code nuance du candidat": "code_candidature",
        },
    },
    "22_PR_T2.csv": {
        "type_scrutin": "presidentielles",
        "date_scrutin": "2022-04-24",
        "tour": 2,
        "code_bv_cols": ["Code de la commune", "Code du b.vote"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nom": "nom_candidature",
            "Code nuance du candidat": "code_candidature",
        },
    },
    "24_EU.csv": {
        "type_scrutin": "europeennes",
        "date_scrutin": "2024-06-09",
        "tour": 1,
        "code_bv_cols": ["Code commune", "Code BV"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix 1": "voix",
            "Voix": "voix",
            "Nuance liste 1": "code_candidature",
            "Libellé abrégé de liste 1": "nom_candidature",
        },
    },
    "24_L_T1.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2024-06-30",
        "tour": 1,
        "code_bv_cols": ["Code commune", "Code BV"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance Liste": "code_candidature",
            "Libellé Abrégé Liste": "nom_candidature",
            "Binôme": "nom_candidature",
        },
    },
    "24_L_T2.csv": {
        "type_scrutin": "legislatives",
        "date_scrutin": "2024-07-07",
        "tour": 2,
        "code_bv_cols": ["Code commune", "Code BV"],
        "rename_map": {
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "Exprimés": "exprimes",
            "Voix": "voix",
            "Nuance Liste": "code_candidature",
            "Libellé Abrégé Liste": "nom_candidature",
            "Binôme": "nom_candidature",
        },
    },
}

DEFAULT_META_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "raw_sources.yaml"


def _resolve_meta_config(raw: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    resolved: Dict[str, Dict[str, Any]] = {}

    def resolve_one(key: str, stack: list[str]) -> Dict[str, Any]:
        if key in resolved:
            return resolved[key]
        if key in stack:
            raise ValueError(f"Cycle detecte dans meta-config: {' -> '.join(stack + [key])}")
        meta = dict(raw[key])
        base_key = meta.pop("copy_from", None)
        if base_key:
            if base_key not in raw:
                raise KeyError(f"copy_from cible introuvable: {base_key}")
            base = resolve_one(base_key, stack + [key])
            merged = dict(base)
            rename_base = dict(base.get("rename_map", {}))
            rename_override = dict(meta.get("rename_map", {}))
            merged.update(meta)
            if rename_base or rename_override:
                merged["rename_map"] = {**rename_base, **rename_override}
            resolved[key] = merged
        else:
            resolved[key] = meta
        return resolved[key]

    for name in raw:
        resolve_one(name, [])
    return resolved


def load_meta_config(meta_path: Path | None) -> Dict[str, Dict[str, Any]]:
    if meta_path is None:
        if DEFAULT_META_CONFIG_PATH.exists():
            meta_path = DEFAULT_META_CONFIG_PATH
        else:
            return DEFAULT_META_CONFIG
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta-config file not found: {meta_path}")
    if meta_path.suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to read YAML meta-config files.") from exc
        raw = yaml.safe_load(meta_path.read_text()) or {}
    else:
        raw = json.loads(meta_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Meta-config invalide: attendu un mapping de fichiers vers meta-donnees.")
    return _resolve_meta_config(raw)


def preprocess_all(raw_dir: Path, output_dir: Path, meta_config: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    missing: list[str] = []
    for file_name, meta in meta_config.items():
        path = raw_dir / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        LOGGER.info("Standardisation de %s", file_name)
        df_std = data_prep.standardize_election(
            path,
            meta,
            rename_map=meta.get("rename_map", {}),
            sep=meta.get("sep", ";"),
            encoding=meta.get("encoding", ("cp1252", "utf-8-sig", "latin-1")),
            decimal=meta.get("decimal", ","),
        )  # type: ignore[arg-type]
        frames.append(df_std)
    if missing:
        LOGGER.warning("Fichiers manquants ignorés: %s", ", ".join(sorted(missing)))
    if not frames:
        raise RuntimeError("Aucune donnée chargée : vérifier le dossier raw et la configuration meta.")

    elections_long = pd.concat(frames, ignore_index=True)
    elections_long["date_scrutin"] = pd.to_datetime(elections_long["date_scrutin"])
    elections_long["annee"] = elections_long["date_scrutin"].dt.year
    elections_long["type_scrutin"] = elections_long["type_scrutin"].str.lower()
    elections_long["code_commune"] = elections_long["code_bv"].astype(str).str.split("-").str[0]

    issues = data_prep.validate_consistency(elections_long)
    for name, df_issue in issues.items():
        if len(df_issue) > 0:
            LOGGER.warning("%s : %s lignes a inspecter", name, len(df_issue))

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "elections_long.parquet"
    csv_path = output_dir / "elections_long.csv"
    elections_long.to_parquet(parquet_path, index=False)
    elections_long.to_csv(csv_path, sep=";", index=False)
    LOGGER.info("Long format sauvegarde (%s lignes) -> %s / %s", len(elections_long), parquet_path, csv_path)
    return elections_long


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prétraitement des fichiers bruts en format long standardisé.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Répertoire des fichiers bruts CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"), help="Destination du format long harmonisé.")
    parser.add_argument(
        "--meta-config",
        type=Path,
        default=None,
        help="Chemin vers un fichier JSON/YAML décrivant les meta-données des scrutins. Par défaut, utilise la configuration embarquée.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    meta_config = load_meta_config(args.meta_config)
    preprocess_all(args.raw_dir, args.output_dir, meta_config)


if __name__ == "__main__":
    main()
