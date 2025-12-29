from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

# Columns kept across all scrutins
STANDARD_COLUMNS: List[str] = [
    "code_bv",
    "nom_bv",
    "annee",
    "date_scrutin",
    "type_scrutin",
    "tour",
    "inscrits",
    "votants",
    "abstentions",
    "blancs",
    "nuls",
    "exprimes",
    "code_candidature",
    "nom_candidature",
    "voix",
]

NUMERIC_COLUMNS = [
    "inscrits",
    "votants",
    "abstentions",
    "blancs",
    "nuls",
    "exprimes",
    "voix",
]


_MOJIBAKE_REPLACEMENTS = {
    "Ã©": "é",
    "Ã¨": "è",
    "Ãª": "ê",
    "Ã«": "ë",
    "Ã ": "à",
    "Ã¢": "â",
    "Ã§": "ç",
    "Ã¹": "ù",
    "Ã»": "û",
    "Ã¯": "ï",
    "Ã´": "ô",
    "Ã¶": "ö",
    "Ã‰": "É",
    "Ãˆ": "È",
    "ÃŠ": "Ê",
    "Ã‹": "Ë",
    "Ã€": "À",
    "Ã‚": "Â",
    "Ã‡": "Ç",
    "ï¿½": "°",
    "�": "°",
}


def _normalize_label(label: str) -> str:
    """
    Attempt to repair mojibake in column labels (UTF-8 read as latin-1 or vice versa).
    """
    fixed = label
    try:
        fixed = label.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = label
    else:
        if "Â" in fixed:
            fixed = fixed.replace("Â", "")
    try:
        # Alternate path: utf-8 bytes decoded as latin1 then re-decoded
        fixed = fixed.encode("utf-8").decode("latin1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        if bad in fixed:
            fixed = fixed.replace(bad, good)
    fixed = fixed.replace("\ufeff", "")  # remove BOM
    fixed = " ".join(fixed.split())  # normalise whitespace
    return fixed


def _canonical_label(label: str) -> str:
    """
    Lowercase alpha-numeric only version of a label for fuzzy matching.
    """
    import re

    norm = _normalize_label(label).lower()
    return re.sub(r"[^0-9a-z]", "", norm)


def _unpivot_wide_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect wide candidate columns (e.g., 'Voix 1', 'Nuance liste 2') and unpivot to long.
    Keeps one row per candidate with standard columns 'voix' and 'code_candidature'.
    """
    pattern = re.compile(r"^(?P<base>.*?)(?:\s+|_)?(?P<idx>\d+)$")
    candidate_map: Dict[str, Dict[str, str]] = {}
    wide_cols: set[str] = set()
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        wide_cols.add(col)
        base = match.group("base").strip()
        idx = match.group("idx")
        canon = _canonical_label(base)
        field = None
        if canon == "voix":
            field = "voix"
        elif canon in {"nuance", "nuanceliste", "codenuance", "codenuanceducandidat", "codenuanceliste"}:
            field = "code_candidature"
        if field:
            candidate_map.setdefault(idx, {})[field] = col

    indices = [
        idx for idx, fields in candidate_map.items()
        if {"voix", "code_candidature"}.issubset(fields.keys())
    ]
    if len(indices) <= 1:
        return df

    candidate_cols = {col for fields in candidate_map.values() for col in fields.values()}
    base_cols = [c for c in df.columns if c not in wide_cols]
    frames = []
    for idx in sorted(indices, key=lambda v: int(v)):
        fields = candidate_map[idx]
        use_cols = base_cols + list(fields.values())
        sub = df[use_cols].copy()
        sub = sub.rename(
            columns={
                fields["voix"]: "voix",
                fields["code_candidature"]: "code_candidature",
            }
        )
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple columns end up with the same name after rename/normalization,
    keep the first non-null value across duplicates and drop the extras.
    """
    df = df.copy()
    duplicates = df.columns[df.columns.duplicated()].unique()
    for col in duplicates:
        cols = [c for c in df.columns if c == col]
        base = df[cols[0]]
        for extra in cols[1:]:
            base = base.fillna(df[extra])
        df[col] = base
        df = df.drop(columns=cols[1:])
    # ensure uniqueness
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_raw(
    path: Path,
    *,
    sep: str = ";",
    encoding: str | Iterable[str] = "cp1252",
    decimal: str = ",",
    dtype: Optional[Mapping[str, str]] = None,
    engine: str = "c",
) -> pd.DataFrame:
    """
    Wrapper around read_csv with encoding fallbacks to mitigate mojibake.

    Tries encodings in order (default: cp1252, utf-8-sig, latin-1) until column
    names no longer contain replacement artefacts (� or Ã), then normalises labels.
    """
    encoding_choices: List[str] = []
    if isinstance(encoding, str):
        encoding_choices.append(encoding)
    else:
        encoding_choices.extend(list(encoding))
    encoding_choices.extend([e for e in ["utf-8-sig", "latin-1"] if e not in encoding_choices])

    last_exc: Optional[Exception] = None
    for enc in encoding_choices:
        try:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    decimal=decimal,
                    dtype=dtype,  # type: ignore
                    engine=engine,  # type: ignore
                    low_memory=False,
                )
            except pd.errors.ParserError:
                # Retry with python engine and skip malformed lines (low_memory not supported)
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    decimal=decimal,
                    dtype=dtype,  # type: ignore
                    engine="python",
                    on_bad_lines="skip",
                )
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue

        bad_cols = any(("�" in col) or ("Ã" in col) for col in df.columns)
        if bad_cols and enc != encoding_choices[-1]:
            # try next encoding candidate
            continue

    df.columns = [_normalize_label(c) for c in df.columns]
    return df

    if last_exc:
        raise last_exc
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "unable to decode with provided encodings")


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """
    Add missing columns with NaN placeholders to guarantee downstream compatibility.
    """
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    return df


def add_election_metadata(df: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """
    Attach metadata about the scrutin to each row.

    Required meta keys:
        - type_scrutin
        - tour
        - date_scrutin

    Optional:
        - annee (otherwise derived from date_scrutin)
    """
    df["type_scrutin"] = meta["type_scrutin"]
    df["tour"] = int(meta["tour"]) # type: ignore
    df["date_scrutin"] = pd.to_datetime(meta["date_scrutin"]) # type: ignore
    df["annee"] = meta.get("annee", df["date_scrutin"].dt.year) # type: ignore
    return df


def build_code_bv(df: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """
    Ensure a code_bv column exists. If already present, it is left intact.

    Optionally, pass in meta["code_bv_cols"] as a list of column names to combine.
    """
    if "code_bv" in df.columns:
        df["code_bv"] = df["code_bv"].astype(str).str.strip()
        return df

    columns_to_concat: Optional[List[str]] = meta.get("code_bv_cols")  # type: ignore[arg-type]
    if columns_to_concat:
        actual_cols: List[str] = []
        canon_map = {_canonical_label(col): col for col in df.columns}
        for target in columns_to_concat:
            canon = _canonical_label(target)
            if canon in canon_map:
                actual_cols.append(canon_map[canon])
            else:
                raise KeyError(f"{target!r} not found in columns. Available: {list(df.columns)}")

        df["code_bv"] = (
            df[actual_cols]
            .astype(str)
            .apply(lambda row: "-".join([v.zfill(3) if v.isdigit() else v for v in row]), axis=1)
        )
    else:
        raise KeyError("code_bv not found in dataframe and no code_bv_cols provided in meta.")
    return df


def coerce_numeric(df: pd.DataFrame, numeric_cols: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply harmonisations common to all scrutins.
    """
    df = df.copy()
    df["voix"] = df.get("voix", 0).fillna(0) # type: ignore

    # Recompute exprimes when possible
    mask_expr = (
        df["exprimes"].isna()
        & df["votants"].notna()
        & df["blancs"].notna()
        & df["nuls"].notna()
    )
    df.loc[mask_expr, "exprimes"] = (
        df.loc[mask_expr, "votants"] - df.loc[mask_expr, "blancs"] - df.loc[mask_expr, "nuls"]
    )

    # Remove rows without minimal identifiers
    df = df[df["code_bv"].notna()]
    return df


def standardize_election(
    path: Path,
    meta: Mapping[str, object],
    *,
    rename_map: Optional[Mapping[str, str]] = None,
    sep: str = ";",
    encoding: str | Iterable[str] = ("cp1252", "utf-8-sig", "latin-1"),
    decimal: str = ",",
    dtype: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Load and standardise a single raw table to the long format expected downstream.

    Parameters
    ----------
    path : Path
        CSV path to the raw election table.
    meta : Mapping
        Must contain type_scrutin, tour, date_scrutin. Optionally code_bv_cols and annee.
    rename_map : Mapping
        Columns to rename from the raw schema to the standard schema.
    """
    df_raw = load_raw(path, sep=sep, encoding=encoding, decimal=decimal, dtype=dtype)
    rename_norm = {_normalize_label(k): v for k, v in (rename_map or {}).items()}

    def _process(df: pd.DataFrame, meta_for_tour: Mapping[str, object]) -> pd.DataFrame:
        df_local = df.copy()
        df_local.columns = [_normalize_label(c) for c in df_local.columns]
        df_local = _unpivot_wide_candidates(df_local)
        if rename_norm:
            # Renommer en se basant sur une version canonique (sans accents/espaces) et en ignorant d'éventuels suffixes numériques.
            import re

            def canonical_base(label: str) -> str:
                base = _canonical_label(label)
                return re.sub(r"\\d+$", "", base)

            rename_by_base = {canonical_base(k): v for k, v in rename_norm.items()}
            rename_using = {}
            for col in df_local.columns:
                base = canonical_base(col)
                if base in rename_by_base:
                    rename_using[col] = rename_by_base[base]
            df_local = df_local.rename(columns=rename_using)
        df_local = deduplicate_columns(df_local)
        df_local = df_local.loc[:, ~df_local.columns.duplicated()]

        df_local = build_code_bv(df_local, meta_for_tour)
        df_local = add_election_metadata(df_local, meta_for_tour)
        df_local = ensure_columns(df_local, STANDARD_COLUMNS)
        df_local = coerce_numeric(df_local)
        df_local = basic_cleaning(df_local)
        ordered_cols = STANDARD_COLUMNS + [col for col in df_local.columns if col not in STANDARD_COLUMNS]
        return df_local[ordered_cols]

    # Multi-tour handling: split on tour_column if provided and "tour" not explicit
    if meta.get("tour_column") and "tour" not in meta:
        tour_col = _normalize_label(str(meta["tour_column"]))
        if tour_col not in df_raw.columns:
            # Fallback: considérer un seul tour = 1 si la colonne est introuvable
            meta_single = {k: v for k, v in meta.items() if k != "tour_column"}
            meta_single["tour"] = int(meta.get("tour", 1))
            return _process(df_raw, meta_single)
        tours = meta.get("tours") or sorted(df_raw[tour_col].dropna().unique())
        frames: list[pd.DataFrame] = []
        for tour_val in tours:
            meta_tour = {k: v for k, v in meta.items() if k != "tour_column"}
            meta_tour["tour"] = int(tour_val)
            frames.append(_process(df_raw[df_raw[tour_col] == tour_val], meta_tour))
        if not frames:
            raise RuntimeError(f"Aucun tour détecté pour {path.name}")
        return pd.concat(frames, ignore_index=True)

    return _process(df_raw, meta)


def validate_consistency(df: pd.DataFrame, *, tolerance: float = 0.02) -> Dict[str, pd.DataFrame]:
    """
    Quick validation checks. Returns a dict of issues to inspect.
    """
    issues: Dict[str, pd.DataFrame] = {}

    if {"votants", "inscrits"}.issubset(df.columns):
        issues["votants_gt_inscrits"] = df[df["votants"] > df["inscrits"]]

    if {"exprimes", "blancs", "nuls", "votants"}.issubset(df.columns):
        expr_gap = df.copy()
        expr_gap["gap"] = (
            (expr_gap["exprimes"] + expr_gap["blancs"] + expr_gap["nuls"] - expr_gap["votants"])
            / expr_gap["votants"].replace(0, np.nan)
        )
        issues["exprimes_balance_off"] = expr_gap[expr_gap["gap"].abs() > tolerance]

    if {"code_bv", "type_scrutin", "tour", "exprimes", "voix"}.issubset(df.columns):
        sums = df.groupby(["code_bv", "type_scrutin", "tour"], as_index=False)[["exprimes", "voix"]].sum()
        sums["gap"] = (sums["voix"] - sums["exprimes"]) / sums["exprimes"].replace(0, np.nan)
        issues["sum_voix_vs_exprimes"] = sums[sums["gap"].abs() > tolerance]

    return issues
