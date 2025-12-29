from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import warnings

from .constants import CANDIDATE_CATEGORIES
from .pipeline import normalize_bloc

try:
    from numpy import RankWarning as NP_RANK_WARNING  # type: ignore[attr-defined]
except Exception:
    class NP_RANK_WARNING(UserWarning):
        pass


@dataclass
class PredictionResult:
    category: str
    predicted_share: float
    predicted_count: int


@dataclass
class PredictionSummary:
    bloc_predictions: list[PredictionResult]
    inscrits: Optional[int]
    votants: Optional[int]
    blancs: Optional[int]
    nuls: Optional[int]
    abstention: Optional[int]
    exprimes: Optional[int]


DISPLAY_BLOC_ORDER = [
    "extreme_gauche",
    "gauche_dure",
    "gauche_modere",
    "centre",
    "droite_modere",
    "droite_dure",
    "extreme_droite",
]
EXTRA_CATEGORIES = ["blancs", "nuls", "abstention"]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _last_share(df: pd.DataFrame, bloc: str, *, election: Optional[str] = None, year: Optional[int] = None) -> Optional[float]:
    subset = df[df["bloc"] == bloc]
    if election:
        subset = subset[subset["type_scrutin"] == election]
    if year is not None:
        subset = subset[subset["annee"] == year]
    if subset.empty:
        return None
    valid = subset.sort_values("date_scrutin")["part_bloc"].dropna()
    if valid.empty:
        return None
    return valid.iloc[-1]  # type: ignore[index]


def _last_value(series: pd.Series) -> Optional[float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _project_share(series: pd.Series, years: pd.Series, target_year: int) -> Optional[float]:
    df = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce"), "year": pd.to_numeric(years, errors="coerce")})
    df = df.dropna()
    if df.empty:
        return None
    if len(df["year"].unique()) >= 2 and len(df) >= 2:
        # Guard against poorly conditioned fits on tiny samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NP_RANK_WARNING)
            try:
                slope, intercept = np.polyfit(df["year"], df["value"], 1)
                projected = slope * target_year + intercept
            except Exception:
                projected = df["value"].iloc[-1]
    else:
        projected = df["value"].iloc[-1]
    return _clip01(float(projected))


def _project_rate(
    series: pd.Series,
    years: pd.Series,
    target_year: int,
    *,
    min_points_trend: int = 3,
    clamp_to_observed: bool = True,
) -> Optional[float]:
    df = pd.DataFrame(
        {"value": pd.to_numeric(series, errors="coerce"), "year": pd.to_numeric(years, errors="coerce")}
    ).dropna()
    if df.empty:
        return None
    values = df["value"].to_numpy()
    years_arr = df["year"].to_numpy()
    if len(set(years_arr)) >= min_points_trend and len(df) >= min_points_trend:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NP_RANK_WARNING)
            try:
                slope, intercept = np.polyfit(years_arr, values, 1)
                projected = slope * target_year + intercept
            except Exception:
                projected = values[-1]
    else:
        projected = values[-1]
    if clamp_to_observed and len(values):
        projected = min(max(projected, float(np.nanmin(values))), float(np.nanmax(values)))
    return _clip01(float(projected))


def _allocate_counts(shares: list[float], total: int) -> list[int]:
    if total <= 0 or not shares:
        return [0 for _ in shares]
    arr = np.clip(np.asarray(shares, dtype=float), 0, None)
    if arr.sum() == 0:
        return [0 for _ in shares]
    arr = arr / arr.sum()
    raw = arr * total
    floors = np.floor(raw)
    remainder = int(total - floors.sum())
    if remainder > 0:
        order = np.argsort(-(raw - floors))
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors.astype(int).tolist()


def compute_predictions(
    history: pd.DataFrame,
    *,
    target_election: str = "municipales",
    target_year: int = 2026,
    inscrits_override: Optional[float] = None,
) -> PredictionSummary:
    if history.empty:
        return PredictionSummary([], None, None, None, None, None, None)

    df = history.copy()
    target_election = str(target_election).strip().lower()
    df["bloc"] = df["bloc"].apply(normalize_bloc)
    if "type_scrutin" in df.columns:
        df["type_scrutin"] = df["type_scrutin"].astype(str).str.strip().str.lower()
    # Coerce numeric and infer exprimes when missing from the sum of voix_bloc
    for col in ["voix_bloc", "exprimes", "inscrits", "votants", "blancs", "nuls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["inscrits", "votants", "blancs", "nuls"]:
        if col not in df.columns:
            df[col] = np.nan
    if "exprimes" in df.columns:
        sum_voix = df.groupby(["code_bv", "date_scrutin"])["voix_bloc"].transform("sum")
        df["exprimes"] = df["exprimes"].fillna(sum_voix)
        df.loc[df["exprimes"] == 0, "exprimes"] = sum_voix
    if "part_bloc" not in df.columns or df["part_bloc"].isna().all():
        df["part_bloc"] = df["voix_bloc"] / df["exprimes"]
    df["part_bloc"] = pd.to_numeric(df["part_bloc"], errors="coerce").clip(upper=1)
    df = df.dropna(subset=["bloc"])

    bloc_order = [b for b in DISPLAY_BLOC_ORDER if b in CANDIDATE_CATEGORIES]
    raw_shares: dict[str, float] = {}
    for bloc in bloc_order:
        bloc_hist = df[df["bloc"] == bloc].sort_values("date_scrutin")
        last_overall = _last_share(bloc_hist, bloc)
        base_series = bloc_hist["part_bloc"]
        base_years = bloc_hist["annee"]
        if not bloc_hist.empty and target_election in bloc_hist["type_scrutin"].values:
            base_series = bloc_hist[bloc_hist["type_scrutin"] == target_election]["part_bloc"]
            base_years = bloc_hist[bloc_hist["type_scrutin"] == target_election]["annee"]

        projected = _project_share(base_series, base_years, target_year)
        if projected is None and last_overall is not None:
            projected = last_overall
        predicted = _clip01(projected or 0.0)
        raw_shares[bloc] = predicted

    share_values = np.array([raw_shares.get(b, 0.0) for b in bloc_order], dtype=float)
    share_sum = share_values.sum()
    if share_sum > 0:
        share_values = share_values / share_sum
    else:
        share_values = np.zeros_like(share_values)

    event_cols = [col for col in ["code_bv", "date_scrutin", "type_scrutin", "tour", "annee"] if col in df.columns]
    event_df = df.groupby(event_cols, as_index=False).agg(
        inscrits=("inscrits", "max"),
        votants=("votants", "max"),
        blancs=("blancs", "max"),
        nuls=("nuls", "max"),
    )
    if "date_scrutin" in event_df.columns:
        event_df = event_df.sort_values("date_scrutin")
    if "type_scrutin" not in event_df.columns:
        event_df["type_scrutin"] = ""
    if "annee" not in event_df.columns:
        if "date_scrutin" in event_df.columns:
            event_df["annee"] = pd.to_datetime(event_df["date_scrutin"], errors="coerce").dt.year
        else:
            event_df["annee"] = np.nan
    base_inscrits = event_df["inscrits"].replace(0, pd.NA)
    event_df["taux_participation"] = event_df["votants"] / base_inscrits
    event_df["taux_blancs"] = event_df["blancs"] / base_inscrits
    event_df["taux_nuls"] = event_df["nuls"] / base_inscrits

    def _select_series(col: str) -> tuple[pd.Series, pd.Series]:
        scoped = event_df
        if "tour" in event_df.columns:
            round1 = event_df[event_df["tour"] == 1]
            if not round1.empty:
                scoped = round1
        if not scoped.empty and target_election in scoped["type_scrutin"].values:
            mask = scoped["type_scrutin"] == target_election
            return scoped.loc[mask, col], scoped.loc[mask, "annee"]
        return scoped[col], scoped["annee"]

    turnout_series, turnout_years = _select_series("taux_participation")
    blancs_series, blancs_years = _select_series("taux_blancs")
    nuls_series, nuls_years = _select_series("taux_nuls")

    taux_participation = _project_rate(turnout_series, turnout_years, target_year)
    taux_blancs = _project_rate(blancs_series, blancs_years, target_year)
    taux_nuls = _project_rate(nuls_series, nuls_years, target_year)

    inscrits_used = None
    if inscrits_override is not None:
        try:
            value = float(inscrits_override)
            if value > 0:
                inscrits_used = value
        except (TypeError, ValueError):
            inscrits_used = None
    if inscrits_used is None:
        inscrits_used = _last_value(event_df["inscrits"])
    if inscrits_used is None:
        return PredictionSummary([], None, None, None, None, None, None)

    if taux_participation is None:
        taux_participation = 0.0
    if taux_blancs is None:
        taux_blancs = 0.0
    if taux_nuls is None:
        taux_nuls = 0.0

    if taux_blancs + taux_nuls > taux_participation and (taux_blancs + taux_nuls) > 0:
        scale = taux_participation / (taux_blancs + taux_nuls)
        taux_blancs *= scale
        taux_nuls *= scale

    inscrits_total = int(round(inscrits_used))
    votants_total = int(round(inscrits_total * taux_participation))
    blancs_total = int(round(inscrits_total * taux_blancs))
    nuls_total = int(round(inscrits_total * taux_nuls))
    if blancs_total + nuls_total > votants_total and (blancs_total + nuls_total) > 0:
        scale = votants_total / (blancs_total + nuls_total)
        blancs_total = int(round(blancs_total * scale))
        nuls_total = int(round(nuls_total * scale))
    exprimes_total = max(0, votants_total - blancs_total - nuls_total)
    abstention_total = max(0, inscrits_total - votants_total)

    bloc_counts = _allocate_counts(share_values.tolist(), exprimes_total)
    bloc_predictions: list[PredictionResult] = []
    for bloc, share, count in zip(bloc_order, share_values.tolist(), bloc_counts):
        bloc_predictions.append(
            PredictionResult(
                category=bloc,
                predicted_share=float(share),
                predicted_count=int(count),
            )
        )

    return PredictionSummary(
        bloc_predictions=bloc_predictions,
        inscrits=inscrits_total,
        votants=votants_total,
        blancs=blancs_total,
        nuls=nuls_total,
        abstention=abstention_total,
        exprimes=exprimes_total,
    )


def predictions_as_dataframe(summary: PredictionSummary) -> pd.DataFrame:
    if summary is None or not summary.bloc_predictions:
        return pd.DataFrame(columns=["categorie", "nombre"])
    rows = []
    pred_map = {item.category: item for item in summary.bloc_predictions}
    for bloc in [b for b in DISPLAY_BLOC_ORDER if b in pred_map]:
        item = pred_map[bloc]
        rows.append({"categorie": bloc, "nombre": int(item.predicted_count)})
    if summary.blancs is not None:
        rows.append({"categorie": "blancs", "nombre": int(summary.blancs)})
    if summary.nuls is not None:
        rows.append({"categorie": "nuls", "nombre": int(summary.nuls)})
    if summary.abstention is not None:
        rows.append({"categorie": "abstention", "nombre": int(summary.abstention)})
    return pd.DataFrame(rows)


__all__ = ["compute_predictions", "predictions_as_dataframe", "PredictionResult", "PredictionSummary"]
