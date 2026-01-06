from __future__ import annotations

import base64
import io
import json
import logging
import re
import warnings
from html import escape
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.constants import CANDIDATE_CATEGORIES
from src.db.schema import get_engine
from src.features.build_features import (
    aggregate_by_event,
    compute_national_reference,
    expand_by_category,
    load_elections_long,
    load_mapping,
)

LOGGER = logging.getLogger(__name__)
COMMUNE_CODE_SETE = "34301"
MODEL_DIR = Path("models")
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.json"
RESIDUAL_INTERVALS_PATH = Path("reports/residual_intervals.json")
GEO_DIR = Path("data/geo")
DEFAULT_TARGETS = [
    ("municipales", 2026),
    ("legislatives", 2027),
    ("presidentielles", 2027),
]
FEATURE_CACHE: Dict[Tuple[str, int], Tuple[pd.DataFrame, Dict[str, Dict[Tuple[str, str], float]]]] = {}
ELECTION_KEY_SEP = "|"
ELECTION_TYPE_LABELS = {
    "municipales": "Municipales",
    "legislatives": "Législatives",
    "presidentielles": "Présidentielles",
    "europeennes": "Européennes",
    "regionales": "Régionales",
    "departementales": "Départementales",
}
HISTORY_OUTPUT_COLUMNS = ["categorie", "score_%"]
PREDICTION_OUTPUT_COLUMNS = ["categorie", "nombre"]
INTERVAL_OUTPUT_COLUMNS = ["categorie", "baseline_%", "min_%", "max_%", "baseline", "min", "max"]
SIM_OUTPUT_COLUMNS = ["categorie", "baseline", "apres_transfert", "delta"]
OPPORTUNITY_OUTPUT_COLUMNS = [
    "bureau",
    "gain_cible",
    "score_base",
    "score_apres",
    "top_base",
    "top_apres",
    "bascule",
]
DISPLAY_CATEGORY_ORDER = [
    "extreme_gauche",
    "gauche_dure",
    "gauche_modere",
    "centre",
    "droite_modere",
    "droite_dure",
    "extreme_droite",
]
PREDICTION_CATEGORY_ORDER = DISPLAY_CATEGORY_ORDER + ["blancs", "nuls", "abstention"]
DISPLAY_CATEGORY_LABELS = {
    "extreme_gauche": "extrême-gauche",
    "gauche_dure": "gauche dure",
    "gauche_modere": "gauche modérée",
    "centre": "centre",
    "droite_modere": "droite modérée",
    "droite_dure": "droite dure",
    "extreme_droite": "extrême-droite",
    "blancs": "blancs",
    "nuls": "nuls",
    "abstention": "abstentions",
}
DISPLAY_CATEGORY_COLORS = {
    "extreme_gauche": "#7f1d1d",
    "gauche_dure": "#dc2626",
    "gauche_modere": "#f472b6",
    "centre": "#facc15",
    "droite_modere": "#60a5fa",
    "droite_dure": "#1e3a8a",
    "extreme_droite": "#111827",
}
EXTRA_CATEGORY_COLORS = {
    "blancs": "#e5e7eb",
    "nuls": "#9ca3af",
    "abstention": "#6b7280",
}
DISPLAY_LABEL_COLORS = {
    DISPLAY_CATEGORY_LABELS[key]: color for key, color in DISPLAY_CATEGORY_COLORS.items()
}
DISPLAY_LABEL_COLORS.update(
    {DISPLAY_CATEGORY_LABELS[key]: color for key, color in EXTRA_CATEGORY_COLORS.items()}
)
CATEGORY_LABEL_TO_KEY = {label: key for key, label in DISPLAY_CATEGORY_LABELS.items()}
TRANSFER_CATEGORY_LABELS = [DISPLAY_CATEGORY_LABELS[key] for key in PREDICTION_CATEGORY_ORDER]
DEFAULT_RESIDUAL_SPREAD = 0.03
INTERVAL_BANDS = {
    "80% (p10-p90)": ("q10", "q90"),
    "90% (p05-p95)": ("q05", "q95"),
}
NEUTRAL_MARGIN_SHARE = 0.10
TYPE_HISTORY_BLEND = {
    "presidentielles": 0.4,
    "legislatives": 0.35,
    "europeennes": 0.3,
    "regionales": 0.3,
    "departementales": 0.3,
    "municipales": 0.2,
}

try:
    from numpy import RankWarning as NP_RANK_WARNING  # type: ignore[attr-defined]
except Exception:
    class NP_RANK_WARNING(UserWarning):
        pass


def ordered_categories() -> list[str]:
    return [cat for cat in DISPLAY_CATEGORY_ORDER if cat in CANDIDATE_CATEGORIES]


def load_residual_intervals(path: Path = RESIDUAL_INTERVALS_PATH) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def get_interval_bounds(
    residuals: Dict[str, Dict[str, float]],
    category: str,
    band_label: str,
) -> Tuple[float, float]:
    keys = INTERVAL_BANDS.get(band_label, ("q10", "q90"))
    cat_resid = residuals.get(category, {})
    low = cat_resid.get(keys[0])
    high = cat_resid.get(keys[1])
    if low is None or high is None:
        return -DEFAULT_RESIDUAL_SPREAD, DEFAULT_RESIDUAL_SPREAD
    return float(low), float(high)


def build_interval_table(
    shares_by_cat: Dict[str, float],
    exprimes_total: int,
    residuals: Dict[str, Dict[str, float]],
    band_label: str,
) -> pd.DataFrame:
    rows = []
    for cat in ordered_categories():
        share = float(shares_by_cat.get(cat, 0.0))
        low_resid, high_resid = get_interval_bounds(residuals, cat, band_label)
        share_low = float(np.clip(share + low_resid, 0.0, 1.0))
        share_high = float(np.clip(share + high_resid, 0.0, 1.0))
        count = int(round(share * exprimes_total))
        count_low = int(round(share_low * exprimes_total))
        count_high = int(round(share_high * exprimes_total))
        if count_low > count_high:
            count_low, count_high = count_high, count_low
            share_low, share_high = share_high, share_low
        rows.append(
            {
                "categorie": DISPLAY_CATEGORY_LABELS.get(cat, cat),
                "baseline_%": round(share * 100, 1),
                "min_%": round(share_low * 100, 1),
                "max_%": round(share_high * 100, 1),
                "baseline": count,
                "min": count_low,
                "max": count_high,
            }
        )
    return pd.DataFrame(rows, columns=INTERVAL_OUTPUT_COLUMNS)


def build_interval_chart(
    df: pd.DataFrame,
    *,
    value_col: str = "baseline",
    low_col: str = "min",
    high_col: str = "max",
    color_map: Dict[str, str] | None = None,
    ylabel: str = "Nombre d'électeurs",
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if df.empty or value_col not in df.columns:
        return None
    labels = df["categorie"].astype(str).tolist()
    values = df[value_col].astype(float).to_numpy()
    low_vals = df[low_col].astype(float).to_numpy()
    high_vals = df[high_col].astype(float).to_numpy()
    lower_err = np.maximum(0.0, values - low_vals)
    upper_err = np.maximum(0.0, high_vals - values)
    yerr = np.vstack([lower_err, upper_err])
    colors = [color_map.get(label, "#3b82f6") for label in labels] if color_map else "#3b82f6"
    plt.figure(figsize=(6, 3))
    plt.bar(labels, values, color=colors, yerr=yerr, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    return plt


def blend_with_type_history(
    preds_by_cat: Dict[str, float],
    row: pd.Series,
    target_type: str,
) -> Dict[str, float]:
    base_weight = TYPE_HISTORY_BLEND.get(str(target_type).lower(), 0.0)
    if base_weight <= 0:
        return preds_by_cat
    available = 0
    hist_vals: Dict[str, float | None] = {}
    for cat in CANDIDATE_CATEGORIES:
        val = row.get(f"prev_share_type_lag1_{cat}")
        if val is not None and not pd.isna(val):
            hist_vals[cat] = float(val)
            available += 1
        else:
            hist_vals[cat] = None
    if available == 0:
        return preds_by_cat
    weight = base_weight * (available / len(CANDIDATE_CATEGORIES))
    blended: Dict[str, float] = {}
    for cat in CANDIDATE_CATEGORIES:
        base = float(preds_by_cat.get(cat, 0.0))
        hist = hist_vals.get(cat)
        if hist is None:
            blended[cat] = base
        else:
            blended[cat] = (1 - weight) * base + weight * hist
    total = sum(blended.values())
    if total > 0:
        for cat in blended:
            blended[cat] /= total
    return blended


def apply_transfers(
    counts: Dict[str, int],
    total_inscrits: int,
    transfers: list[Tuple[str, str, float]],
) -> Dict[str, int]:
    updated = {key: int(value) for key, value in counts.items()}
    for source, target, delta_pct in transfers:
        if delta_pct <= 0 or source == target:
            continue
        delta_count = int(round(total_inscrits * float(delta_pct) / 100.0))
        if delta_count <= 0:
            continue
        available = max(0, int(updated.get(source, 0)))
        moved = min(available, delta_count)
        updated[source] = available - moved
        updated[target] = int(updated.get(target, 0)) + moved
    return updated


def build_simulation_table(
    baseline: Dict[str, int],
    updated: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    for cat in PREDICTION_CATEGORY_ORDER:
        base = int(baseline.get(cat, 0))
        new = int(updated.get(cat, 0))
        rows.append(
            {
                "categorie": DISPLAY_CATEGORY_LABELS.get(cat, cat),
                "baseline": base,
                "apres_transfert": new,
                "delta": new - base,
            }
        )
    return pd.DataFrame(rows, columns=SIM_OUTPUT_COLUMNS)


def load_geojson_features(geo_dir: Path = GEO_DIR) -> list[dict]:
    if not geo_dir.exists():
        return []
    paths = sorted(geo_dir.glob("*.geojson")) + sorted(geo_dir.glob("*.json"))
    features: list[dict] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            features.extend(payload.get("features", []))
    return features


def extract_bureau_number(label: str | None) -> int | None:
    if not label:
        return None
    match = re.search(r"(\d+)", str(label))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def match_bureau_code(commune_code: str, bureau_num: int, available_codes: set[str]) -> str:
    padded = str(bureau_num).zfill(4)
    candidates = [f"{commune_code}-{padded}", f"{commune_code}{padded}"]
    for candidate in candidates:
        if candidate in available_codes:
            return candidate
    return candidates[-1]


def _iter_coords(geom: dict) -> list[Tuple[float, float]]:
    coords = []
    geom_type = geom.get("type")
    if geom_type == "Polygon":
        for ring in geom.get("coordinates", []):
            coords.extend([(lon, lat) for lon, lat in ring])
    elif geom_type == "MultiPolygon":
        for polygon in geom.get("coordinates", []):
            for ring in polygon:
                coords.extend([(lon, lat) for lon, lat in ring])
    return coords


def geojson_bounds(features: list[dict]) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
    lons = []
    lats = []
    for feature in features:
        geom = feature.get("geometry") or {}
        for lon, lat in _iter_coords(geom):
            lons.append(lon)
            lats.append(lat)
    if not lons or not lats:
        return None
    return (min(lats), min(lons)), (max(lats), max(lons))


def build_prediction_table_from_counts(counts_by_cat: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for cat in ordered_categories():
        rows.append({"categorie": DISPLAY_CATEGORY_LABELS.get(cat, cat), "nombre": int(counts_by_cat.get(cat, 0))})
    for extra in ["blancs", "nuls", "abstention"]:
        rows.append(
            {
                "categorie": DISPLAY_CATEGORY_LABELS[extra],
                "nombre": int(counts_by_cat.get(extra, 0)),
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_OUTPUT_COLUMNS)


def chart_base64_from_df(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    color_map: Dict[str, str],
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if df.empty or value_col not in df.columns:
        return None
    labels = df["categorie"].astype(str).tolist()
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0).tolist()
    colors = [color_map.get(label, "#3b82f6") for label in labels]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_map_popup_html(
    bureau_label: str,
    table_df: pd.DataFrame,
    chart_b64: str | None,
    meta: str | None,
) -> str:
    title_html = f"<strong>{escape(bureau_label)}</strong>"
    meta_html = f"<div style='margin:4px 0;'>{escape(meta)}</div>" if meta else ""
    table_html = table_df.to_html(index=False, border=0)
    img_html = ""
    if chart_b64:
        img_html = (
            "<div style='margin-top:6px;'>"
            f"<img src='data:image/png;base64,{chart_b64}' style='width:320px;height:auto;'/>"
            "</div>"
        )
    return f"<div style='font-size:12px;'>{title_html}{meta_html}{table_html}{img_html}</div>"


def build_map_legend_html() -> str:
    parts = []
    for key in DISPLAY_CATEGORY_ORDER:
        label = DISPLAY_CATEGORY_LABELS.get(key, key)
        color = DISPLAY_CATEGORY_COLORS.get(key, "#9ca3af")
        parts.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:10px;'>"
            f"<span style='width:12px;height:12px;background:{color};display:inline-block;margin-right:6px;border:1px solid #111827;'></span>"
            f"{escape(label)}</span>"
        )
    parts.append(
        "<span style='display:inline-flex;align-items:center;margin-right:10px;'>"
        "<span style='width:12px;height:12px;background:#ffffff;display:inline-block;margin-right:6px;border:1px solid #111827;'></span>"
        "écart gauche/droite ≤ 10%</span>"
    )
    parts.append(
        "<span style='display:inline-flex;align-items:center;margin-right:10px;'>"
        "<span style='width:12px;height:12px;background:#9ca3af;display:inline-block;margin-right:6px;border:1px solid #111827;'></span>"
        "données indisponibles</span>"
    )
    parts.append("<span style='font-size:12px;color:#6b7280;'>abstention non utilisée pour la couleur</span>")
    return "<div style='margin-bottom:8px;'>" + " ".join(parts) + "</div>"


def build_bureau_map_html(
    backend: "PredictorBackend",
    target_type: str,
    target_year: int,
) -> str:
    try:
        import folium
    except Exception:
        return "<p>Folium n'est pas disponible. Installe-le via requirements.txt.</p>"

    features = load_geojson_features()
    if not features:
        return "<p>Aucune geojson trouvée dans data/geo.</p>"

    bounds = geojson_bounds(features)
    if bounds is None:
        return "<p>Impossible de calculer l'emprise de la carte.</p>"
    (min_lat, min_lon), (max_lat, max_lon) = bounds
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    fmap = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    available_codes = set(backend.available_bureaux())
    for feature in features:
        props = feature.get("properties", {})
        label = props.get("name") or "Bureau"
        bureau_num = extract_bureau_number(label)
        if bureau_num is None:
            code_bv = None
        else:
            code_bv = match_bureau_code(COMMUNE_CODE_SETE, bureau_num, available_codes)

        fill_color = "#9ca3af"
        popup_html = None
        if code_bv is not None:
            details, _, meta = backend.predict_bureau_details(code_bv, target_type, target_year)
            if details is not None:
                shares = details["shares_by_cat"]
                left_share = float(shares.get("gauche_dure", 0.0) + shares.get("gauche_modere", 0.0))
                right_share = float(shares.get("droite_dure", 0.0) + shares.get("droite_modere", 0.0))
                if abs(left_share - right_share) <= NEUTRAL_MARGIN_SHARE:
                    fill_color = "#ffffff"
                else:
                    winner = max(shares, key=shares.get)
                    fill_color = DISPLAY_CATEGORY_COLORS.get(winner, fill_color)

                table_df = build_prediction_table_from_counts(details["counts"])
                chart_b64 = chart_base64_from_df(
                    table_df,
                    value_col="nombre",
                    ylabel="Nombre d'electeurs",
                    color_map=DISPLAY_LABEL_COLORS,
                )
                popup_html = build_map_popup_html(str(label), table_df, chart_b64, meta)

        def _style(_: dict, color=fill_color):
            return {
                "fillColor": color,
                "color": "#111827",
                "weight": 1,
                "fillOpacity": 0.6,
            }

        geo = folium.GeoJson(feature, style_function=_style)
        if popup_html:
            geo.add_child(folium.Popup(popup_html, max_width=450))
        geo.add_child(folium.Tooltip(str(label)))
        geo.add_to(fmap)

    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return fmap._repr_html_()


def _project_rate(
    series: pd.Series,
    years: pd.Series,
    target_year: int,
    *,
    min_points_trend: int = 3,
    clamp_to_observed: bool = True,
) -> float | None:
    df = pd.DataFrame(
        {
            "value": pd.to_numeric(series, errors="coerce"),
            "year": pd.to_numeric(years, errors="coerce"),
        }
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
    return float(min(1.0, max(0.0, projected)))


def _allocate_counts(shares: np.ndarray, total: int) -> np.ndarray:
    if total <= 0 or shares.size == 0:
        return np.zeros_like(shares, dtype=int)
    shares = np.clip(shares, 0, None)
    if shares.sum() == 0:
        return np.zeros_like(shares, dtype=int)
    shares = shares / shares.sum()
    raw = shares * total
    floors = np.floor(raw)
    remainder = int(total - floors.sum())
    if remainder > 0:
        order = np.argsort(-(raw - floors))
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors.astype(int)


def load_bureau_event_stats(commune_code: str) -> pd.DataFrame:
    candidates = [
        Path("data/processed/elections_blocs.parquet"),
        Path("data/processed/elections_blocs.csv"),
        Path("data/interim/elections_long.parquet"),
        Path("data/interim/elections_long.csv"),
    ]
    df = pd.DataFrame()
    best = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep=";")
        if df.empty:
            continue
        if "type_scrutin" not in df.columns and "election_type" in df.columns:
            df["type_scrutin"] = df["election_type"]
        if "annee" not in df.columns and "election_year" in df.columns:
            df["annee"] = df["election_year"]
        if "tour" not in df.columns and "round" in df.columns:
            df["tour"] = df["round"]
        df["date_scrutin"] = pd.to_datetime(df.get("date_scrutin"), errors="coerce")
        for col in ["inscrits", "votants", "blancs", "nuls"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        if "code_commune" in df.columns:
            df["code_commune"] = df["code_commune"].astype(str)
            df = df[df["code_commune"] == str(commune_code)]
        else:
            df = df[df["code_bv"].astype(str).str.startswith(str(commune_code))]
        df = df.dropna(subset=["code_bv"])
        if df.empty:
            continue
        has_blancs = df["blancs"].notna().any() or df["nuls"].notna().any()
        if has_blancs:
            best = df
            break
        if best.empty:
            best = df
    df = best
    if df.empty:
        return df
    group_cols = [col for col in ["code_bv", "type_scrutin", "annee", "tour", "date_scrutin"] if col in df.columns]
    agg = df.groupby(group_cols, as_index=False).agg(
        inscrits=("inscrits", "max"),
        votants=("votants", "max"),
        blancs=("blancs", "max"),
        nuls=("nuls", "max"),
    )
    if "date_scrutin" in agg.columns:
        agg = agg.sort_values("date_scrutin")
    agg["election_type"] = agg.get("type_scrutin")
    agg["election_type"] = agg["election_type"].astype("string").str.strip().str.lower()
    agg["election_year"] = pd.to_numeric(agg.get("annee"), errors="coerce")
    agg["round"] = pd.to_numeric(agg.get("tour"), errors="coerce").fillna(1).astype(int)
    base_inscrits = agg["inscrits"].replace(0, np.nan)
    agg["turnout_pct"] = agg["votants"] / base_inscrits
    agg["blancs_pct"] = agg["blancs"] / base_inscrits
    agg["nuls_pct"] = agg["nuls"] / base_inscrits
    return agg[
        [
            "code_bv",
            "election_type",
            "election_year",
            "round",
            "date_scrutin",
            "inscrits",
            "votants",
            "blancs",
            "nuls",
            "turnout_pct",
            "blancs_pct",
            "nuls_pct",
        ]
    ]


def load_commune_event_stats(commune_code: str) -> pd.DataFrame:
    candidates = [
        Path("data/processed/commune_event_stats.parquet"),
        Path("data/processed/commune_event_stats.csv"),
    ]
    df = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep=";")
        if not df.empty:
            break
    if df.empty:
        return df
    if "type_scrutin" not in df.columns and "election_type" in df.columns:
        df["type_scrutin"] = df["election_type"]
    if "annee" not in df.columns and "election_year" in df.columns:
        df["annee"] = df["election_year"]
    if "tour" not in df.columns and "round" in df.columns:
        df["tour"] = df["round"]
    df["date_scrutin"] = pd.to_datetime(df.get("date_scrutin"), errors="coerce")
    for col in ["inscrits", "votants", "blancs", "nuls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    if "code_commune" in df.columns:
        df["code_commune"] = df["code_commune"].astype(str)
        df = df[df["code_commune"] == str(commune_code)]
    else:
        return pd.DataFrame()
    if df.empty:
        return df
    base_inscrits = df["inscrits"].replace(0, np.nan)
    if "turnout_pct" not in df.columns:
        df["turnout_pct"] = df["votants"] / base_inscrits
    if "blancs_pct" not in df.columns:
        df["blancs_pct"] = df["blancs"] / base_inscrits
    if "nuls_pct" not in df.columns:
        df["nuls_pct"] = df["nuls"] / base_inscrits
    df["election_type"] = df["type_scrutin"].astype("string").str.strip().str.lower()
    df["election_year"] = pd.to_numeric(df.get("annee"), errors="coerce")
    df["round"] = pd.to_numeric(df.get("tour"), errors="coerce").fillna(1).astype(int)
    return df[
        [
            "code_commune",
            "election_type",
            "election_year",
            "round",
            "date_scrutin",
            "inscrits",
            "votants",
            "blancs",
            "nuls",
            "turnout_pct",
            "blancs_pct",
            "nuls_pct",
        ]
    ]


def format_backend_label(backend_kind: str) -> str:
    return "PostgreSQL" if backend_kind == "postgres" else "fichiers locaux"


def format_election_type_label(election_type: str) -> str:
    label = ELECTION_TYPE_LABELS.get(election_type)
    if label:
        return label
    return str(election_type).replace("_", " ").title()


def format_election_label(
    election_type: str,
    election_year: int,
    round_num: int,
    date_scrutin: pd.Timestamp | None = None,
) -> str:
    base = f"{format_election_type_label(election_type)} {election_year} - Tour {round_num}"
    if date_scrutin is None or pd.isna(date_scrutin):
        return base
    date_value = pd.to_datetime(date_scrutin).date().isoformat()
    return f"{base} ({date_value})"


def format_election_key(election_type: str, election_year: int, round_num: int) -> str:
    return f"{election_type}{ELECTION_KEY_SEP}{election_year}{ELECTION_KEY_SEP}{round_num}"


def parse_election_key(key: str) -> Tuple[str, int, int]:
    parts = key.split(ELECTION_KEY_SEP)
    if len(parts) != 3:
        raise ValueError(f"Clé d'élection invalide: {key!r}")
    return parts[0], int(parts[1]), int(parts[2])


def format_bureau_label(code_bv: str, bureau_label: str | None) -> str:
    code = str(code_bv)
    suffix = code.split("-")[-1] if "-" in code else code
    if bureau_label is not None and not pd.isna(bureau_label):
        label = str(bureau_label).strip()
        if label and label != code:
            return f"{label} ({code})"
    return f"Bureau {suffix} ({code})"


def build_bureau_choices(history: pd.DataFrame) -> list[tuple[str, str]]:
    if history.empty:
        return []
    if "bureau_label" in history.columns:
        label_map = (
            history[["code_bv", "bureau_label"]]
            .dropna(subset=["code_bv"])
            .drop_duplicates()
            .sort_values("code_bv")
            .groupby("code_bv", as_index=False)["bureau_label"]
            .first()
        )
        return [
            (format_bureau_label(row.code_bv, row.bureau_label), row.code_bv)
            for row in label_map.itertuples(index=False)
        ]
    codes = sorted(history["code_bv"].dropna().unique().tolist())
    return [(format_bureau_label(code, None), code) for code in codes]


def build_history_choices(history: pd.DataFrame) -> list[tuple[str, str]]:
    if history.empty:
        return []
    events = (
        history[["election_type", "election_year", "round", "date_scrutin"]]
        .dropna(subset=["election_type", "election_year", "round"])
        .drop_duplicates()
        .groupby(["election_type", "election_year", "round"], as_index=False)
        .agg(date_scrutin=("date_scrutin", "min"))
        .sort_values(["election_year", "election_type", "round"])
    )
    return [
        (
            format_election_label(
                row.election_type,
                int(row.election_year),
                int(row.round),
                row.date_scrutin,
            ),
            format_election_key(row.election_type, int(row.election_year), int(row.round)),
        )
        for row in events.itertuples(index=False)
    ]


def clean_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    clean = history.copy()
    clean["code_bv"] = clean["code_bv"].astype("string").str.strip()
    clean["election_type"] = clean["election_type"].astype("string").str.strip().str.lower()
    clean["category"] = clean["category"].astype("string").str.strip().str.lower()
    if "bureau_label" in clean.columns:
        clean["bureau_label"] = clean["bureau_label"].astype("string").str.strip()
    clean["election_year"] = pd.to_numeric(clean["election_year"], errors="coerce")
    clean["round"] = pd.to_numeric(clean["round"], errors="coerce").fillna(1)
    clean["date_scrutin"] = pd.to_datetime(clean["date_scrutin"], errors="coerce")
    for col in ["share", "share_nat", "turnout_pct"]:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").clip(lower=0, upper=1)
    clean = clean.dropna(subset=["code_bv", "election_type", "election_year", "round", "category"])
    clean["election_year"] = clean["election_year"].astype(int)
    clean["round"] = clean["round"].astype(int)
    clean = clean[clean["category"].isin(CANDIDATE_CATEGORIES)]
    return clean


def prepare_history_table(history_slice: pd.DataFrame) -> pd.DataFrame:
    if history_slice.empty:
        return pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
    grouped = history_slice.groupby("category", as_index=False).agg(share=("share", "sum"))
    clean = pd.DataFrame({"category": ordered_categories()}).merge(grouped, on="category", how="left")
    clean["share"] = pd.to_numeric(clean["share"], errors="coerce").fillna(0).clip(lower=0, upper=1)
    clean["score_%"] = (clean["share"] * 100).round(1)
    clean["categorie"] = clean["category"].map(DISPLAY_CATEGORY_LABELS).fillna(clean["category"])
    return clean[HISTORY_OUTPUT_COLUMNS]


def format_history_meta(history_slice: pd.DataFrame) -> str:
    if history_slice.empty:
        return ""
    parts = []
    dates = history_slice["date_scrutin"].dropna()
    if not dates.empty:
        date_value = pd.to_datetime(dates.iloc[0]).date().isoformat()
        parts.append(f"Date du scrutin : {date_value}")
    turnout_vals = pd.to_numeric(history_slice["turnout_pct"], errors="coerce").dropna()
    if not turnout_vals.empty:
        parts.append(f"Participation : {turnout_vals.iloc[0] * 100:.1f}%")
    return " | ".join(parts)


def _code_bv_full(commune_code: str, bureau_code: str) -> str:
    bureau_code = str(bureau_code).zfill(4)
    return f"{commune_code}-{bureau_code}"


def load_history_from_db(commune_code: str) -> pd.DataFrame:
    engine = get_engine()
    query = sa.text(
        """
        select cm.insee_code as commune_code,
               b.bureau_code,
               b.bureau_label,
               e.election_type,
               e.election_year,
               coalesce(e.round, 1) as round,
               e.date as date_scrutin,
               c.name as category,
               rl.share_pct,
               rl.turnout_pct,
               rn.share_pct as share_nat
        from results_local rl
        join bureaux b on rl.bureau_id = b.id
        join communes cm on b.commune_id = cm.id
        join elections e on rl.election_id = e.id
        join categories c on rl.category_id = c.id
        left join results_national rn on rn.election_id = e.id and rn.category_id = rl.category_id
        where cm.insee_code = :commune
        """
    )
    df = pd.read_sql(query, engine, params={"commune": commune_code})
    if df.empty:
        raise RuntimeError("Aucune donnée dans la base pour la commune demandée.")
    df["code_bv"] = df.apply(lambda r: _code_bv_full(r["commune_code"], r["bureau_code"]), axis=1)
    df["date_scrutin"] = pd.to_datetime(df["date_scrutin"])
    df["share"] = pd.to_numeric(df["share_pct"], errors="coerce") / 100
    df["share_nat"] = pd.to_numeric(df["share_nat"], errors="coerce") / 100
    df["turnout_pct"] = pd.to_numeric(df["turnout_pct"], errors="coerce") / 100
    df["election_year"] = pd.to_numeric(df["election_year"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(1).astype(int)
    return df[
        [
            "commune_code",
            "code_bv",
            "bureau_label",
            "election_type",
            "election_year",
            "round",
            "date_scrutin",
            "category",
            "share",
            "share_nat",
            "turnout_pct",
        ]
    ]


def load_history_from_files(commune_code: str) -> pd.DataFrame:
    elections_long_all = load_elections_long(
        Path("data/interim/elections_long.parquet"),
        commune_code=commune_code,
    )
    mapping = load_mapping(Path("data/mapping_candidats_blocs.csv"))
    expanded_all = expand_by_category(elections_long_all, mapping)
    local_all = aggregate_by_event(expanded_all)
    nat = compute_national_reference(local_all)

    local = local_all[local_all["commune_code"] == commune_code].copy()
    local = local.merge(nat, on=["election_type", "election_year", "round", "category"], how="left")
    # Columns already in aggregate_by_event/compute_national_reference
    if "share" not in local.columns:
        raise RuntimeError("Colonne share absente du dataset local (fallback fichiers).")
    local["bureau_label"] = None
    local["share_nat"] = local.get("share_nat")
    local["turnout_pct"] = local.get("turnout_pct")
    return local.rename(
        columns={
            "annee": "election_year",
            "tour": "round",
        }
    )[
        [
            "commune_code",
            "code_bv",
            "bureau_label",
            "election_type",
            "election_year",
            "round",
            "date_scrutin",
            "category",
            "share",
            "share_nat",
            "turnout_pct",
        ]
    ]


def references_from_history(history: pd.DataFrame, target_year: int) -> Dict[str, Dict[Tuple[str, str], float]]:
    hist = history[history["election_year"] < target_year].copy()
    leg = (
        hist[hist["election_type"] == "legislatives"]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])["share"]
        .last()
    )
    mun2020 = (
        hist[(hist["election_type"] == "municipales") & (hist["election_year"] == 2020)]
        .sort_values("date_scrutin")
        .groupby(["code_bv", "category"])["share"]
        .last()
    )
    return {"leg": leg.to_dict(), "mun2020": mun2020.to_dict()}


def build_features_from_history(history: pd.DataFrame, target_type: str, target_year: int) -> pd.DataFrame:
    hist = history[history["election_year"] < target_year].copy()
    if hist.empty:
        return pd.DataFrame()
    hist = hist.sort_values("date_scrutin")
    hist["dev_to_nat"] = hist["share"] - hist["share_nat"]

    last_any_share = hist.groupby(["code_bv", "category"])["share"].last()
    last_any_dev = hist.groupby(["code_bv", "category"])["dev_to_nat"].last()
    last_type_share = (
        hist[hist["election_type"] == target_type]
        .groupby(["code_bv", "category"])["share"]
        .last()
    )
    last_type_dev = (
        hist[hist["election_type"] == target_type]
        .groupby(["code_bv", "category"])["dev_to_nat"]
        .last()
    )
    swing_any = (
        hist.groupby(["code_bv", "category"])["share"]
        .apply(lambda s: s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan)
        .rename("swing_any")
    )
    turnout_any = hist.groupby("code_bv")["turnout_pct"].last()
    turnout_type = (
        hist[hist["election_type"] == target_type]
        .groupby("code_bv")["turnout_pct"]
        .last()
    )

    bureaux = sorted(hist["code_bv"].dropna().unique())
    records = []
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


def load_model() -> Path:
    best_path = MODEL_DIR / "best_model.json"
    if best_path.exists():
        try:
            payload = json.loads(best_path.read_text())
            name = payload.get("name")
            if name:
                candidate = MODEL_DIR / f"{name}.joblib"
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    if (MODEL_DIR / "hist_gradient_boosting.joblib").exists():
        return MODEL_DIR / "hist_gradient_boosting.joblib"
    joblibs = sorted(MODEL_DIR.glob("*.joblib"))
    if not joblibs:
        raise FileNotFoundError("Aucun modèle trouvé dans models/. Lancez src/model/train.py.")
    return joblibs[0]


def load_feature_columns(path: Path, df: pd.DataFrame) -> list[str]:
    if path.exists():
        return json.loads(path.read_text())
    exclude = {"commune_code", "code_bv", "election_type", "election_year", "round", "date_scrutin"}
    return [c for c in df.columns if c not in exclude]


def format_delta(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{round(value, 1)}"


class PredictorBackend:
    def __init__(self, commune_code: str = COMMUNE_CODE_SETE):
        self.commune_code = commune_code
        self.backend = "local"
        try:
            self.history = load_history_from_db(commune_code)
            self.backend = "postgres"
            LOGGER.info("Backend PostgreSQL chargé (%s lignes)", len(self.history))
        except Exception as exc:
            LOGGER.warning("PostgreSQL indisponible (%s) -> fallback fichiers.", exc)
            self.history = load_history_from_files(commune_code)
            self.backend = "files"
            LOGGER.info("Backend fichiers chargé (%s lignes)", len(self.history))
        self.history = clean_history_frame(self.history)
        self.event_stats = load_bureau_event_stats(commune_code)
        self.commune_stats = load_commune_event_stats(commune_code)
        self.default_rates = {}
        self.default_rates_by_type: dict[str, dict[str, float]] = {}
        stats = self.commune_stats if not self.commune_stats.empty else self.event_stats
        if not stats.empty:
            if "round" in stats.columns:
                round1 = stats[stats["round"] == 1]
                if not round1.empty:
                    stats = round1
            self.default_rates = {
                "turnout_pct": float(stats["turnout_pct"].median(skipna=True)),
                "blancs_pct": float(stats["blancs_pct"].median(skipna=True)),
                "nuls_pct": float(stats["nuls_pct"].median(skipna=True)),
            }
            if "election_type" in stats.columns:
                for etype, group in stats.groupby("election_type"):
                    self.default_rates_by_type[str(etype)] = {
                        "turnout_pct": float(group["turnout_pct"].median(skipna=True)),
                        "blancs_pct": float(group["blancs_pct"].median(skipna=True)),
                        "nuls_pct": float(group["nuls_pct"].median(skipna=True)),
                    }
        self.model_path = load_model()
        self.model = joblib.load(self.model_path)
        # feature cache per target
        self.refs_cache: Dict[Tuple[str, int], Dict[str, Dict[Tuple[str, str], float]]] = {}

    def available_bureaux(self) -> list[str]:
        return sorted(self.history["code_bv"].dropna().unique().tolist())

    def available_targets(self) -> list[Tuple[str, int]]:
        existing = set()
        for row in self.history.itertuples(index=False):
            try:
                year = int(row.election_year) # type: ignore
            except Exception:
                continue
            existing.add((row.election_type, year))
        for t in DEFAULT_TARGETS:
            existing.add(t)
        return sorted(existing, key=lambda x: (x[1], x[0]))

    def _get_features_and_refs(self, target_type: str, target_year: int) -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[str, str], float]]]:
        key = (target_type, target_year)
        if key not in FEATURE_CACHE:
            feature_df = build_features_from_history(self.history, target_type, target_year)
            refs = references_from_history(self.history, target_year)
            FEATURE_CACHE[key] = (feature_df, refs)
        return FEATURE_CACHE[key]

    def predict_bureau_details(
        self,
        code_bv: str,
        target_type: str,
        target_year: int,
        inscrits_override: float | None = None,
    ) -> Tuple[Dict[str, object] | None, str, str]:
        feature_df, _ = self._get_features_and_refs(target_type, target_year)
        if feature_df.empty:
            return None, "Données insuffisantes", ""
        row = feature_df[feature_df["code_bv"] == code_bv].copy()
        if row.empty:
            return None, "Bureau non trouvé dans l'historique.", ""

        feature_cols = load_feature_columns(FEATURE_COLS_PATH, feature_df)
        missing = [c for c in feature_cols if c not in row.columns]
        for col in missing:
            row[col] = np.nan
        preds = self.model.predict(row[feature_cols])
        preds = np.clip(preds, 0, 1)
        sums = preds.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        preds = preds / sums
        preds_share = preds.flatten()

        preds_by_cat = {cat: float(preds_share[idx]) for idx, cat in enumerate(CANDIDATE_CATEGORIES)}
        preds_by_cat = blend_with_type_history(preds_by_cat, row.iloc[0], target_type)
        ordered = ordered_categories()
        share_vec = np.array([preds_by_cat.get(cat, 0.0) for cat in ordered], dtype=float)

        stats = self.event_stats[self.event_stats["code_bv"] == code_bv].sort_values("date_scrutin")
        inscrits_used = None
        if inscrits_override is not None:
            try:
                value = float(inscrits_override)
                if value > 0:
                    inscrits_used = value
            except (TypeError, ValueError):
                inscrits_used = None
        if inscrits_used is None and not stats.empty:
            serie = pd.to_numeric(stats["inscrits"], errors="coerce").dropna()
            if not serie.empty:
                inscrits_used = float(serie.iloc[-1])
        if inscrits_used is None:
            return None, "Inscrits indisponibles pour ce bureau.", ""

        def pick_rate(col: str) -> float:
            default = self.default_rates.get(col, 0.0)
            default = 0.0 if default is None or np.isnan(default) else float(default)
            type_default = self.default_rates_by_type.get(target_type, {}).get(col)
            if type_default is None or np.isnan(type_default):
                type_default = default

            bureau_scoped = self.event_stats
            if not bureau_scoped.empty and "round" in bureau_scoped.columns:
                round1 = bureau_scoped[bureau_scoped["round"] == 1]
                if not round1.empty:
                    bureau_scoped = round1

            series = None
            years = None
            if (
                not bureau_scoped.empty
                and col in bureau_scoped.columns
                and "election_type" in bureau_scoped.columns
            ):
                if target_type in bureau_scoped["election_type"].values:
                    mask = bureau_scoped["election_type"] == target_type
                    series = bureau_scoped.loc[mask, col]
                    years = bureau_scoped.loc[mask, "election_year"]

            if series is None and not self.commune_stats.empty and col in self.commune_stats.columns:
                commune_scoped = self.commune_stats
                if "round" in commune_scoped.columns:
                    round1 = commune_scoped[commune_scoped["round"] == 1]
                    if not round1.empty:
                        commune_scoped = round1
                if target_type in commune_scoped["election_type"].values:
                    mask = commune_scoped["election_type"] == target_type
                    series = commune_scoped.loc[mask, col]
                    years = commune_scoped.loc[mask, "election_year"]
                else:
                    series = commune_scoped[col]
                    years = commune_scoped["election_year"]

            if series is None:
                if bureau_scoped.empty or col not in bureau_scoped.columns:
                    return type_default
                series = bureau_scoped[col]
                years = bureau_scoped["election_year"]

            rate = _project_rate(series, years, target_year)
            if rate is None or np.isnan(rate):
                return type_default
            return float(rate)

        turnout_rate = pick_rate("turnout_pct")
        blancs_rate = pick_rate("blancs_pct")
        nuls_rate = pick_rate("nuls_pct")
        if blancs_rate + nuls_rate > turnout_rate and (blancs_rate + nuls_rate) > 0:
            scale = turnout_rate / (blancs_rate + nuls_rate)
            blancs_rate *= scale
            nuls_rate *= scale

        inscrits_total = int(round(inscrits_used))
        votants_total = int(round(inscrits_total * turnout_rate))
        blancs_total = int(round(inscrits_total * blancs_rate))
        nuls_total = int(round(inscrits_total * nuls_rate))
        if blancs_total + nuls_total > votants_total and (blancs_total + nuls_total) > 0:
            scale = votants_total / (blancs_total + nuls_total)
            blancs_total = int(round(blancs_total * scale))
            nuls_total = int(round(nuls_total * scale))
        exprimes_total = max(0, votants_total - blancs_total - nuls_total)
        abstention_total = max(0, inscrits_total - votants_total)

        bloc_counts = _allocate_counts(share_vec, exprimes_total)
        counts_by_cat = {cat: int(count) for cat, count in zip(ordered, bloc_counts)}
        counts_by_cat.update(
            {
                "blancs": int(blancs_total),
                "nuls": int(nuls_total),
                "abstention": int(abstention_total),
            }
        )
        backend_label = format_backend_label(self.backend)
        meta = (
            f"Inscrits utilisés : {inscrits_total} | Votants : {votants_total} | "
            f"Blancs : {blancs_total} | Nuls : {nuls_total} | Abstentions : {abstention_total}"
        )
        details = {
            "shares_by_cat": preds_by_cat,
            "share_vec": share_vec,
            "ordered": ordered,
            "counts": counts_by_cat,
            "totals": {
                "inscrits": inscrits_total,
                "votants": votants_total,
                "blancs": blancs_total,
                "nuls": nuls_total,
                "abstention": abstention_total,
                "exprimes": exprimes_total,
            },
        }
        return details, backend_label, meta

    def predict_bureau(
        self,
        code_bv: str,
        target_type: str,
        target_year: int,
        inscrits_override: float | None = None,
    ) -> Tuple[pd.DataFrame, str, str]:
        details, backend_label, meta = self.predict_bureau_details(
            code_bv,
            target_type,
            target_year,
            inscrits_override,
        )
        if details is None:
            return pd.DataFrame(), backend_label, ""
        counts_by_cat = details["counts"]
        ordered = details["ordered"]
        rows = []
        for cat in ordered:
            rows.append(
                {
                    "categorie": DISPLAY_CATEGORY_LABELS.get(cat, cat),
                    "nombre": int(counts_by_cat.get(cat, 0)),
                }
            )
        for extra in ["blancs", "nuls", "abstention"]:
            rows.append(
                {
                    "categorie": DISPLAY_CATEGORY_LABELS[extra],
                    "nombre": int(counts_by_cat.get(extra, 0)),
                }
            )
        return pd.DataFrame(rows), backend_label, meta


def build_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    *,
    color: str = "#3b82f6",
    color_map: Dict[str, str] | None = None,
    category_col: str = "categorie",
    ylabel: str = "Score (%)",
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if df.empty or value_col not in df.columns:
        return None
    plt.figure(figsize=(6, 3))
    labels = df[category_col].astype(str).tolist() if category_col in df.columns else []
    if color_map:
        colors = [color_map.get(label, color) for label in labels]
    else:
        colors = color
    plt.bar(labels, df[value_col], color=colors)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    return plt


def create_interface() -> gr.Blocks:
    backend = PredictorBackend()
    bureau_choices = build_bureau_choices(backend.history)
    bureau_labels = [label for label, _ in bureau_choices]
    bureau_map = {label: value for label, value in bureau_choices}
    bureau_label_by_code = {value: label for label, value in bureau_choices}
    targets = backend.available_targets()
    target_labels = [f"{t} {y}" for t, y in targets]
    history_choices = build_history_choices(backend.history)
    history_labels = [label for label, _ in history_choices]
    history_map = {label: value for label, value in history_choices}
    if ("municipales", 2026) in targets:
        default_target = "municipales 2026"
    elif targets:
        default_target = f"{targets[-1][0]} {targets[-1][1]}"
    else:
        default_target = "municipales 2026"
    default_bv = bureau_labels[0] if bureau_labels else None
    default_history = history_labels[-1] if history_labels else None
    backend_label = format_backend_label(backend.backend)
    residual_payload = load_residual_intervals()
    residuals = residual_payload.get("residuals", {}) if isinstance(residual_payload, dict) else {}
    residual_model = residual_payload.get("model", "inconnu") if isinstance(residual_payload, dict) else "inconnu"
    interval_choices = list(INTERVAL_BANDS.keys()) or ["80% (p10-p90)"]
    interval_default = interval_choices[0]
    bloc_labels = [DISPLAY_CATEGORY_LABELS.get(cat, cat) for cat in ordered_categories()]

    with gr.Blocks(title="Prévision Municipales — Ville de Sète") as demo:
        gr.Markdown(
            """
            # Prévision Municipales — Ville de Sète
            Choisissez un bureau de vote et une élection cible.  
            Le modèle estime un volume par catégorie politique, ainsi que les abstentions, blancs et nuls.

            Auteur : [Stéphane Manet](https://manet-conseil.fr) - [Linkedin](https://www.linkedin.com/in/stephanemanet) - [GitHub](https://github.com/stephmnt)
            """
        )
        with gr.Tabs():
            with gr.Tab("Prévisions"):
                with gr.Row():
                    bureau_dd = gr.Dropdown(choices=bureau_labels, value=default_bv, label="Bureau de vote")
                    target_dd = gr.Dropdown(choices=target_labels, value=default_target, label="Élection cible (type année)")
                    inscrits_in = gr.Number(value=None, label="Inscrits (optionnel)", precision=0)
                predict_btn = gr.Button("Prédire")
                source_box = gr.Markdown(value=f"Source des données : {backend_label}")
                output_df = gr.Dataframe(
                    headers=PREDICTION_OUTPUT_COLUMNS,
                    label="Prédictions (nombres)",
                )
                chart = gr.Plot()

            with gr.Tab("Historique"):
                gr.Markdown(
                    """
                    Consultation des résultats passés (sans machine learning).  
                    Sélectionnez un bureau et une élection pour afficher l'histogramme des parts par tendance politique.
                    """
                )
                with gr.Row():
                    history_bureau_dd = gr.Dropdown(choices=bureau_labels, value=default_bv, label="Bureau de vote")
                    history_election_dd = gr.Dropdown(
                        choices=history_labels,
                        value=default_history,
                        label="Élection (type année tour)",
                    )
                history_btn = gr.Button("Afficher l'historique")
                history_source = gr.Markdown(value=f"Source des données : {backend_label}")
                history_df = gr.Dataframe(headers=HISTORY_OUTPUT_COLUMNS, label="Résultats historiques")
                history_chart = gr.Plot()
                history_meta = gr.Markdown()

            with gr.Tab("Carte"):
                gr.Markdown(
                    """
                    Carte des bureaux de vote de Sète.  
                    Cliquez sur un polygone pour afficher la prédiction (table + graphique).
                    """
                )
                map_legend = gr.HTML(value=build_map_legend_html())
                with gr.Row():
                    map_target_dd = gr.Dropdown(
                        choices=target_labels,
                        value=default_target,
                        label="Élection cible (type année)",
                    )
                    map_btn = gr.Button("Afficher la carte")
                map_html = gr.HTML(value="<p>Cliquez sur 'Afficher la carte' pour charger la carte.</p>")

            with gr.Tab("Stratégie"):
                gr.Markdown(
                    """
                    Analyse stratégique par bureau : intervalles d'incertitude issus des résidus CV,
                    puis simulateur de transferts pour estimer des bascules potentielles.
                    """
                )
                with gr.Row():
                    strategy_bureau_dd = gr.Dropdown(choices=bureau_labels, value=default_bv, label="Bureau de vote")
                    strategy_target_dd = gr.Dropdown(
                        choices=target_labels,
                        value=default_target,
                        label="Élection cible (type année)",
                    )
                    strategy_inscrits_in = gr.Number(value=None, label="Inscrits (optionnel)", precision=0)
                    interval_dd = gr.Dropdown(
                        choices=interval_choices,
                        value=interval_default,
                        label="Intervalle CV",
                    )
                strategy_btn = gr.Button("Analyser l'incertitude")
                interval_source = gr.Markdown(
                    value=(
                        f"Intervalle CV basé sur le modèle : {residual_model}"
                        if residuals
                        else "Intervalle CV indisponible (fallback ±3%)."
                    )
                )
                interval_df = gr.Dataframe(
                    headers=INTERVAL_OUTPUT_COLUMNS,
                    label="Plage empirique par bloc",
                )
                interval_chart = gr.Plot()

                gr.Markdown("### Simulateur de transferts (points d'inscrits)")
                with gr.Row():
                    target_bloc_dd = gr.Dropdown(choices=bloc_labels, value=bloc_labels[0] if bloc_labels else None, label="Bloc cible")
                with gr.Row():
                    source_1_dd = gr.Dropdown(choices=TRANSFER_CATEGORY_LABELS, value=DISPLAY_CATEGORY_LABELS["abstention"], label="Source 1")
                    target_1_dd = gr.Dropdown(choices=TRANSFER_CATEGORY_LABELS, value=DISPLAY_CATEGORY_LABELS["droite_dure"], label="Cible 1")
                    delta_1 = gr.Slider(minimum=0, maximum=10, value=3, step=0.1, label="Delta 1 (points %)")
                with gr.Row():
                    source_2_dd = gr.Dropdown(choices=TRANSFER_CATEGORY_LABELS, value=DISPLAY_CATEGORY_LABELS["droite_modere"], label="Source 2")
                    target_2_dd = gr.Dropdown(choices=TRANSFER_CATEGORY_LABELS, value=DISPLAY_CATEGORY_LABELS["gauche_modere"], label="Cible 2")
                    delta_2 = gr.Slider(minimum=0, maximum=10, value=3, step=0.1, label="Delta 2 (points %)")
                simulate_btn = gr.Button("Simuler les transferts")
                sim_df = gr.Dataframe(headers=SIM_OUTPUT_COLUMNS, label="Simulation par catégorie")
                sim_chart = gr.Plot()
                opportunity_df = gr.Dataframe(headers=OPPORTUNITY_OUTPUT_COLUMNS, label="Bureaux à potentiel (trié)")

        def _predict(bv_label: str, target_label: str, inscrits_override: float | None):
            if not bv_label or not target_label:
                return pd.DataFrame(), "Entrée invalide", None
            code_bv = bureau_map.get(bv_label)
            if not code_bv:
                return pd.DataFrame(), "Bureau invalide", None
            try:
                parts = target_label.split()
                target_type, target_year = parts[0].lower(), int(parts[1])
            except Exception:
                target_type, target_year = "municipales", 2026
            df, backend_label, meta = backend.predict_bureau(code_bv, target_type, target_year, inscrits_override)
            plot = build_bar_chart(
                df,
                value_col="nombre",
                ylabel="Nombre d'électeurs",
                color_map=DISPLAY_LABEL_COLORS,
            )
            meta_label = f" | {meta}" if meta else ""
            return df, f"Source des données : {backend_label}{meta_label}", plot

        def _parse_target_label(target_label: str) -> Tuple[str, int]:
            try:
                parts = target_label.split()
                return parts[0].lower(), int(parts[1])
            except Exception:
                return "municipales", 2026

        def _map(target_label: str):
            if not target_label:
                return "<p>Élection invalide.</p>"
            target_type, target_year = _parse_target_label(target_label)
            return build_bureau_map_html(backend, target_type, target_year)

        def _history(bv_label: str, election_label: str):
            if not bv_label or not election_label:
                empty = pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
                return empty, "Entrée invalide", None, ""
            code_bv = bureau_map.get(bv_label)
            if not code_bv:
                empty = pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
                return empty, "Bureau invalide", None, ""
            election_key = history_map.get(election_label)
            if not election_key:
                empty = pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
                return empty, "Élection invalide", None, ""
            try:
                election_type, election_year, round_num = parse_election_key(election_key)
            except Exception:
                empty = pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
                return empty, "Élection invalide", None, ""
            history_slice = backend.history[
                (backend.history["code_bv"] == code_bv)
                & (backend.history["election_type"] == election_type)
                & (backend.history["election_year"] == election_year)
                & (backend.history["round"] == round_num)
            ].copy()
            if history_slice.empty:
                empty = pd.DataFrame(columns=HISTORY_OUTPUT_COLUMNS)
                return empty, f"Source des données : {backend_label}", None, "Aucun résultat pour ce bureau."
            table = prepare_history_table(history_slice)
            plot = build_bar_chart(
                table,
                value_col="score_%",
                ylabel="Score (%)",
                color_map=DISPLAY_LABEL_COLORS,
            )
            meta = format_history_meta(history_slice)
            return table, f"Source des données : {backend_label}", plot, meta

        def _strategy_interval(
            bv_label: str,
            target_label: str,
            inscrits_override: float | None,
            band_label: str,
        ):
            empty = pd.DataFrame(columns=INTERVAL_OUTPUT_COLUMNS)
            if not bv_label or not target_label:
                return empty, "Entrée invalide", None
            code_bv = bureau_map.get(bv_label)
            if not code_bv:
                return empty, "Bureau invalide", None
            target_type, target_year = _parse_target_label(target_label)
            details, backend_label_local, _ = backend.predict_bureau_details(
                code_bv,
                target_type,
                target_year,
                inscrits_override,
            )
            if details is None:
                return empty, backend_label_local, None
            totals = details["totals"]
            exprimes_total = int(totals.get("exprimes", 0))
            table = build_interval_table(
                details["shares_by_cat"],
                exprimes_total,
                residuals,
                band_label,
            )
            plot = build_interval_chart(table, color_map=DISPLAY_LABEL_COLORS)
            source = (
                f"Intervalle CV ({band_label}) basé sur le modèle : {residual_model}"
                if residuals
                else "Intervalle CV indisponible (fallback ±3%)."
            )
            return table, source, plot

        def _strategy_simulate(
            bv_label: str,
            target_label: str,
            inscrits_override: float | None,
            bloc_cible_label: str,
            source_1: str,
            target_1: str,
            delta_1_val: float,
            source_2: str,
            target_2: str,
            delta_2_val: float,
        ):
            empty_sim = pd.DataFrame(columns=SIM_OUTPUT_COLUMNS)
            empty_oppo = pd.DataFrame(columns=OPPORTUNITY_OUTPUT_COLUMNS)
            if not bv_label or not target_label:
                return empty_sim, None, empty_oppo
            code_bv = bureau_map.get(bv_label)
            if not code_bv:
                return empty_sim, None, empty_oppo
            target_type, target_year = _parse_target_label(target_label)
            details, _, _ = backend.predict_bureau_details(
                code_bv,
                target_type,
                target_year,
                inscrits_override,
            )
            if details is None:
                return empty_sim, None, empty_oppo

            transfers = []
            for src_label, dst_label, delta in [
                (source_1, target_1, delta_1_val),
                (source_2, target_2, delta_2_val),
            ]:
                src_key = CATEGORY_LABEL_TO_KEY.get(src_label)
                dst_key = CATEGORY_LABEL_TO_KEY.get(dst_label)
                if src_key and dst_key and delta and delta > 0:
                    transfers.append((src_key, dst_key, float(delta)))

            counts = details["counts"]
            totals = details["totals"]
            inscrits_total = int(totals.get("inscrits", 0))
            updated = apply_transfers(counts, inscrits_total, transfers)
            sim_table = build_simulation_table(counts, updated)
            sim_plot = build_bar_chart(
                sim_table,
                value_col="apres_transfert",
                ylabel="Nombre d'électeurs",
                color_map=DISPLAY_LABEL_COLORS,
            )

            target_bloc = CATEGORY_LABEL_TO_KEY.get(bloc_cible_label, bloc_cible_label)
            opp_rows = []
            if target_bloc in ordered_categories():
                for bv_code in backend.available_bureaux():
                    override = inscrits_override if bv_code == code_bv else None
                    bv_details, _, _ = backend.predict_bureau_details(
                        bv_code,
                        target_type,
                        target_year,
                        override,
                    )
                    if bv_details is None:
                        continue
                    base_counts = bv_details["counts"]
                    bv_totals = bv_details["totals"]
                    bv_inscrits = int(bv_totals.get("inscrits", 0))
                    updated_counts = apply_transfers(base_counts, bv_inscrits, transfers)
                    bloc_counts = {cat: int(base_counts.get(cat, 0)) for cat in ordered_categories()}
                    updated_blocs = {cat: int(updated_counts.get(cat, 0)) for cat in ordered_categories()}
                    top_base = max(bloc_counts, key=bloc_counts.get) if bloc_counts else None
                    top_after = max(updated_blocs, key=updated_blocs.get) if updated_blocs else None
                    gain = int(updated_counts.get(target_bloc, 0) - base_counts.get(target_bloc, 0))
                    opp_rows.append(
                        {
                            "bureau": bureau_label_by_code.get(bv_code, bv_code),
                            "gain_cible": gain,
                            "score_base": int(base_counts.get(target_bloc, 0)),
                            "score_apres": int(updated_counts.get(target_bloc, 0)),
                            "top_base": DISPLAY_CATEGORY_LABELS.get(top_base, top_base),
                            "top_apres": DISPLAY_CATEGORY_LABELS.get(top_after, top_after),
                            "bascule": "oui" if top_base != target_bloc and top_after == target_bloc else "non",
                        }
                    )
            opp_df = pd.DataFrame(opp_rows, columns=OPPORTUNITY_OUTPUT_COLUMNS)
            if not opp_df.empty:
                opp_df = opp_df.sort_values(["bascule", "gain_cible"], ascending=[False, False])
            return sim_table, sim_plot, opp_df

        predict_btn.click(_predict, inputs=[bureau_dd, target_dd, inscrits_in], outputs=[output_df, source_box, chart])
        history_btn.click(
            _history,
            inputs=[history_bureau_dd, history_election_dd],
            outputs=[history_df, history_source, history_chart, history_meta],
        )
        map_btn.click(
            _map,
            inputs=[map_target_dd],
            outputs=[map_html],
        )
        strategy_btn.click(
            _strategy_interval,
            inputs=[strategy_bureau_dd, strategy_target_dd, strategy_inscrits_in, interval_dd],
            outputs=[interval_df, interval_source, interval_chart],
        )
        simulate_btn.click(
            _strategy_simulate,
            inputs=[
                strategy_bureau_dd,
                strategy_target_dd,
                strategy_inscrits_in,
                target_bloc_dd,
                source_1_dd,
                target_1_dd,
                delta_1,
                source_2_dd,
                target_2_dd,
                delta_2,
            ],
            outputs=[sim_df, sim_chart, opportunity_df],
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
