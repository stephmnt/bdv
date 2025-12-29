from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.constants import CANDIDATE_CATEGORIES

LOGGER = logging.getLogger(__name__)

TARGET_COLS = [f"target_share_{c}" for c in CANDIDATE_CATEGORIES]
META_COLS = [
    "commune_code",
    "code_bv",
    "election_type",
    "election_year",
    "round",
    "date_scrutin",
    "target_sum_before_renorm",
    "target_sum_after_renorm",
]

MODEL_GRIDS: Dict[str, List[Dict[str, object]]] = {
    "ridge": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
        {"alpha": 50.0},
    ],
    "hist_gradient_boosting": [
        {"max_depth": 3, "learning_rate": 0.08, "max_iter": 400, "min_samples_leaf": 30, "l2_regularization": 0.1},
        {"max_depth": 4, "learning_rate": 0.05, "max_iter": 600, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"max_depth": 4, "learning_rate": 0.1, "max_iter": 300, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"max_depth": 6, "learning_rate": 0.05, "max_iter": 500, "min_samples_leaf": 40, "l2_regularization": 0.5},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 500, "min_samples_leaf": 80, "l2_regularization": 1.0},
        {"max_depth": 3, "learning_rate": 0.04, "max_iter": 600, "min_samples_leaf": 120, "l2_regularization": 2.0},
        {"max_depth": 2, "learning_rate": 0.08, "max_iter": 500, "min_samples_leaf": 150, "l2_regularization": 3.0},
    ],
    "lightgbm": [
        {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 400, "learning_rate": 0.08, "num_leaves": 16, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7},
    ],
    "xgboost": [
        {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7},
    ],
    "two_stage_hgb": [
        {
            "clf_params": {"max_depth": 3, "learning_rate": 0.08, "max_iter": 300, "min_samples_leaf": 30, "l2_regularization": 0.1},
            "reg_params": {"max_depth": 3, "learning_rate": 0.08, "max_iter": 400, "min_samples_leaf": 30, "l2_regularization": 0.1},
            "epsilon": 1e-4,
            "use_logit": True,
            "use_proba": True,
        },
        {
            "clf_params": {"max_depth": 2, "learning_rate": 0.1, "max_iter": 300, "min_samples_leaf": 60, "l2_regularization": 0.2},
            "reg_params": {"max_depth": 2, "learning_rate": 0.08, "max_iter": 500, "min_samples_leaf": 60, "l2_regularization": 0.5},
            "epsilon": 1e-4,
            "use_logit": True,
            "use_proba": True,
        },
    ],
    "catboost": [
        {"depth": 6, "learning_rate": 0.05, "iterations": 500},
        {"depth": 4, "learning_rate": 0.08, "iterations": 400},
    ],
}


@dataclass
class SplitConfig:
    train_end_year: int
    valid_end_year: int
    test_start_year: int


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Panel introuvable : {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";")
    df["election_year"] = pd.to_numeric(df["election_year"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(TARGET_COLS + META_COLS)
    candidates = [c for c in df.columns if c not in exclude]
    numeric_feats = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_feats


def temporal_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["election_year"] <= cfg.train_end_year]
    valid = df[(df["election_year"] > cfg.train_end_year) & (df["election_year"] <= cfg.valid_end_year)]
    test = df[df["election_year"] >= cfg.test_start_year]
    return train, valid, test


def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_cols)
        ],
        remainder="drop",
    )


def normalize_predictions(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 0, 1)
    sums = y_pred.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return y_pred / sums


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = normalize_predictions(y_pred)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    medae = float(median_absolute_error(y_true_flat, y_pred_flat))
    r2 = float(r2_score(y_true_flat, y_pred_flat)) if len(y_true_flat) > 1 else np.nan
    evs = float(explained_variance_score(y_true_flat, y_pred_flat)) if len(y_true_flat) > 1 else np.nan
    denom = float(np.sum(np.abs(y_true_flat)))
    wape = float(np.sum(np.abs(y_true_flat - y_pred_flat)) / denom) if denom > 0 else np.nan
    smape = float(np.mean(2 * np.abs(y_pred_flat - y_true_flat) / (np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-9)))
    bias = float(np.mean(y_pred_flat - y_true_flat))
    winner_true = np.argmax(y_true, axis=1)
    winner_pred = np.argmax(y_pred, axis=1)
    winner_acc = float(np.mean(winner_true == winner_pred)) if len(winner_true) else np.nan
    metrics = {
        "mae_mean": mae,
        "rmse": rmse,
        "medae": medae,
        "r2": r2,
        "explained_var": evs,
        "wape": wape,
        "smape": smape,
        "bias": bias,
        "winner_accuracy": winner_acc,
    }
    for idx, cat in enumerate(CANDIDATE_CATEGORIES):
        metrics[f"mae_{cat}"] = float(mean_absolute_error(y_true[:, idx], y_pred[:, idx]))
    return metrics


def build_event_folds(df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if df.empty:
        return []
    work = df.copy()
    work["date_scrutin"] = pd.to_datetime(work.get("date_scrutin"), errors="coerce") # type: ignore
    if work["date_scrutin"].isna().all():
        work["date_scrutin"] = pd.to_datetime(work["election_year"], format="%Y", errors="coerce")
    work["event_key"] = (
        work["election_type"].astype(str).str.lower().str.strip()
        + "|"
        + work["election_year"].astype(str)
        + "|"
        + work["round"].astype(str)
    )
    events = (
        work[["event_key", "date_scrutin"]]
        .dropna(subset=["event_key", "date_scrutin"])
        .drop_duplicates()
        .sort_values("date_scrutin")
        .reset_index(drop=True)
    )
    if len(events) < 2:
        return []
    max_splits = min(n_splits, len(events) - 1)
    tscv = TimeSeriesSplit(n_splits=max_splits)
    folds = []
    for train_evt_idx, test_evt_idx in tscv.split(events):
        train_keys = set(events.iloc[train_evt_idx]["event_key"])
        test_keys = set(events.iloc[test_evt_idx]["event_key"])
        train_idx = work.index[work["event_key"].isin(train_keys)].to_numpy()
        test_idx = work.index[work["event_key"].isin(test_keys)].to_numpy()
        folds.append((train_idx, test_idx))
    return folds


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        classifier: Optional[BaseEstimator] = None,
        regressor: Optional[BaseEstimator] = None,
        epsilon: float = 1e-4,
        positive_threshold: float = 0.5,
        use_proba: bool = True,
        use_logit: bool = True,
        logit_eps: float = 1e-6,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.epsilon = epsilon
        self.positive_threshold = positive_threshold
        self.use_proba = use_proba
        self.use_logit = use_logit
        self.logit_eps = logit_eps

    def _default_classifier(self) -> BaseEstimator:
        return HistGradientBoostingClassifier(random_state=42)

    def _default_regressor(self) -> BaseEstimator:
        return HistGradientBoostingRegressor(random_state=42)

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        mask_pos = y > self.epsilon

        self._constant_proba = None
        if mask_pos.all() or (~mask_pos).all():
            self._constant_proba = float(mask_pos.mean())
            self.classifier_ = None
        else:
            classifier = self.classifier if self.classifier is not None else self._default_classifier()
            self.classifier_ = clone(classifier)
            self.classifier_.fit(X, mask_pos.astype(int))

        self.regressor_ = None
        if mask_pos.any():
            regressor = self.regressor if self.regressor is not None else self._default_regressor()
            self.regressor_ = clone(regressor)
            y_reg = y[mask_pos]
            if self.use_logit:
                y_reg = np.clip(y_reg, self.logit_eps, 1 - self.logit_eps)
                y_reg = np.log(y_reg / (1 - y_reg))
            self.regressor_.fit(X[mask_pos], y_reg)
        return self

    def predict(self, X):
        if self._constant_proba is not None:
            proba = np.full(len(X), self._constant_proba, dtype=float)
        else:
            check_is_fitted(self, ["classifier_"])
            if self.use_proba and hasattr(self.classifier_, "predict_proba"):
                proba = self.classifier_.predict_proba(X)[:, 1] # type: ignore
            else:
                proba = self.classifier_.predict(X) # type: ignore
        proba = np.asarray(proba, dtype=float)

        if self.regressor_ is None:
            reg_pred = np.zeros(len(proba), dtype=float)
        else:
            reg_pred = np.asarray(self.regressor_.predict(X), dtype=float)
            if self.use_logit:
                reg_pred = 1 / (1 + np.exp(-reg_pred))
            reg_pred = np.clip(reg_pred, 0, 1)

        if self.use_proba:
            preds = proba * reg_pred
        else:
            preds = np.where(proba >= self.positive_threshold, reg_pred, 0.0)
        return preds


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params: float | int | str):
        self.params = dict(params)
        self.model_ = None

    def fit(self, X, y, **fit_params):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(**self.params) # type: ignore
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("CatBoostRegressorWrapper n'est pas entraîné.")
        return self.model_.predict(X)

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self


def make_model(model_name: str, feature_cols: List[str], params: Dict[str, object]) -> Optional[Pipeline]:
    preprocessor = make_preprocessor(feature_cols)
    if model_name == "ridge":
        estimator = Ridge(**params) # type: ignore
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(random_state=42, **params) # type: ignore
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception:
            LOGGER.info("LightGBM indisponible, ignoré.")
            return None
        estimator = LGBMRegressor(random_state=42, force_row_wise=True, verbosity=-1, **params) # type: ignore
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception:
            LOGGER.info("XGBoost indisponible, ignoré.")
            return None
        estimator = XGBRegressor(random_state=42, **params)
    elif model_name == "two_stage_hgb":
        clf_params = params.get("clf_params", {})
        reg_params = params.get("reg_params", {})
        estimator = TwoStageRegressor(
            classifier=HistGradientBoostingClassifier(random_state=42, **clf_params), # type: ignore
            regressor=HistGradientBoostingRegressor(random_state=42, **reg_params), # type: ignore
            epsilon=params.get("epsilon", 1e-4), # type: ignore
            positive_threshold=params.get("positive_threshold", 0.5), # type: ignore
            use_proba=bool(params.get("use_proba", True)),
            use_logit=bool(params.get("use_logit", True)),
            logit_eps=params.get("logit_eps", 1e-6), # type: ignore
        )
    elif model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception:
            LOGGER.info("CatBoost indisponible, ignoré.")
            return None
        if not hasattr(CatBoostRegressor, "__sklearn_tags__"):
            estimator = CatBoostRegressorWrapper(verbose=0, random_state=42, **params) # type: ignore
        else:
            estimator = CatBoostRegressor(verbose=0, random_state=42, **params) # type: ignore
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    # n_jobs=1 to avoid process-based parallelism issues in some environments.
    model = MultiOutputRegressor(estimator, n_jobs=1) # type: ignore
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model: Pipeline, X, y_true: np.ndarray) -> Dict[str, float]:
    if X is None or len(X) == 0:
        return {"mae_mean": np.nan}
    y_pred = model.predict(X)
    return regression_metrics(y_true, y_pred) # type: ignore


def evaluate_cv(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int,
    target_cols: List[str],
) -> Dict[str, float]:
    folds = build_event_folds(df, n_splits)
    if not folds:
        return {"folds_used": 0}
    metrics_acc: Dict[str, list[float]] = {}
    for train_idx, test_idx in folds:
        model_clone = clone(model)
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_cols].values
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_cols].values
        model_clone.fit(X_train, y_train)
        fold_metrics = evaluate(model_clone, X_test, y_test)
        for key, value in fold_metrics.items():
            metrics_acc.setdefault(key, []).append(value)
    summary = {f"cv_{k}": float(np.nanmean(v)) for k, v in metrics_acc.items()}
    summary["folds_used"] = len(folds)
    return summary


def compute_cv_residual_intervals(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    n_splits: int,
    quantiles: Tuple[float, ...] = (0.05, 0.1, 0.9, 0.95),
) -> Dict[str, object]:
    folds = build_event_folds(df, n_splits)
    if not folds:
        return {"folds_used": 0, "quantiles": list(quantiles), "residuals": {}}

    residuals_by_cat: Dict[str, list[float]] = {cat: [] for cat in CANDIDATE_CATEGORIES}
    for train_idx, test_idx in folds:
        model_clone = clone(model)
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_cols].values
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_cols].values
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        y_pred = normalize_predictions(y_pred)
        resid = y_pred - y_test
        for idx, cat in enumerate(CANDIDATE_CATEGORIES):
            residuals_by_cat[cat].extend(resid[:, idx].tolist())

    quantile_keys = [f"q{int(q * 100):02d}" for q in quantiles]
    summary: Dict[str, Dict[str, float]] = {}
    for cat, values in residuals_by_cat.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        q_vals = np.quantile(arr, quantiles).tolist()
        entry = {key: float(val) for key, val in zip(quantile_keys, q_vals)}
        entry["mean"] = float(np.mean(arr))
        entry["std"] = float(np.std(arr))
        entry["n"] = int(arr.size)
        summary[cat] = entry

    return {
        "folds_used": len(folds),
        "quantiles": list(quantiles),
        "residuals": summary,
    }


def add_cv_selection_helpers(cv_summary: pd.DataFrame) -> pd.DataFrame:
    work = cv_summary.copy()
    block_cols = [c for c in work.columns if c.startswith("cv_mae_") and c != "cv_mae_mean"]
    if block_cols:
        work["worst_block_mae"] = work[block_cols].max(axis=1)
    if "cv_bias" in work.columns:
        work["bias_abs"] = work["cv_bias"].abs()
    return work


def select_best_model(cv_summary: pd.DataFrame) -> Tuple[str, Dict[str, object]]:
    if cv_summary.empty:
        raise RuntimeError("Aucun modèle évalué.")
    work = add_cv_selection_helpers(cv_summary)
    bias_threshold = 0.02
    candidates = work
    if "bias_abs" in work.columns:
        filtered = work[work["bias_abs"] <= bias_threshold]
        if not filtered.empty:
            candidates = filtered
    sort_cols = [c for c in ["cv_mae_mean", "worst_block_mae", "bias_abs", "cv_rmse", "cv_smape"] if c in candidates.columns]
    best_row = candidates.sort_values(sort_cols, na_position="last").iloc[0]
    return str(best_row["model"]), dict(best_row["params"])


def save_metrics(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    cv_summary: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if cv_summary is not None and not cv_summary.empty:
        cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    lines = ["# Métriques (parts, 0-1)\n"]
    for model_name, splits in metrics.items():
        lines.append(f"## {model_name}")
        for split, vals in splits.items():
            lines.append(
                f"- {split} mae_mean: {vals.get('mae_mean', float('nan')):.4f}, "
                f"rmse: {vals.get('rmse', float('nan')):.4f}, "
                f"wape: {vals.get('wape', float('nan')):.4f}, "
                f"winner_acc: {vals.get('winner_accuracy', float('nan')):.3f}"
            )
        lines.append("")
    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def save_model_card(
    model_name: str,
    cfg: SplitConfig,
    feature_cols: List[str],
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    lines = [
        "# Model card",
        f"- Modèle: {model_name}",
        f"- Split temporel: train<= {cfg.train_end_year}, valid<= {cfg.valid_end_year}, test>= {cfg.test_start_year}",
        f"- Features: {len(feature_cols)} colonnes numériques (lags, écarts national, swing, turnout)",
        "- Cibles: parts par bloc (7 catégories) renormalisées.",
        "- Métriques principales (MAE moyen, jeux valid/test):",
        f"  - Valid: {metrics[model_name]['valid'].get('mae_mean', float('nan')):.4f}",
        f"  - Test: {metrics[model_name]['test'].get('mae_mean', float('nan')):.4f}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def plot_mae_per_category(model_name: str, mae_scores: Dict[str, float], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOGGER.warning("Matplotlib indisponible, skip figure.")
        return
    if not all(f"mae_{c}" in mae_scores for c in CANDIDATE_CATEGORIES):
        LOGGER.warning("Scores MAE par categorie indisponibles, skip figure.")
        return
    cats = CANDIDATE_CATEGORIES
    values = [mae_scores[f"mae_{c}"] for c in cats]
    plt.figure(figsize=(8, 4))
    plt.bar(cats, values, color="#2c7fb8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE (part)")
    plt.title(f"MAE par catégorie - {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "mae_per_category.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement et évaluation temporelle multi-blocs.")
    parser.add_argument("--panel", type=Path, default=Path("data/processed/panel.parquet"), help="Dataset panel parquet.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Répertoire de sauvegarde des modèles.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Répertoire de sortie des rapports.")
    parser.add_argument("--train-end-year", type=int, default=2019, help="Dernière année incluse dans le train.")
    parser.add_argument("--valid-end-year", type=int, default=2021, help="Dernière année incluse dans la validation.")
    parser.add_argument("--test-start-year", type=int, default=2022, help="Première année du test (inclusif).")
    parser.add_argument("--cv-splits", type=int, default=4, help="Nombre de folds temporels pour la CV par scrutin.")
    parser.add_argument("--no-tune", action="store_true", help="Désactiver la recherche d'hyperparamètres.")
    parser.add_argument("--max-trials", type=int, default=0, help="Limiter le nombre d'essais par modèle (0=all).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_GRIDS.keys()),
        help="Liste des modèles à tester (ridge, hist_gradient_boosting, lightgbm, xgboost, two_stage_hgb, catboost).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = SplitConfig(train_end_year=args.train_end_year, valid_end_year=args.valid_end_year, test_start_year=args.test_start_year)

    panel = load_panel(args.panel)
    panel = panel.dropna(subset=TARGET_COLS)
    feature_cols = get_feature_columns(panel)
    all_na = [c for c in feature_cols if panel[c].isna().all()]
    if all_na:
        LOGGER.warning("Features supprimées car entièrement NA: %s", all_na)
        feature_cols = [c for c in feature_cols if c not in all_na]

    train_df, valid_df, test_df = temporal_split(panel, cfg)
    train_valid_df = panel[panel["election_year"] < cfg.test_start_year].copy().reset_index(drop=True)

    models_to_run = [m for m in args.models if m in MODEL_GRIDS]
    if not models_to_run:
        raise RuntimeError("Aucun modèle demandé n'est reconnu.")

    cv_rows: List[Dict[str, object]] = []
    if not args.no_tune:
        rng = np.random.default_rng(42)
        for model_name in models_to_run:
            grid = MODEL_GRIDS[model_name]
            if args.max_trials and len(grid) > args.max_trials:
                indices = rng.choice(len(grid), size=args.max_trials, replace=False)
                grid = [grid[i] for i in indices]
            for params in grid:
                model = make_model(model_name, feature_cols, params)
                if model is None:
                    continue
                cv_metrics = evaluate_cv(model, train_valid_df, feature_cols, args.cv_splits, TARGET_COLS)
                row = {"model": model_name, "params": params, **cv_metrics}
                cv_rows.append(row)

    cv_summary = pd.DataFrame(cv_rows)
    if not cv_summary.empty:
        cv_summary = cv_summary.dropna(subset=["cv_mae_mean"])
        cv_summary = add_cv_selection_helpers(cv_summary)
    if not cv_summary.empty:
        best_model_name, best_params = select_best_model(cv_summary)
        LOGGER.info("Meilleur modèle CV: %s %s", best_model_name, best_params)
    else:
        best_model_name = models_to_run[0]
        best_params = MODEL_GRIDS[best_model_name][0]
        LOGGER.warning("Pas de CV disponible, fallback sur %s %s", best_model_name, best_params)

    residual_payload = {}
    model_for_intervals = make_model(best_model_name, feature_cols, best_params)
    if model_for_intervals is not None and not train_valid_df.empty:
        residual_payload = compute_cv_residual_intervals(
            model_for_intervals,
            train_valid_df,
            feature_cols,
            TARGET_COLS,
            args.cv_splits,
        )
        if residual_payload.get("residuals"):
            args.reports_dir.mkdir(parents=True, exist_ok=True)
            (args.reports_dir / "residual_intervals.json").write_text(
                json.dumps(
                    {
                        "model": best_model_name,
                        **residual_payload,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLS].values
    X_valid, y_valid = valid_df[feature_cols], valid_df[TARGET_COLS].values
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLS].values
    X_train_valid, y_train_valid = train_valid_df[feature_cols], train_valid_df[TARGET_COLS].values

    eval_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    best_model_eval = make_model(best_model_name, feature_cols, best_params)
    if best_model_eval is None:
        raise RuntimeError(f"Modèle indisponible: {best_model_name}")
    best_model_eval.fit(X_train, y_train)
    eval_results[best_model_name] = {
        "train": evaluate(best_model_eval, X_train, y_train),
        "valid": evaluate(best_model_eval, X_valid, y_valid),
        "test": evaluate(best_model_eval, X_test, y_test),
        "train_valid": evaluate(best_model_eval, X_train_valid, y_train_valid),
    }

    best_model_final = make_model(best_model_name, feature_cols, best_params)
    if best_model_final is None:
        raise RuntimeError(f"Modèle indisponible: {best_model_name}")
    best_model_final.fit(X_train_valid, y_train_valid)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_final, args.models_dir / f"{best_model_name}.joblib")
    LOGGER.info("Modèle sauvegardé dans %s", args.models_dir / f"{best_model_name}.joblib")
    (args.models_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (args.models_dir / "best_model.json").write_text(json.dumps({"name": best_model_name}, indent=2), encoding="utf-8")

    save_metrics(eval_results, args.reports_dir, cv_summary=cv_summary)
    plot_mae_per_category(best_model_name, eval_results[best_model_name]["test"], args.reports_dir / "figures")
    save_model_card(best_model_name, cfg, feature_cols, eval_results, args.models_dir)


if __name__ == "__main__":
    main()
