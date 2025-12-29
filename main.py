from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Echec de l'étape '{desc}' (code {result.returncode}). Commande: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline orchestration: preprocess -> features -> train -> predict",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Répertoire des fichiers bruts.")
    parser.add_argument("--mapping", type=Path, default=Path("config/nuances.yaml"), help="Mapping nuances->catégories.")
    parser.add_argument("--target-election", type=str, default="municipales", help="Election cible (ex: municipales).")
    parser.add_argument("--target-year", type=int, default=2026, help="Année cible.")
    parser.add_argument("--commune-code", type=str, default="301", help="Code commune pour la prédiction (Sète=301).")
    parser.add_argument("--skip-preprocess", action="store_true", help="Ne pas relancer le prétraitement.")
    parser.add_argument("--skip-features", action="store_true", help="Ne pas reconstruire le panel.")
    parser.add_argument("--skip-train", action="store_true", help="Ne pas réentraîner le modèle.")
    parser.add_argument("--skip-predict", action="store_true", help="Ne pas générer les prédictions CSV.")
    args = parser.parse_args()

    interim_path = PROJECT_ROOT / "data" / "interim" / "elections_long.parquet"
    panel_path = PROJECT_ROOT / "data" / "processed" / "panel.parquet"
    model_path = PROJECT_ROOT / "models" / "hist_gradient_boosting.joblib"

    if not args.skip_preprocess:
        run_step(
            [
                PYTHON,
                "-m",
                "src.data.preprocess",
                "--raw-dir",
                str(args.raw_dir),
                "--output-dir",
                str(PROJECT_ROOT / "data" / "interim"),
            ],
            "Prétraitement (format long)",
        )

    if not args.skip_features:
        run_step(
            [
                PYTHON,
                "-m",
                "src.features.build_features",
                "--elections-long",
                str(interim_path),
                "--mapping",
                str(args.mapping),
                "--output",
                str(panel_path),
                "--output-csv",
                str(PROJECT_ROOT / "data" / "processed" / "panel.csv"),
            ],
            "Construction du panel features+cibles",
        )

    if not args.skip_train:
        run_step(
            [
                PYTHON,
                "-m",
                "src.model.train",
                "--panel",
                str(panel_path),
                "--reports-dir",
                str(PROJECT_ROOT / "reports"),
                "--models-dir",
                str(PROJECT_ROOT / "models"),
            ],
            "Entraînement / évaluation des modèles",
        )

    if not args.skip_predict:
        run_step(
            [
                PYTHON,
                "-m",
                "src.model.predict",
                "--model-path",
                str(model_path),
                "--feature-columns",
                str(PROJECT_ROOT / "models" / "feature_columns.json"),
                "--elections-long",
                str(interim_path),
                "--mapping",
                str(args.mapping),
                "--target-election-type",
                args.target_election,
                "--target-year",
                str(args.target_year),
                "--commune-code",
                args.commune_code,
                "--output-dir",
                str(PROJECT_ROOT / "predictions"),
            ],
            "Génération des prédictions CSV",
        )

    print("\nPipeline terminé. Lance Gradio avec `python -m app.gradio_app`.")


if __name__ == "__main__":
    main()
