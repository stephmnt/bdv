---
title: Bdv
emoji: 🌍
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# Elections Sète - Prévision municipales

Pipeline complet pour harmoniser les données électorales, construire un dataset panel sans fuite temporelle, entraîner des modèles multi-blocs, charger l'historique dans PostgreSQL et exposer des résultats via Gradio.

## Installation
- Python 3.10+ recommandé.
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

## Fichiers YAML (configuration)
### `config/communes.yaml`
Ce fichier définit **les communes à inclure** (codes INSEE). Il est consommé par le pipeline (`src.pipeline.run_full_pipeline`) pour filtrer les données au niveau commune.

Formats acceptés (les codes sont normalisés en 5 chiffres) :
```yaml
communes:
  "34301": "Sète"
  "34172": "Frontignan"
```
ou
```yaml
communes:
  - code_insee: "34301"
    nom: "Sète"
  - "34172"
```
Si tu modifies ce fichier, il faut **relancer le pipeline** pour régénérer les données filtrées.

### `config/raw_sources.yaml`
Description des fichiers bruts et de leur structure (colonnes, séparateur, métadonnées).
C'est **le point d'entrée** pour ajouter un nouveau CSV au pipeline.

Exemple (copie d'une election precedente + ajustements) :
```yaml
24_L_T1.csv:
  copy_from: 22_L_T1.csv
  date_scrutin: "2024-06-30"
  code_bv_cols: ["Code commune", "Code BV"]
  rename_map:
    Nuance Liste: code_candidature
    Libellé Abrégé Liste: nom_candidature
```

### `config/nuances.yaml`
Mapping des nuances vers les blocs politiques (avec overrides).
Par défaut, le mapping CSV historique est réutilise et on peut **surcharger** ou **ajouter** des nuances :
```yaml
base_mapping: data/mapping_candidats_blocs.csv
overrides:
  - code_candidature: "XYZ"
    nom_candidature: "Exemple"
    blocs: [gauche_modere, centre]
```

### `docker-compose.yml`
Fichier YAML pour démarrer PostgreSQL (et éventuellement pgAdmin). Utilisé par :
```bash
docker-compose up -d postgres
docker-compose --profile admin up
```

## 1. Prétraitement (harmonisation)
```bash
# Harmonisation des CSV bruts -> data/interim/elections_long.parquet
python -m src.data.preprocess --raw-dir data/raw --output-dir data/interim
```
Par défaut, le prétraitement lit `config/raw_sources.yaml`. Tu peux surcharger via `--meta-config`.

## 2. Pipeline communes + features (optionnel mais recommandé si tu filtres par communes)
Le pipeline applique le filtre `config/communes.yaml` et génère `data/processed/elections_blocs.*`.
À lancer depuis un notebook ou un petit script :
```bash
python3 - <<'PY'
from pathlib import Path
from src.pipeline import run_full_pipeline

run_full_pipeline(
    elections_long_path=Path("data/interim/elections_long.parquet"),
    mapping_path=Path("config/nuances.yaml"),
    output_dir=Path("data/processed"),
    target_communes_path=Path("config/communes.yaml"),
)
PY
```

## 3. Construction du panel (features + cibles)
```bash
python -m src.features.build_features \
  --elections-long data/interim/elections_long.parquet \
  --mapping config/nuances.yaml \
  --output data/processed/panel.parquet
```
Le dictionnaire de données est généré dans `data/processed/data_dictionary.md`.

Note : `src.features.build_features` **ne filtre pas** via `config/communes.yaml`. Si tu veux limiter l'entraînement à certaines communes, filtre `elections_long` en amont ou adapte le pipeline.

## 4. Base PostgreSQL
```bash
cp .env.example .env
docker-compose up -d postgres   # pgAdmin en option: `docker-compose --profile admin up`

# Ingestion du panel dans le schéma normalisé
python -m src.db.ingest --input data/processed/panel.parquet
```
Le schéma est défini dans `src/db/schema.py`.

## 5. Entraînement & évaluation
Commande demandée (CV stricte par scrutin) :
```bash
python3 -m src.model.train --cv-splits 4 --models hist_gradient_boosting
```

Options principales :
- `--panel` : chemin du panel (`data/processed/panel.parquet` par défaut).
- `--models-dir` / `--reports-dir` : sorties modèles et rapports.
- `--train-end-year`, `--valid-end-year`, `--test-start-year` : split temporel.
- `--cv-splits` : nb de folds temporels (par scrutin).
- `--no-tune` : désactive la grille d'hyperparamètres.
- `--max-trials` : limite le nombre d'essais par modèle.
- `--models` : liste de modèles à tester (ex: `ridge`, `hist_gradient_boosting`, `lightgbm`, `xgboost`, `two_stage_hgb`, `catboost`).

Sorties :
- Modèle + preprocessor : `models/<nom>.joblib` et `models/feature_columns.json`
- Modèle sélectionné : `models/best_model.json`
- Rapport métriques : `reports/metrics.json` et `reports/metrics.md`
- CV détaillée : `reports/cv_summary.csv`
- Figure : `reports/figures/mae_per_category.png`
- Model card : `models/model_card.md`

## 6. Génération de prédictions hors ligne
```bash
python -m src.model.predict \
  --model-path models/hist_gradient_boosting.joblib \
  --target-election-type municipales \
  --target-year 2026 \
  --commune-code 34301
# -> predictions/pred_municipales_2026_sete.csv
```
Cette commande produit des **parts (%)** et des deltas vs législatives et municipales 2020.

## 7. Application Gradio
```bash
python -m app.gradio_app
```
Comportement :
- Backend PostgreSQL si disponible, sinon fallback fichiers locaux.
- **Historique** : consultation bureau par bureau (pas de ML).
- **Prédiction** : parts par bloc converties en **comptes** (personnes) + `blancs`, `nuls`, `abstentions`.
- `inscrits` peut être fourni par l'utilisateur (sinon valeur historique la plus récente du bureau).
- Cibles proposées : municipales 2026 (tour 1), legislatives 2027 (tour 1), presidentielles 2027 (tour 1).

## Structure des données
- Configurations : `config/`
- Bruts : `data/raw/`
- Long harmonisé : `data/interim/elections_long.parquet`
- Élections blocs (filtrées) : `data/processed/elections_blocs.parquet`
- Stats communales par scrutin : `data/processed/commune_event_stats.parquet`
- Panel features+cibles : `data/processed/panel.parquet`
- Mapping nuances -> catégories : `config/nuances.yaml` (base: `data/mapping_candidats_blocs.csv`)

## Notes
- Aucune fuite temporelle : les features sont calculées uniquement sur des scrutins strictement antérieurs à la cible.
- Les parts sont clipées à [0, 1] puis renormalisées.
- Les blancs/nuls dépendent des colonnes disponibles dans l'historique ; si une source ne les fournit pas, ils seront à 0.

## Inventaire des fichiers (snapshot)
Statuts :
- `actif` : utilisé par le pipeline actuel.
- `généré` : produit par le pipeline/entraînement (recréable).
- `hérité (début projet)` : ancien fichier ou prototype.
- `optionnel` : utile mais non requis au runtime.
- `système (inutile)` : métadonnées OS.

| Fichier | Fonction | Statut |
|---|---|---|
| `.DS_Store` | Métadonnées macOS | système (inutile) |
| `.env.example` | Template des variables d'environnement (DB) | actif |
| `.gitignore` | Règles gitignore | actif |
| `Elections_Sete.code-workspace` | Config VSCode (workspace) | optionnel |
| `README.md` | Documentation projet | actif |
| `app/__init__.py` | Package app (init) | actif |
| `app/app.py` | Ancienne app Gradio (bv_features.parquet) | hérité (début projet) |
| `app/gradio_app.py` | Application Gradio principale | actif |
| `app.py` | Ancienne interface Gradio (compute_predictions) | hérité (début projet) |
| `catboost_info/catboost_training.json` | Artefacts CatBoost (logs/metrics) | généré |
| `catboost_info/learn/events.out.tfevents` | Artefacts CatBoost (logs/metrics) | généré |
| `catboost_info/learn_error.tsv` | Artefacts CatBoost (logs/metrics) | généré |
| `catboost_info/time_left.tsv` | Artefacts CatBoost (logs/metrics) | généré |
| `config/communes.yaml` | Liste des communes cibles (codes INSEE) | actif |
| `config/nuances.yaml` | Overrides mapping nuances -> blocs | actif |
| `config/raw_sources.yaml` | Schéma des CSV bruts (meta-config) | actif |
| `data/.DS_Store` | Métadonnées macOS | système (inutile) |
| `data/contours-france-entiere-latest-v2.geojson` | Fond cartographique (geojson) | optionnel |
| `data/interim/.DS_Store` | Métadonnées macOS | système (inutile) |
| `data/interim/candidates_long.parquet` | Données intermédiaires long format | généré |
| `data/interim/elections_long.csv` | Données intermédiaires long format | généré |
| `data/interim/elections_long.parquet` | Données intermédiaires long format | généré |
| `data/interim/frames_std/14_EU.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/14_MN14_T1T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/17_L_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/17_L_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/17_PR_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/17_PR_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/19_EU.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/20_MN_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/20_MN_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/21_DEP_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/21_DEP_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/21_REG_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/21_REG_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/22_L_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/22_L_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/22_PR_T1.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/22_PR_T2.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/frames_std/24_EU.parquet` | Intermédiaire standardisé par scrutin | généré |
| `data/interim/harmonized/14_EU_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/14_MN14_T1T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/17_L_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/17_L_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/17_PR_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/17_PR_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/19_EU_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/20_MN_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/20_MN_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/21_DEP_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/21_DEP_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/21_REG_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/21_REG_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/22_L_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/22_L_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/22_PR_T1_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/22_PR_T2_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/harmonized/24_EU_harmonized.csv` | CSV harmonisé par scrutin | généré |
| `data/interim/unmapped_nuances.csv` | Données intermédiaires long format | généré |
| `data/mapping_candidats_blocs.csv` | Mapping nuances -> blocs (base) | actif |
| `data/mappings/category_mapping.csv` | Copie/variante de mapping | hérité (début projet) |
| `data/processed/bv_features.parquet` | Features legacy (utilisées par app/app.py) | hérité (début projet) |
| `data/processed/data_dictionary.md` | Dictionnaire de données généré | généré (doc) |
| `data/processed/elections_blocs.csv` | Dataset blocs (filtré communes) | généré (utilisé) |
| `data/processed/elections_blocs.parquet` | Dataset blocs (filtré communes) | généré (utilisé) |
| `data/processed/history_cache.parquet` | Cache local (historique/prédictions) | généré (cache) |
| `data/processed/panel.csv` | Panel features+cibles | généré (utilisé) |
| `data/processed/panel.parquet` | Panel features+cibles | généré (utilisé) |
| `data/processed/predictions_cache.parquet` | Cache local (historique/prédictions) | généré (cache) |
| `data/processed/predictions_municipales_2026.csv` | Exports de prédictions | généré (résultats) |
| `data/processed/predictions_municipales_2026_blocs.csv` | Exports de prédictions | généré (résultats) |
| `data/processed/predictions_municipales_sete_2026.csv` | Exports de prédictions | généré (résultats) |
| `data/raw/14_EU.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/14_MN14_T1T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/17_L_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/17_L_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/17_PR_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/17_PR_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/19_EU.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/20_MN_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/20_MN_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/21_DEP_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/21_DEP_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/21_REG_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/21_REG_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/22_L_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/22_L_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/22_PR_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/22_PR_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/24_EU.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/24_L_T1.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `data/raw/24_L_T2.csv` | Données brutes (entrée prétraitement) | actif (entrée pipeline) |
| `datasets/.DS_Store` | Métadonnées macOS | système (inutile) |
| `datasets/14_EU.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/14_MN14_T1T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/17_L_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/17_L_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/17_PR_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/17_PR_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/19_EU.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/20_MN_T1.tsv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/20_MN_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/21_DEP_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/21_DEP_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/21_REG_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/21_REG_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/22_L_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/22_L_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/22_PR_T1.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/22_PR_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/24_EU.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/24_L_T1T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `datasets/24_L_T2.csv` | Copie brute des datasets (ancienne structure) | hérité (début projet) |
| `docker-compose.yml` | Services Docker (PostgreSQL/pgAdmin) | actif |
| `harmoniser.md` | Notes d'harmonisation | optionnel |
| `main.py` | Orchestrateur pipeline (CLI utilitaire) | optionnel |
| `mission.md` | Backlog / notes projet | optionnel |
| `models/best_model.json` | Nom du meilleur modèle | généré (utilisé) |
| `models/feature_columns.json` | Liste des features du modèle | généré (utilisé) |
| `models/hist_gradient_boosting.joblib` | Modèle entraîné | généré (utilisé) |
| `models/model_card.md` | Model card (synthèse) | généré (doc) |
| `notebooks/01_pretraitement.ipynb` | Notebook d'analyse / exploration | optionnel (exploration) |
| `notebooks/02_feature_engineering.ipynb` | Notebook d'analyse / exploration | optionnel (exploration) |
| `notebooks/03_modelisation_prediction.ipynb` | Notebook d'analyse / exploration | optionnel (exploration) |
| `notebooks/aed.ipynb` | Notebook d'analyse / exploration | optionnel (exploration) |
| `notebooks/catboost_info/catboost_training.json` | Artefacts CatBoost (notebook) | généré |
| `notebooks/catboost_info/learn/events.out.tfevents` | Artefacts CatBoost (notebook) | généré |
| `notebooks/catboost_info/learn_error.tsv` | Artefacts CatBoost (notebook) | généré |
| `notebooks/catboost_info/time_left.tsv` | Artefacts CatBoost (notebook) | généré |
| `output/.DS_Store` | Métadonnées macOS | système (inutile) |
| `output/Sans titre 2.png` | Exports graphiques | hérité (début projet) |
| `output/Sans titre 3.png` | Exports graphiques | hérité (début projet) |
| `output/Sans titre 4.png` | Exports graphiques | hérité (début projet) |
| `output/Sans titre 5.png` | Exports graphiques | hérité (début projet) |
| `output/Sans titre 6.png` | Exports graphiques | hérité (début projet) |
| `output/Sans titre.png` | Exports graphiques | hérité (début projet) |
| `output/output.png` | Exports graphiques | hérité (début projet) |
| `predictions/pred_municipales_2026_sete.csv` | Exports de prédictions | généré (résultats) |
| `reports/colonnes_comparatif.csv` | Rapport / métriques | généré |
| `reports/cv_summary.csv` | Rapport / métriques | généré |
| `reports/figures/mae_per_category.png` | Figures de rapports | généré |
| `reports/metrics.json` | Rapport / métriques | généré |
| `reports/metrics.md` | Rapport / note analytique | généré (doc) |
| `reports/notebook_audit.md` | Rapport / note analytique | généré (doc) |
| `requirements.txt` | Dépendances Python | actif |
| `src/__init__.py` | Package src (init) | actif |
| `src/constants.py` | Constantes projet | actif |
| `src/data/__init__.py` | Module data | actif |
| `src/data/preprocess.py` | Prétraitement/harmonisation | actif |
| `src/data_prep.py` | Librairie d'harmonisation des données | actif |
| `src/database.py` | Accès base SQL (fallback/app) | actif |
| `src/db/__init__.py` | Module DB | actif |
| `src/db/ingest.py` | Ingestion PostgreSQL | actif |
| `src/db/schema.py` | Schéma PostgreSQL | actif |
| `src/features/__init__.py` | Module features | actif |
| `src/features/build_features.py` | Construction du panel features+cibles | actif |
| `src/model/predict.py` | Prédiction hors ligne | actif |
| `src/model/train.py` | Entraînement + CV | actif |
| `src/pipeline.py` | Pipeline de construction (blocs + stats) | actif |
| `src/prediction.py` | Prédiction legacy (app.py) | hérité (début projet) |
| `supports/Plan-2024_Bureaux-de-vote.pdf` | Documents de référence | optionnel |
| `supports/zonages_admin_canton.pdf` | Documents de référence | optionnel |