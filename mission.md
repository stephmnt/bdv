# Mission

## Étape 1

Nous créons un pipeline qui consiste à prendre en entrée des dataframes au format csv et qui les intègre dans une base de données.

La base de données comprends toujours la liste des bureaux de vote de toute la France et tout nouveau dataframe rajouterait des colonnes.

Dans un premier temps, on s'assure que le fichier soit importé et normalisé pour être conforme aux à la base de données pour s'assurer que la fusion puisse se passer.

Dans un second temps le dataset est fusionné. 

## Ancien

Tu es OpenAI Codex dans VS Code. Tu travailles dans un repo Python existant contenant des notebooks et des données dans data/raw, data/interim, data/processed. Objectif métier : au cabinet du maire de Sète, construire un outil prédictif des prochaines municipales (ex: 2026) bureau de vote par bureau de vote, basé sur l’historique électoral et une comparaison au national, puis exposer le tout via une application Gradio. Le projet doit rester opérant à long terme pour les échéances futures (pas “codé en dur” uniquement pour 2026).

Contexte fonctionnel (à respecter strictement)

Commune principale : Sète (outil centré sur Sète). Prévoir configuration pour étendre à d’autres communes ultérieurement (sans casser l’architecture).

L’utilisateur de Gradio choisit :

un bureau de vote

une élection cible à observer (par défaut : municipales 2026, mais l’UI et le backend doivent accepter n’importe quel couple (type, année) présent / futur)

Gradio renvoie :

le score prédit (%) pour chaque catégorie de candidats

entre parenthèses à côté de chaque score, la différence (en points) vs :

la dernière élection législative avant l’élection cible (dans le contexte “municipales 2026”, c’est typiquement les législatives les plus récentes avant 2026)

les municipales 2020

Catégories à utiliser (cibles et affichage) :

centre

gauche_modere

droite_modere

gauche_dure

droite_dure

extreme_gauche

extreme_droite

Données & notebooks existants

Les fichiers 01_pretraitement et 02_feature_engineering existent (notebooks dans notebooks/) et ont déjà fait un premier nettoyage / feature engineering.

Étape 1 : vérifier que ces notebooks sont cohérents avec l’objectif final (prédire municipales 2026 + long terme + bureau par bureau + comparaisons national/local), puis industrialiser : extraire la logique dans des modules Python versionnés sous src/.

Les datasets bruts sont dans data/raw. data/interim et data/processed sont disponibles et doivent être utilisés si pertinents (ne pas refaire inutilement ce qui existe déjà, mais corriger si c’est incohérent).

Exigences méthodologiques non négociables
1) Anti-fuite temporelle (time leakage)

Pour prédire une élection cible (type, année = T), les features doivent être calculées uniquement avec des données strictement antérieures à T.

Interdiction d’utiliser des résultats de l’élection cible dans les features.

Les “écarts au national” doivent être calculés uniquement pour des élections antérieures, avec le score national correspondant à ces élections antérieures.

La validation doit respecter la causalité (split temporel).

2) Structure des données adaptée (panel)

Ne pas rester sur “1 ligne = 1 bureau” wide naïf si cela empêche l’apprentissage.
Implémenter un dataset panel conceptuellement : 1 ligne = (bureau, election_type, election_year) avec :

cibles : parts de voix (%) par catégorie

features : historiques laggés, écarts national antérieurs, participation antérieure, etc.

3) Contraintes de sortie

Les prédictions sont des % par catégorie :

clip à [0, 100]

renormaliser pour sommer à 100 (gérer somme=0)
Alternative bonus : modéliser via log-ratios + softmax, mais renormalisation simple acceptable.

Étape 1 — Audit & industrialisation des notebooks

Lire et analyser notebooks/01_pretraitement.* et notebooks/02_feature_engineering.*.

Produire un diagnostic succinct (dans reports/notebook_audit.md) :

quelles tables/colonnes sont produites ?

est-ce compatible avec “bureau×élection” ?

existe-t-il des risques de leakage ?

est-ce centré sur Sète ou multi-communes ?

Refactorer en code production :

src/data/preprocess.py : chargement, nettoyage, normalisation des identifiants (commune, bureau), harmonisation des colonnes, gestion des tours (si présents).

src/features/build_features.py : construction des features “safe” et panel dataset.

Scripts CLI : python -m src.data.preprocess ..., python -m src.features.build_features ...

Générer (ou régénérer si nécessaire) un dataset final standard :

data/processed/panel.parquet

et un dictionnaire de données data/processed/data_dictionary.md

Étape 2 — Base PostgreSQL pour l’historique (utilisée par Gradio)

Construire une base PostgreSQL (docker-compose recommandé) qui stocke l’historique complet et permet de requêter rapidement par bureau.

2.1 Livrables techniques DB

docker-compose.yml lançant Postgres + un outil admin optionnel (pgAdmin facultatif).

.env.example pour config DB (host, port, user, password, dbname).

Schéma SQL (via Alembic OU SQLAlchemy create_all) versionné dans src/db/.

2.2 Modèle de données (proposition minimale à implémenter)

Tables conseillées (adapter si nécessaire, mais rester normalisé) :

communes : id, name_normalized, insee_code (si dispo)

bureaux : id, commune_id, bureau_code, bureau_label (si dispo), UNIQUE(commune_id, bureau_code)

elections : id, election_type, election_year, round (nullable), date (nullable), UNIQUE(type, year, round)

categories : id, name (les 7 catégories)

results_local : id, bureau_id, election_id, category_id, share_pct, votes (nullable), expressed (nullable), turnout_pct (nullable)

results_national : id, election_id, category_id, share_pct, votes (nullable), expressed (nullable), turnout_pct (nullable)

2.3 Ingestion / ETL vers Postgres

Créer src/db/ingest.py :

lit les données depuis data/processed (préféré) sinon reconstruit depuis data/raw via preprocess + features.

insère/upsère idempotent :

communes, bureaux, elections, categories

résultats locaux et nationaux

logs clairs + contrôles de cohérence (ex: somme des parts ≈ 100, votes ≤ exprimés, etc.)

script CLI : python -m src.db.ingest --input data/processed/panel.parquet

Étape 3 — Modélisation & prédiction

Construire un entraînement robuste + stockage des artefacts + prédiction par bureau.

3.1 Cibles

Multi-sorties : target_share_<categorie> pour les 7 catégories.

3.2 Features attendues (au minimum)

Pour une ligne (bureau, type, year=T) :

historiques laggés par catégorie (antérieurs à T)

prev_share_<cat>_any_lag1

prev_share_<cat>_<type>_lag1 (si existant)

écarts au national sur historiques :

prev_dev_to_national_<cat>_any_lag1 = prev_share_bureau - prev_share_national (sur l’élection antérieure utilisée)

ou par type si disponible

participation / abstention historiques si dispos :

prev_turnout_any_lag1, etc.

variables “swing” :

swing_<cat> = prev_share_lag1 - prev_share_lag2 (si lag2 existe)

Toutes ces features doivent être calculées sans fuite (join-asof temporel ou logique équivalente).

3.3 Split & évaluation (obligatoire)

Interdiction de random split.

Implémenter une évaluation temporelle paramétrable, ex :

train <= 2017, valid 2019–2021, test >= 2022 (exemple : configurable)

Métriques :

MAE moyenne sur les 7 catégories

MAE par catégorie

option : erreur sur “catégorie gagnante”

Générer :

reports/metrics.json

reports/metrics.md

quelques figures (matplotlib) dans reports/figures/

3.4 Modèles à entraîner

Implémenter au moins :

Ridge (baseline interprétable) avec standardisation

HistGradientBoostingRegressor (via MultiOutputRegressor si nécessaire)

LightGBM / XGBoost / CatBoost si installés (détection automatique, sinon skip proprement)

Sauvegarder modèles et preprocessors dans models/ (joblib), avec un model_card.md (date, données, split, features, métriques).

3.5 Prédiction pour une élection cible

Créer src/model/predict.py :

arguments : --target-election-type, --target-year, --commune (par défaut Sète)

produit un CSV :

predictions/pred_<type>_<year>_sete.csv

colonnes : commune, bureau_code, predicted_share_ (7), + comparateurs (voir ci-dessous)

Comparateurs à afficher dans Gradio

Pour chaque catégorie, calculer 2 deltas (points de %):

vs la dernière législative avant l’élection cible

trouver dans la DB l’élection election_type='legislatives' avec année max < target_year (et même round logique si géré)

récupérer le share_pct du bureau sur cette législative (par catégorie)

delta_leg = predicted_share - share_leg

vs les municipales 2020

si target_year != 2020 : récupérer election_type='municipales' et election_year=2020 pour ce bureau

delta_mun2020 = predicted_share - share_mun2020
Si une référence manque (bureau absent, données manquantes), afficher “N/A” au lieu du delta.

Étape 4 — Application Gradio

Créer une app Gradio production-ready dans app/gradio_app.py.

4.1 UI

Titre : “Prévision Municipales — Ville de Sète”

Inputs :

Dropdown bureau : liste des bureaux disponibles pour Sète (requête DB)

Dropdown election : couples (type, année) cibles (par défaut municipale 2026, mais liste configurable). Si 2026 n’existe pas en DB, elle doit pouvoir être sélectionnée quand même comme “cible future”.

Bouton : “Prédire”

4.2 Sorties

Afficher :

Un tableau (pandas dataframe ou composant gradio) avec 7 lignes (catégories) :

categorie

score_predit_%

Δ vs législatives (dernières) (en points)

Δ vs municipales 2020 (en points)

Option bonus : un bar chart matplotlib des scores prédits par catégorie (simple, lisible).

Format texte exigé (si rendu texte au lieu de tableau) :

centre : 21.3% (+1.2 vs législatives, -0.8 vs mun 2020)

et ainsi de suite
Avec N/A si delta indisponible.

4.3 Backend

L’app ne doit pas recalculer tout le dataset à chaque clic.

Au démarrage :

se connecte à Postgres

charge le modèle entraîné + preprocessor

Lors d’une prédiction :

récupère les features “safe” du bureau pour la cible (type, année) :

soit via une table features pré-calculées,

soit en construisant “à la volée” depuis l’historique DB (mais de manière efficace et sans fuite)

applique modèle → prédictions → post-traitement (clip + renormalisation)

calcule deltas vs références (législatives max<target_year, municipales 2020)

renvoie la table + graph

Architecture attendue du repo

Créer / compléter l’arborescence :

src/

data/

features/

db/

model/

utils/

app/

gradio_app.py

data/raw/ (existant)

data/interim/ (existant)

data/processed/ (existant)

models/

predictions/

reports/

notebooks/ (existant)

Inclure :

README.md très clair avec commandes :

(a) preprocess/build_features

(b) lancer Postgres

(c) ingest DB

(d) train/evaluate

(e) lancer Gradio

requirements.txt ou pyproject.toml

logs (INFO) + messages d’erreur actionnables (ex : DB down, modèle absent, fichiers manquants)

code robuste si data/raw vide : doit expliquer quoi mettre et comment nommer.

Points d’attention “réels”

gérer bureaux absents certaines années → imputation + deltas N/A

gérer harmonisation des libellés bureau → normalisation + warning

gérer tours (T1/T2) : inclure colonne round ou config, et éviter mélange non intentionnel

le mapping “candidat/nuance -> catégorie” est critique :

prévoir data/mappings/category_mapping.csv (ou YAML) et documenter la logique

tout non-mappé -> autres puis redistribuer/ignorer selon règle explicite (mais comme les catégories sont imposées, définir une stratégie : soit exclure “autres” du modèle, soit le répartir, soit le conserver et renormaliser sur 7 catégories — choisir une approche et la documenter)

Livrables finaux attendus

Code complet (modules + scripts CLI)

Schéma DB + docker-compose + script ingestion

Pipeline entraînement/évaluation + artefacts modèles

Application Gradio fonctionnelle

Exemples de fichiers mapping :

data/mappings/category_mapping.csv

Documentation complète dans README

Ne pas inventer de données. Travailler avec l’existant (data/interim, data/processed, notebooks), corriger si incohérent, et rendre l’ensemble production-ready (reproductible, configurable, sans fuite temporelle).