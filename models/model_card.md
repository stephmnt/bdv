# Model card
- Modèle: hist_gradient_boosting
- Split temporel: train<= 2019, valid<= 2021, test>= 2022
- Features: 38 colonnes numériques (lags, écarts national, swing, turnout)
- Cibles: parts par bloc (7 catégories) renormalisées.
- Métriques principales (MAE moyen, jeux valid/test):
  - Valid: 0.1233
  - Test: 0.1146