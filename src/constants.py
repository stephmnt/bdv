from __future__ import annotations

# Canonical blocs/categories to surface in the app outputs (7 cibles)
CANDIDATE_CATEGORIES = [
    "centre",
    "gauche_modere",
    "droite_modere",
    "gauche_dure",
    "droite_dure",
    "extreme_gauche",
    "extreme_droite",
]

# Numeric columns used across the pipeline and DB ingestion
NUMERIC_COLUMNS = [
    "voix_bloc",
    "exprimes",
    "inscrits",
    "votants",
    "blancs",
    "nuls",
    "part_bloc",
    "part_bloc_national",
    "taux_participation_national",
    "taux_participation_bv",
    "taux_blancs_bv",
    "taux_nuls_bv",
    "ecart_bloc_vs_national",
    "ecart_participation_vs_nat",
    "croissance_inscrits_depuis_base",
    "part_bloc_lag1",
    "ecart_bloc_vs_national_lag1",
    "taux_participation_bv_lag1",
    "annee_centre",
]
