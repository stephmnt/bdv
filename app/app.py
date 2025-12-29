import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Chargement des données
# =========================

DATA_PATH = "data/processed/bv_features.parquet"

df = pd.read_parquet(DATA_PATH)
df["date_scrutin"] = pd.to_datetime(df.get("date_scrutin"), errors="coerce") # type: ignore
df["tour"] = pd.to_numeric(df.get("tour"), errors="coerce").astype("Int64") # type: ignore

# -------------------------
# Filtrage Sète uniquement
# -------------------------
# Hypothèse : code_commune INSEE
SETE_CODE_INSEE = "34301"

def resolve_code_commune(df_in: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    df_out = df_in.copy()
    if "code_commune" in df_out.columns:
        df_out["code_commune"] = df_out["code_commune"].astype("string")
        return df_out, None
    if "Code de la commune" in df_out.columns:
        df_out = df_out.rename(columns={"Code de la commune": "code_commune"})
        df_out["code_commune"] = df_out["code_commune"].astype("string")
        return df_out, None
    if "code_bv" in df_out.columns:
        df_out["code_commune"] = df_out["code_bv"].astype(str).str.slice(0, 5)
        df_out["code_commune"] = df_out["code_commune"].astype("string")
        valid = df_out["code_commune"].str.len() == 5
        if not valid.any():
            return df_out, "Impossible de dériver code_commune depuis code_bv (format inattendu)."
        return df_out, None
    df_out["code_commune"] = pd.NA
    return df_out, "Aucune colonne commune disponible (code_commune/Code de la commune/code_bv)."


df, commune_warning = resolve_code_commune(df)
df["code_commune"] = (
    df["code_commune"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.replace(r"\D", "", regex=True)
    .str.zfill(5)
    .astype("string")
)
df_sete = df[df["code_commune"] == SETE_CODE_INSEE].copy()
df_sete["tour"] = pd.to_numeric(df_sete["tour"], errors="coerce").astype("Int64")

# Colonnes blocs
BASE_BLOCS = [
    "droite_modere",
    "gauche_modere",
    "gauche_dure",
    "droite_dure",
    "centre",
    "extreme_gauche",
    "extreme_droite",
    "autre",
]
BLOC_LABELS = [b for b in BASE_BLOCS if f"part_bloc_{b}" in df_sete.columns]
BLOC_COLS = [f"part_bloc_{b}" for b in BLOC_LABELS]

# =========================
# Fonctions métier
# =========================

def compute_national_reference(df_all, type_scrutin, tour):
    """
    Calcule les parts nationales par bloc pour un scrutin et un tour donnés.
    """
    if not BLOC_COLS:
        return {}
    df_nat = df_all[
        (df_all["type_scrutin"] == type_scrutin)
        & (df_all["tour"] == tour)
    ]

    # pondération par exprimés
    weights = df_nat["exprimes"].replace(0, np.nan)

    national = {}
    for col in BLOC_COLS:
        national[col] = np.nansum(df_nat[col] * weights) / np.nansum(weights)

    return national


def table_sete(type_scrutin, tour):
    if not BLOC_COLS:
        return pd.DataFrame({"info": ["Colonnes part_bloc_* absentes."]})
    tour_val = pd.to_numeric(tour, errors="coerce")
    if pd.isna(tour_val):
        return pd.DataFrame({"info": ["Tour invalide."]})
    # données locales
    local = df_sete[
        (df_sete["type_scrutin"] == type_scrutin)
        & (df_sete["tour"] == int(tour_val))
    ].copy()

    if local.empty:
        return pd.DataFrame({"info": ["Aucune donnée disponible"]})

    # référence nationale
    nat = compute_national_reference(df, type_scrutin, tour)

    # construction tableau affiché
    rows = []

    for _, row in local.iterrows():
        r = {
            "code_bv": row["code_bv"],
            "nom_bv": row.get("nom_bv", ""),
        }

        for col in BLOC_COLS:
            part = row[col]
            ecart = part - nat.get(col, 0)

            r[col.replace("part_bloc_", "")] = round(part * 100, 2)
            r[col.replace("part_bloc_", "") + "_ecart_nat"] = round(ecart * 100, 2)

        rows.append(r)

    result = pd.DataFrame(rows)

    # tri par écart extrême droite (exemple)
    if "extreme_droite_ecart_nat" in result.columns:
        result = result.sort_values(
            "extreme_droite_ecart_nat", ascending=False
        )

    return result


def get_bv_timeseries(code_bv: str, tour: int | None) -> pd.DataFrame:
    if df_sete.empty or not BLOC_COLS:
        return pd.DataFrame(columns=["date_scrutin"] + BLOC_COLS)
    subset = df_sete[df_sete["code_bv"].astype(str) == str(code_bv)].copy()
    subset["tour"] = pd.to_numeric(subset["tour"], errors="coerce").astype("Int64")
    if tour is not None:
        subset = subset[subset["tour"] == tour]
    subset = subset.dropna(subset=["date_scrutin"]).sort_values("date_scrutin")
    return subset[["date_scrutin"] + BLOC_COLS]


def plot_bv_timeseries(code_bv: str, tour_choice, bloc_choices=None):
    tour = None if tour_choice == "Tous" else int(tour_choice)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not BLOC_COLS:
        ax.text(0.5, 0.5, "Colonnes part_bloc_* absentes.", ha="center", va="center")
        ax.axis("off")
        return fig
    df_ts = get_bv_timeseries(code_bv, tour)
    if df_ts.empty:
        tours_avail = (
            df_sete[df_sete["code_bv"].astype(str) == str(code_bv)]["tour"]
            .dropna()
            .unique()
            .tolist()
        )
        ax.text(
            0.5,
            0.5,
            f"Aucune donnée après filtre tour={tour}. Valeurs disponibles: {sorted(tours_avail)}",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.axis("off")
        return fig

    selected = bloc_choices or BLOC_LABELS
    selected_cols = [f"part_bloc_{b}" for b in selected if f"part_bloc_{b}" in df_ts.columns]
    if not selected_cols:
        ax.text(0.5, 0.5, "Aucun bloc sélectionné.", ha="center", va="center")
        ax.axis("off")
        return fig
    for col in selected_cols:
        ax.plot(df_ts["date_scrutin"], df_ts[col], label=col.replace("part_bloc_", ""))
    ax.set_title(f"Évolution politique – BV {code_bv}")
    ax.set_ylabel("Part des voix (exprimés)")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =========================
# Interface Gradio
# =========================

def format_bv_label(code_bv: str) -> str:
    code_str = str(code_bv)
    if code_str.isdigit() and code_str.startswith(SETE_CODE_INSEE) and len(code_str) == 9:
        bureau_num = code_str[-4:]
        return f"BV {int(bureau_num)} ({code_str})"
    return code_str


bv_values = (
    sorted(df_sete["code_bv"].astype(str).unique().tolist())
    if "code_bv" in df_sete.columns
    else []
)
bv_choices = [(format_bv_label(code), code) for code in bv_values]
scrutins = sorted(df_sete["type_scrutin"].unique())
tours = sorted(df_sete["tour"].dropna().unique())
tour_options = ["Tous"] + [str(t) for t in tours]
status_messages = []
if commune_warning:
    status_messages.append(commune_warning)
if df_sete.empty:
    status_messages.append(
        "Aucune ligne pour la commune 34301 (Sète). Vérifie `code_commune` / le filtre."
    )
if not BLOC_COLS:
    status_messages.append("Colonnes part_bloc_* absentes dans bv_features.")
missing_blocs = [f"part_bloc_{b}" for b in BASE_BLOCS if f"part_bloc_{b}" not in df_sete.columns]
if missing_blocs:
    status_messages.append(f"Colonnes blocs manquantes: {', '.join(missing_blocs)}")
tour_dtype = str(df_sete["tour"].dtype) if "tour" in df_sete.columns else "n/a"
tour_sample = sorted(df_sete["tour"].dropna().unique().tolist())[:10]
status_messages.append(f"tour dtype: {tour_dtype}")
status_messages.append(f"tours disponibles (échantillon): {tour_sample}")
status_messages.append(
    f"df_sete: {len(df_sete)} lignes, {df_sete['code_bv'].nunique() if 'code_bv' in df_sete.columns else 0} BV"
)
status_messages.append(f"blocs actifs: {', '.join(BLOC_LABELS) if BLOC_LABELS else 'aucun'}")
status_text = "\n".join(f"- {msg}" for msg in status_messages)

with gr.Blocks(title="Résultats électoraux – Bureaux de vote de Sète") as app:
    gr.Markdown(
        """
        # 🗳️ Résultats électoraux – Ville de Sète

        **Bureaux de vote uniquement – comparaison au niveau national**

        Les pourcentages sont exprimés en **% des exprimés**.  
        Les écarts sont en **points par rapport au national**.
        """
    )
    if status_text:
        gr.Markdown(f"**Alertes**\n{status_text}")

    with gr.Tabs():
        with gr.Tab("Bureaux de vote"):
            with gr.Row():
                type_scrutin = gr.Dropdown(
                    scrutins,
                    label="Type de scrutin",
                    value=scrutins[0] if scrutins else None,
                )
                tour = gr.Dropdown(
                    tours,
                    label="Tour",
                    value=tours[0] if tours else None,
                )

            output = gr.Dataframe(
                label="Bureaux de vote – parts locales et écart au national",
                interactive=False,
                wrap=True,
            )

            btn = gr.Button("Afficher")

            btn.click(
                fn=table_sete,
                inputs=[type_scrutin, tour],
                outputs=output,
            )

        with gr.Tab("Évolution temporelle"):
            bv_selector = gr.Dropdown(
                bv_choices,
                label="Bureau de vote",
                value=bv_values[0] if bv_values else None,
            )
            tour_selector = gr.Dropdown(
                tour_options,
                label="Tour",
                value="Tous",
            )
            blocs_selector = gr.Dropdown(
                BLOC_LABELS,
                label="Blocs à afficher",
                value=BLOC_LABELS,
                multiselect=True,
            )
            plot = gr.Plot(
                value=plot_bv_timeseries(
                    bv_values[0] if bv_values else "", "Tous", BLOC_LABELS
                )
            )

            bv_selector.change(
                fn=plot_bv_timeseries,
                inputs=[bv_selector, tour_selector, blocs_selector],
                outputs=plot,
            )
            tour_selector.change(
                fn=plot_bv_timeseries,
                inputs=[bv_selector, tour_selector, blocs_selector],
                outputs=plot,
            )
            blocs_selector.change(
                fn=plot_bv_timeseries,
                inputs=[bv_selector, tour_selector, blocs_selector],
                outputs=plot,
            )

# =========================
# Lancement
# =========================
# Tests manuels:
# 1) Lancer l'app.
# 2) Onglet "Évolution temporelle": choisir un BV, tester Tous / Tour 1 / Tour 2.
# 3) Vérifier que la légende n'occulte pas les courbes et que seuls 8 blocs apparaissent.
# 4) Vérifier le libellé BV (BV X + code) et les alertes en haut de page.

if __name__ == "__main__":
    app.launch()
