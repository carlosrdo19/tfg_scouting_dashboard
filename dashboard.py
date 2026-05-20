import os

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "raw_data"

POSITION_MAP = {
    "DF": "Defensa",
    "MF": "Centrocampista",
    "FW": "Delantero",
}

PREDICTION_FILES = {
    "Regresión Lineal": "predicciones_lr.csv",
    "XGBoost": "predicciones_xgb.csv",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Scouting — Jugadores Infravalorados",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_predictions(model_name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, PREDICTION_FILES[model_name])
    df = pd.read_csv(path)
    df["Posición"] = df["Pos_clean"].map(POSITION_MAP).fillna(df["Pos_clean"])
    return df

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("⚽ Scouting Dashboard — Jugadores Infravalorados")
st.caption(
    "Detecta jugadores cuyo valor de mercado real está por debajo de lo que predicen "
    "los modelos de Machine Learning (Regresión Lineal y XGBoost)."
)

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Filtros")

    model_sel = st.radio(
        "Modelo predictivo",
        list(PREDICTION_FILES.keys()),
        help="Selecciona el modelo que se usa para estimar el valor de mercado.",
    )
    st.divider()

    df_active = load_predictions(model_sel)

    all_ligas = sorted(df_active["Comp"].unique())
    ligas_sel = st.multiselect("Liga", all_ligas, default=all_ligas)

    pos_labels = sorted(df_active["Posición"].unique())
    pos_sel = st.multiselect("Posición", pos_labels, default=pos_labels)

    max_val_95 = float(df_active["valor_real"].quantile(0.95))
    min_val = st.slider(
        "Valor real mínimo (M€)",
        min_value=0.0,
        max_value=max_val_95,
        value=0.0,
        step=0.5,
        format="%.1f M€",
        help="Filtra jugadores con un valor de mercado real por encima de este umbral.",
    )

    top_n = st.slider("Top N jugadores", min_value=5, max_value=50, value=20)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
mask_base = (
    df_active["Comp"].isin(ligas_sel)
    & df_active["Posición"].isin(pos_sel)
    & (df_active["valor_real"] >= min_val)
)
df_all_filtered = df_active[mask_base].copy()
df_infraval = (
    df_all_filtered[df_all_filtered["diferencia_pct"] > 0]
    .sort_values("diferencia_pct", ascending=False)
    .head(top_n)
)

# ---------------------------------------------------------------------------
# KPI metrics
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jugadores analizados", len(df_all_filtered))
k2.metric("Infravalorados", int((df_all_filtered["diferencia_pct"] > 0).sum()))
k3.metric(
    "Infravalorización media (top)",
    f"{df_infraval['diferencia_pct'].mean() * 100:.1f}%" if not df_infraval.empty else "—",
)
k4.metric(
    "Mayor infravaloración",
    f"{df_infraval['diferencia_pct'].max() * 100:.1f}%" if not df_infraval.empty else "—",
)

st.divider()

# ---------------------------------------------------------------------------
# Ranking table + bar chart
# ---------------------------------------------------------------------------
col_table, col_chart = st.columns([1.1, 1], gap="large")

with col_table:
    st.subheader(f"🏆 Top {top_n} más infravalorados — {model_sel}")

    if df_infraval.empty:
        st.info("No hay jugadores con los filtros seleccionados.")
    else:
        display = df_infraval[
            ["Player", "Squad", "Comp", "Posición", "Age",
             "valor_real", "valor_predicho", "diferencia_pct"]
        ].copy()
        display.columns = [
            "Jugador", "Club", "Liga", "Posición", "Edad",
            "Valor Real (M€)", "Valor Predicho (M€)", "Infravalorización (%)",
        ]
        for col in ("Valor Real (M€)", "Valor Predicho (M€)"):
            display[col] = display[col].round(2)
        display["Infravalorización (%)"] = (display["Infravalorización (%)"] * 100).round(1)
        display = display.reset_index(drop=True)
        display.index += 1

        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "Infravalorización (%)": st.column_config.ProgressColumn(
                    "Infravalorización (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=float(display["Infravalorización (%)"].max()),
                )
            },
        )

with col_chart:
    st.subheader("📊 Infravalorización por jugador")
    if not df_infraval.empty:
        df_bar = df_infraval.copy()
        df_bar["diferencia_pct"] = df_bar["diferencia_pct"] * 100
        fig_bar = px.bar(
            df_bar.sort_values("diferencia_pct"),
            x="diferencia_pct",
            y="Player",
            orientation="h",
            color="Posición",
            hover_data={
                "Squad": True,
                "Comp": True,
                "valor_real": ":.2f",
                "valor_predicho": ":.2f",
                "diferencia_pct": ":.1f",
            },
            labels={
                "diferencia_pct": "Infravalorización (%)",
                "Player": "",
                "Posición": "Posición",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_layout(
            height=500,
            margin=dict(l=0, r=10, t=10, b=0),
            yaxis={"categoryorder": "total ascending"},
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Scatter: Real vs Predicted
# ---------------------------------------------------------------------------
st.subheader("🔍 Valor Real vs. Valor Predicho")

if df_all_filtered.empty:
    st.info("Sin datos para mostrar con los filtros actuales.")
else:
    fig_scatter = px.scatter(
        df_all_filtered,
        x="valor_real",
        y="valor_predicho",
        color="Posición",
        hover_data={"Player": True, "Squad": True, "Comp": True},
        labels={
            "valor_real": "Valor Real (M€)",
            "valor_predicho": "Valor Predicho (M€)",
            "Posición": "Posición",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7,
    )

    axis_max = (
        max(df_all_filtered["valor_real"].max(), df_all_filtered["valor_predicho"].max())
        * 1.05
    )
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
        line=dict(color="#e74c3c", dash="dash", width=1.5),
    )
    fig_scatter.add_annotation(
        x=axis_max * 0.82, y=axis_max * 0.72,
        text="Valor justo",
        showarrow=False,
        font=dict(color="#e74c3c", size=11),
    )

    if not df_infraval.empty:
        fig_scatter.add_scatter(
            x=df_infraval["valor_real"],
            y=df_infraval["valor_predicho"],
            mode="markers",
            marker=dict(size=10, color="gold", line=dict(width=1.5, color="black")),
            name=f"Top {top_n} infravalorados",
            hovertext=df_infraval["Player"],
            hoverinfo="text",
        )

    fig_scatter.update_layout(height=480, margin=dict(t=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "Predicciones calculadas en el notebook sobre datos de la temporada 2024/25 (FBRef) "
    "y valoraciones de Transfermarkt. Los puntos dorados en el scatter destacan los "
    f"top {top_n} jugadores infravalorados según el modelo seleccionado."
)
