# ============================================================
# pages/1_Ügyfélszegmentáció.py
# ============================================================
# Ki az ügyfelem? – RFM K-Means szegmentáció részletes elemzése
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CUSTOMER_SEGMENTS_PARQUET, SCALER_PATH, KMEANS_MODEL_PATH

st.set_page_config(
    page_title="Ügyfélszegmentáció",
    page_icon="👥",
    layout="wide",
)

# ── Konstansok ──
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#009E73",
    "Lemorzsolódó / Alvó":    "#E69F00",
    "Új / Ígéretes":          "#CC79A7",
    "Elvesztett / Inaktív":   "#56B4E9",
}


@st.cache_data
def load_segments() -> pd.DataFrame:
    df = pd.read_parquet(CUSTOMER_SEGMENTS_PARQUET)
    return df


@st.cache_resource
def load_model_and_scaler():
    import joblib
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_MODEL_PATH)
    return scaler, kmeans


# ── Adatok betöltése ──
try:
    df = load_segments()
except FileNotFoundError as e:
    st.error(f"❌ Hiányzó adatfájl: {e}\n\nFuttasd le a 02-es notebookot!")
    st.stop()

# ── Sidebar szűrők ──
with st.sidebar:
    st.markdown("### 👥 Szegmentáció")
    st.markdown("---")

    all_segments = sorted(df["Segment"].unique())
    selected_segments = st.multiselect(
        "Szegmensek",
        options=all_segments,
        default=all_segments,
        help="Szűrj egy vagy több szegmensre",
    )

    st.markdown("---")
    recency_max = int(df["recency_days"].max())
    recency_range = st.slider(
        "Recency (nap)",
        min_value=0,
        max_value=recency_max,
        value=(0, recency_max),
        step=10,
    )

    monetary_max = float(df["monetary_total"].quantile(0.99))
    monetary_range = st.slider(
        "Monetary (£, 99. percentilis-ig)",
        min_value=0.0,
        max_value=monetary_max,
        value=(0.0, monetary_max),
        step=100.0,
        format="£%.0f",
    )
    st.caption("A csúszka 99. percentilig mutat (outlierek kizárva a megjelenítésből)")

# ── Szűrés ──
mask = (
    df["Segment"].isin(selected_segments)
    & df["recency_days"].between(*recency_range)
    & df["monetary_total"].between(*monetary_range)
)
filtered = df[mask].copy()

# ── Fejléc ──
st.title("👥 Ügyfélszegmentáció")
st.markdown(
    "**RFM Feature Engineering + StandardScaler + K-Means (K=4)**  \n"
    "Az ügyfelek viselkedési profilja alapján 4 szegmensbe sorolva. "
    "Cutoff: **2011-09-09**"
)
st.divider()

if filtered.empty:
    st.warning("⚠️ A szűrési feltételeknek megfelelő ügyfél nem található.")
    st.stop()

# ── KPI sor ──
c1, c2, c3, c4 = st.columns(4)
for col_st, seg in zip(
    [c1, c2, c3, c4],
    ["VIP Bajnokok", "Lemorzsolódó / Alvó", "Új / Ígéretes", "Elvesztett / Inaktív"],
):
    n = (filtered["Segment"] == seg).sum()
    total = len(filtered)
    col_st.metric(seg, f"{n:,}", f"{n/total*100:.1f}%" if total > 0 else "–")

st.divider()

# ── 1. sor: Scatter plot + Snake Plot ──
col_scatter, col_snake = st.columns([1.3, 1])

with col_scatter:
    st.subheader("🔵 RFM Scatter – Recency vs Monetary")
    st.caption("Pontméret = Frequency | Logaritmikus Monetary tengely")

    plot_df = filtered.copy()
    plot_df["monetary_log"] = np.log1p(plot_df["monetary_total"])
    # Frequency méretezés: arányos, de olvasható
    size_ref = plot_df["frequency"].max() / 25

    fig_scatter = px.scatter(
        plot_df,
        x="recency_days",
        y="monetary_total",
        color="Segment",
        size="frequency",
        size_max=20,
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.6,
        log_y=True,
        labels={
            "recency_days":   "Recency (nap)",
            "monetary_total": "Monetary – £ (log skála)",
            "frequency":      "Frequency",
            "Segment":        "Szegmens",
        },
        hover_data={
            "recency_days":   True,
            "monetary_total": ":,.0f",
            "frequency":      True,
            "return_ratio":   ":.1%",
        },
    )
    fig_scatter.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_snake:
    st.subheader("🐍 Snake Plot – Szegmensprofil")
    st.caption("Standardizált RFM értékek szegmensenként")

    # Snake plot: szegmensátlagok a skálázott értékeken
    # Ha a scaled oszlopok léteznek – ha nem, újraskálázzuk
    scale_cols = ["recency_scaled", "frequency_scaled", "monetary_scaled"]
    if all(c in filtered.columns for c in scale_cols):
        snake_src = filtered
    else:
        # Fallback: log1p + StandardScaler manuálisan a szűrt adatokra
        from sklearn.preprocessing import StandardScaler
        tmp = filtered[["recency_days", "frequency", "monetary_total"]].copy()
        tmp = np.log1p(tmp)
        sc  = StandardScaler()
        arr = sc.fit_transform(tmp)
        snake_src = filtered.copy()
        snake_src[["recency_scaled", "frequency_scaled", "monetary_scaled"]] = arr

    snake_agg = (
        snake_src.groupby("Segment")[scale_cols]
        .mean()
        .reset_index()
        .melt(id_vars="Segment", var_name="Metrika", value_name="Standardizált érték")
    )
    label_map = {
        "recency_scaled":   "Recency",
        "frequency_scaled": "Frequency",
        "monetary_scaled":  "Monetary",
    }
    snake_agg["Metrika"] = snake_agg["Metrika"].map(label_map)

    fig_snake = px.line(
        snake_agg,
        x="Metrika",
        y="Standardizált érték",
        color="Segment",
        markers=True,
        color_discrete_map=SEGMENT_COLORS,
        labels={"Standardizált érték": "Std. érték (átlag)"},
    )
    fig_snake.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_snake.update_traces(line_width=2.5, marker_size=9)
    fig_snake.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_snake, use_container_width=True)

st.divider()

# ── 2. sor: Box plot + Összesítő tábla ──
col_box, col_table = st.columns([1.2, 1])

with col_box:
    st.subheader("📦 Eloszlás boxplot szegmensenként")

    metric_choice = st.selectbox(
        "Mutató",
        options=["monetary_total", "frequency", "recency_days", "return_ratio"],
        format_func=lambda x: {
            "monetary_total": "💰 Monetary (£)",
            "frequency":      "🔁 Frequency (vásárlások száma)",
            "recency_days":   "📅 Recency (napok)",
            "return_ratio":   "↩️ Visszaküldési arány",
        }[x],
        key="box_metric",
    )

    # 99. percentilis clip az outlierek miatt
    clip_val = filtered[metric_choice].quantile(0.99)
    plot_clip = filtered[filtered[metric_choice] <= clip_val].copy()

    fig_box = px.box(
        plot_clip,
        x="Segment",
        y=metric_choice,
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        points=False,
        labels={metric_choice: metric_choice, "Segment": ""},
    )
    fig_box.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Outlierek (99. percentilis felett) kizárva a vizualizációból")

with col_table:
    st.subheader("📋 Szegmens-profilok (átlagok)")

    profile = (
        filtered.groupby("Segment")
        .agg(
            Ügyfelek=("recency_days", "count"),
            Recency_nap=("recency_days", "mean"),
            Frequency=("frequency", "mean"),
            Monetary_GBP=("monetary_total", "mean"),
            Visszaküldes=("return_ratio", "mean"),
        )
        .round(2)
        .reset_index()
    )
    profile["Arány"] = profile["Ügyfelek"] / profile["Ügyfelek"].sum()

    st.dataframe(
        profile.rename(columns={
            "Segment":       "Szegmens",
            "Recency_nap":   "Recency (nap)",
            "Monetary_GBP":  "Monetary (£)",
            "Visszaküldes":  "Visszaküldés %",
            "Arány":         "Arány",
        })
        .style.format({
            "Recency (nap)": "{:.0f}",
            "Frequency":     "{:.1f}",
            "Monetary (£)":  "£{:,.0f}",
            "Visszaküldés %":"{:.1%}",
            "Arány":         "{:.1%}",
        })
        .background_gradient(subset=["Monetary (£)"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
        height=260,
    )

    st.markdown("---")
    st.markdown("**💡 Insight:**")
    st.info(
        "A VIP Bajnokok rendelkeznek a legmagasabb visszaküldési aránnyal (~16%) – "
        "ez tipikus e-kereskedelmi viselkedés: a leglojálisabb vásárlók használják "
        "legbátrabban a visszaküldési politikát."
    )

st.divider()

# ── 3. sor: Visszaküldési arány vs Monetary bubble chart ──
st.subheader("🔴 Visszaküldési kockázat vs. Ügyfélérték")
st.caption(
    "Buborékméret = Frequency | X tengely: visszaküldési arány | Y tengely: nettó elköltött összeg"
)

bubble_clip = filtered[filtered["monetary_total"] <= filtered["monetary_total"].quantile(0.99)].copy()

fig_bubble = px.scatter(
    bubble_clip,
    x="return_ratio",
    y="monetary_total",
    color="Segment",
    size="frequency",
    size_max=18,
    opacity=0.55,
    color_discrete_map=SEGMENT_COLORS,
    log_y=True,
    labels={
        "return_ratio":   "Visszaküldési arány",
        "monetary_total": "Monetary – £ (log)",
        "frequency":      "Frequency",
    },
    hover_data={
        "recency_days":   True,
        "monetary_total": ":,.0f",
        "return_ratio":   ":.1%",
        "frequency":      True,
    },
)
fig_bubble.update_layout(
    margin=dict(t=10, b=10, l=10, r=10),
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_bubble, use_container_width=True)

# ── Letölthető szegmens-lista ──
st.divider()
st.subheader("⬇️ Szegmentált ügyféllista letöltése")

download_df = filtered[
    ["recency_days", "frequency", "monetary_total", "return_ratio", "Segment"]
].copy()
download_df.index.name = "Customer ID"

csv_bytes = download_df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 CSV letöltése (szűrt lista)",
    data=csv_bytes,
    file_name="customer_segments_filtered.csv",
    mime="text/csv",
    help=f"Jelenleg {len(download_df):,} ügyfél a szűrési feltételek alapján",
)
