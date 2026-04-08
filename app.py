# ============================================================
# app.py – Streamlit Dashboard főoldal (Áttekintő)
# ============================================================
# Futtatás: streamlit run app.py
# Szükséges: pip install streamlit pandas plotly joblib pyarrow
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Projekt gyökér a sys.path-hoz (config.py importálhatósága) ──
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHURN_PREDICTIONS_PARQUET, CUSTOMER_SEGMENTS_PARQUET

# ── Oldal konfiguráció ──
st.set_page_config(
    page_title="E-Commerce Ügyfél Analitika",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Segédfüggvények ──
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#009E73",
    "Lemorzsolódó / Alvó":    "#E69F00",
    "Új / Ígéretes":          "#CC79A7",
    "Elvesztett / Inaktív":   "#56B4E9",
}

ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció":  "#d62728",
    "⚠️  Lemorzsolódó – Win-Back Kampány":     "#ff7f0e",
    "💎 VIP Stabil – Lojalitás Program":       "#2ca02c",
    "✅ Stabil – Standard Kommunikáció":        "#1f77b4",
}


@st.cache_data
def load_segments() -> pd.DataFrame:
    return pd.read_parquet(CUSTOMER_SEGMENTS_PARQUET)


@st.cache_data
def load_churn() -> pd.DataFrame:
    return pd.read_parquet(CHURN_PREDICTIONS_PARQUET)


# ── Adatok betöltése ──
try:
    seg_df   = load_segments()
    churn_df = load_churn()
    data_ok  = True
except FileNotFoundError as e:
    st.error(f"❌ Hiányzó adatfájl: {e}\n\nFuttasd le a notebookokat először!")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.image("https://img.shields.io/badge/UCI-Online%20Retail%20II-blue", width=200)
    st.markdown("### 🛒 E-Commerce Analitika")
    st.markdown("**UCI Online Retail II** dataset alapján")
    st.markdown("---")
    st.markdown(
        """
        **Oldalak:**
        - 🏠 Áttekintő *(ez az oldal)*
        - 👥 Ügyfélszegmentáció
        - 🔮 Churn Előrejelzés
        """
    )
    st.markdown("---")
    st.markdown(
        f"**Ügyfelek száma:** {len(seg_df):,}  \n"
        f"**Előrejelzett ügyfelek:** {len(churn_df):,}"
    )

# ── Fejléc ──
st.title("🛒 E-Commerce Ügyfél Analitika Dashboard")
st.markdown(
    "**UCI Online Retail II** adathalmaz alapján készült prediktív elemzés.  \n"
    "A megfigyelési ablak: **2009-12-01 → 2011-09-09** | Célablak: **2011-09-09 → 2011-12-09** (90 nap)"
)
st.divider()

# ── KPI kártyák (felső sor) ──
churn_threshold = churn_df["churn_pred"].sum() / len(churn_df)
vip_at_risk = churn_df[
    churn_df["action"] == "🚨 VIP Veszélyben – Azonnali Retenció"
]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="👥 Szegmentált ügyfelek",
        value=f"{len(seg_df):,}",
        help="K-Means klaszterezéssel szegmentált ügyfélbázis mérete",
    )
with col2:
    vip_count = (seg_df["Segment"] == "VIP Bajnokok").sum()
    st.metric(
        label="💎 VIP Bajnokok",
        value=f"{vip_count:,}",
        delta=f"{vip_count/len(seg_df)*100:.1f}%",
        help="Legmagasabb recency + frequency + monetary értékű ügyfelek",
    )
with col3:
    churn_rate = churn_df["churn_pred"].mean() * 100
    st.metric(
        label="📉 Becsült churn ráta",
        value=f"{churn_rate:.1f}%",
        delta=f"{int(churn_df['churn_pred'].sum()):,} ügyfél",
        delta_color="inverse",
        help="Optimalizált küszöb (0.419) alapján lemorzsolódónak minősített ügyfelek aránya",
    )
with col4:
    st.metric(
        label="🚨 VIP veszélyben",
        value=f"{len(vip_at_risk):,}",
        delta=f"£{vip_at_risk['monetary_total'].sum():,.0f} kockázatos bevétel",
        delta_color="inverse",
        help="Magas értékű ÉS magas churn-kockázatú ügyfelek – azonnali beavatkozás szükséges",
    )
with col5:
    pr_auc = 0.8107  # 04-es notebook kimenete
    st.metric(
        label="🎯 Modell PR-AUC",
        value=f"{pr_auc:.4f}",
        delta="F1 = 0.785 (opt. threshold)",
        help="XGBoost modell teljesítménye a holdout teszt szetten. Optimális threshold: 0.419",
    )

st.divider()

# ── Középső sor: Szegmens eloszlás + Churn akció eloszlás ──
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("👥 Ügyfélszegmensek eloszlása")

    seg_counts = seg_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "count"]

    fig_pie = go.Figure(
        go.Pie(
            labels=seg_counts["Segment"],
            values=seg_counts["count"],
            hole=0.45,
            marker_colors=[SEGMENT_COLORS.get(s, "#aaa") for s in seg_counts["Segment"]],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Ügyfelek: %{value:,}<br>Arány: %{percent}<extra></extra>",
        )
    )
    fig_pie.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("🎯 Üzleti akció-kategóriák")

    action_counts = churn_df["action"].value_counts().reset_index()
    action_counts.columns = ["action", "count"]
    # Rendezési sorrend (veszélyesség szerint)
    order = [
        "🚨 VIP Veszélyben – Azonnali Retenció",
        "⚠️  Lemorzsolódó – Win-Back Kampány",
        "💎 VIP Stabil – Lojalitás Program",
        "✅ Stabil – Standard Kommunikáció",
    ]
    action_counts["order"] = action_counts["action"].map(
        {a: i for i, a in enumerate(order)}
    )
    action_counts = action_counts.sort_values("order")

    fig_bar = go.Figure(
        go.Bar(
            x=action_counts["count"],
            y=action_counts["action"],
            orientation="h",
            marker_color=[ACTION_COLORS.get(a, "#aaa") for a in action_counts["action"]],
            text=action_counts["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Ügyfelek: %{x:,}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        xaxis_title="Ügyfelek száma",
        margin=dict(t=20, b=20, l=20, r=120),
        height=340,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Alsó sor: RFM átlagok szegmensenként ──
st.subheader("📊 RFM profil szegmensenként")

profile_cols = ["recency_days", "frequency", "monetary_total", "return_ratio"]
profile = (
    seg_df.groupby("Segment")[profile_cols]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={
        "Segment":        "Szegmens",
        "recency_days":   "Recency (nap)",
        "frequency":      "Frequency",
        "monetary_total": "Monetary (£)",
        "return_ratio":   "Visszaküldési arány",
    })
)

# Formázott táblázat
st.dataframe(
    profile.style
    .format({
        "Recency (nap)":       "{:.0f}",
        "Frequency":           "{:.1f}",
        "Monetary (£)":        "£{:,.0f}",
        "Visszaküldési arány": "{:.1%}",
    })
    .background_gradient(subset=["Monetary (£)"], cmap="Greens")
    .background_gradient(subset=["Recency (nap)"], cmap="Reds_r"),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "ℹ️ Recency: minél kisebb, annál frissebb az ügyfél. "
    "Monetary: megfigyelési ablakban elköltött nettó összeg (£). "
    "Visszaküldési arány: sztornó rendelések / összes aktivitás."
)
