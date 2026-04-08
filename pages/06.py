# ============================================================
# pages/2_Churn_Előrejelzés.py
# ============================================================
# Ki fog lemorzsolódni? – Churn előrejelzés, threshold elemzés,
# akciólista és kampánycél-lista
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

from config import CHURN_PREDICTIONS_PARQUET

st.set_page_config(
    page_title="Churn Előrejelzés",
    page_icon="🔮",
    layout="wide",
)

# ── Konstansok ──
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#009E73",
    "Lemorzsolódó / Alvó":    "#E69F00",
    "Új / Ígéretes":          "#CC79A7",
    "Elvesztett / Inaktív":   "#56B4E9",
    "Ismeretlen":             "#999999",
}

ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció":  "#d62728",
    "⚠️  Lemorzsolódó – Win-Back Kampány":     "#ff7f0e",
    "💎 VIP Stabil – Lojalitás Program":       "#2ca02c",
    "✅ Stabil – Standard Kommunikáció":        "#1f77b4",
}

BEST_THRESHOLD = 0.419   # 04-es notebook kimenete

ACTION_ORDER = [
    "🚨 VIP Veszélyben – Azonnali Retenció",
    "⚠️  Lemorzsolódó – Win-Back Kampány",
    "💎 VIP Stabil – Lojalitás Program",
    "✅ Stabil – Standard Kommunikáció",
]


@st.cache_data
def load_churn() -> pd.DataFrame:
    return pd.read_parquet(CHURN_PREDICTIONS_PARQUET)


# ── Adatok betöltése ──
try:
    df = load_churn()
except FileNotFoundError as e:
    st.error(f"❌ Hiányzó adatfájl: {e}\n\nFuttasd le a 04-es notebookot!")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🔮 Churn Előrejelzés")
    st.markdown("---")

    st.markdown("**Threshold (küszöbérték)**")
    threshold = st.slider(
        "Churn küszöb",
        min_value=0.1,
        max_value=0.9,
        value=BEST_THRESHOLD,
        step=0.01,
        format="%.3f",
        help=(
            f"Az optimalizált érték: {BEST_THRESHOLD:.3f} (F1-maximalizáló, "
            "Precision=0.724, Recall=0.856). Az 0.5 az alapértelmezett."
        ),
    )

    st.markdown("---")
    all_segments = sorted(df["rfm_segment"].unique())
    selected_segments = st.multiselect(
        "RFM szegmens szűrő",
        options=all_segments,
        default=all_segments,
    )

    st.markdown("---")
    action_filter = st.multiselect(
        "Akció-kategória szűrő",
        options=ACTION_ORDER,
        default=ACTION_ORDER,
    )

# ── Dinamikus akció újraszámítás az interaktív threshold alapján ──
monetary_median = df["monetary_total"].median()


def assign_action_dynamic(row, thr):
    high_value = row["monetary_total"] > monetary_median
    high_churn = row["churn_proba"] >= thr
    if   high_value and     high_churn: return "🚨 VIP Veszélyben – Azonnali Retenció"
    elif high_value and not high_churn: return "💎 VIP Stabil – Lojalitás Program"
    elif not high_value and high_churn: return "⚠️  Lemorzsolódó – Win-Back Kampány"
    else:                               return "✅ Stabil – Standard Kommunikáció"


df["action_dynamic"] = df.apply(assign_action_dynamic, axis=1, thr=threshold)
df["churn_pred_dynamic"] = (df["churn_proba"] >= threshold).astype(int)

# ── Szűrés ──
mask = df["rfm_segment"].isin(selected_segments) & df["action_dynamic"].isin(action_filter)
filtered = df[mask].copy()

# ── Fejléc ──
st.title("🔮 Churn Előrejelzés & Akciótervek")
st.markdown(
    "**XGBoost modell** – RFM feature-ök alapján predikált lemorzsolódási valószínűség.  \n"
    f"Aktuális küszöb: **{threshold:.3f}** "
    f"{'(optimalizált)' if abs(threshold - BEST_THRESHOLD) < 0.001 else '(módosított)'}"
)

if abs(threshold - BEST_THRESHOLD) > 0.001:
    st.info(
        f"ℹ️ Az optimalizált küszöb ({BEST_THRESHOLD:.3f}) F1=0.785, Recall=0.856, Precision=0.724 "
        f"értékeket ad. Jelenlegi küszöb: {threshold:.3f}"
    )

st.divider()

# ── KPI sor ──
total_churn = df["churn_pred_dynamic"].sum()
total = len(df)
vip_at_risk = df[df["action_dynamic"] == "🚨 VIP Veszélyben – Azonnali Retenció"]
winback      = df[df["action_dynamic"] == "⚠️  Lemorzsolódó – Win-Back Kampány"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "📉 Lemorzsolódók (pred.)",
    f"{total_churn:,}",
    f"{total_churn/total*100:.1f}% az ügyfélbázisból",
    delta_color="inverse",
)
c2.metric(
    "🚨 VIP Veszélyben",
    f"{len(vip_at_risk):,}",
    delta=f"£{vip_at_risk['monetary_total'].sum():,.0f} kockázat",
    delta_color="inverse",
)
c3.metric(
    "⚠️ Win-Back kampány",
    f"{len(winback):,}",
    delta=f"£{winback['monetary_total'].sum():,.0f} bevétel",
    delta_color="inverse",
)
c4.metric(
    "💎 VIP Stabil",
    f"{(df['action_dynamic'] == '💎 VIP Stabil – Lojalitás Program').sum():,}",
    delta="Upsell lehetőség",
    delta_color="normal",
)
c5.metric(
    "✅ Stabil",
    f"{(df['action_dynamic'] == '✅ Stabil – Standard Kommunikáció').sum():,}",
    delta="Standard kommunikáció",
    delta_color="off",
)

st.divider()

# ── 1. sor: Valószínűség-eloszlás + Akció-elosztás ──
col_hist, col_action = st.columns([1.2, 1])

with col_hist:
    st.subheader("📊 Churn valószínűség eloszlása")

    fig_hist = go.Figure()
    # Marad (0) és Lemorzsolódik (1) külön
    for label_val, label_name, color in [
        (1, "Lemorzsolódik (1)", "#d62728"),
        (0, "Marad (0)", "#1f77b4"),
    ]:
        sub = df[df["actual_churn"] == label_val]["churn_proba"]
        fig_hist.add_trace(
            go.Histogram(
                x=sub,
                name=label_name,
                nbinsx=40,
                marker_color=color,
                opacity=0.65,
            )
        )

    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Küszöb: {threshold:.3f}",
        annotation_position="top right",
    )
    # Szürke zóna (0.30–0.70)
    fig_hist.add_vrect(
        x0=0.30, x1=0.70,
        fillcolor="orange", opacity=0.06,
        annotation_text="Szürke zóna", annotation_position="top left",
    )
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Churn valószínűség",
        yaxis_title="Ügyfelek száma",
        margin=dict(t=10, b=10, l=10, r=10),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Szürke zóna (0.30–0.70): bizonytalan döntési tartomány. "
        "A küszöb mozgatható a bal oldali csúszkával."
    )

with col_action:
    st.subheader("🎯 Akció-kategóriák megoszlása")

    action_counts = (
        df["action_dynamic"]
        .value_counts()
        .reindex(ACTION_ORDER)
        .fillna(0)
        .reset_index()
    )
    action_counts.columns = ["action", "count"]
    action_counts["pct"] = action_counts["count"] / len(df) * 100

    fig_funnel = go.Figure(
        go.Bar(
            x=action_counts["count"],
            y=action_counts["action"],
            orientation="h",
            marker_color=[ACTION_COLORS[a] for a in action_counts["action"]],
            text=action_counts.apply(lambda r: f"{int(r['count']):,} ({r['pct']:.1f}%)", axis=1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Ügyfelek: %{x:,}<extra></extra>",
        )
    )
    fig_funnel.update_layout(
        xaxis_title="Ügyfelek száma",
        margin=dict(t=10, b=10, l=20, r=120),
        height=360,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# ── 2. sor: RFM szegmens × Akció kereszttábla + Scatter ──
col_cross, col_bubble = st.columns([1, 1.2])

with col_cross:
    st.subheader("🔀 RFM szegmens × Akció kereszttábla")
    st.caption("Hány ügyfél esik az egyes szegmens-akció kombinációkba?")

    cross = pd.crosstab(
        df["rfm_segment"],
        df["action_dynamic"],
    ).reindex(columns=ACTION_ORDER, fill_value=0)

    # Százalékos heatmap
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig_heat = px.imshow(
        cross_pct.round(1),
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="Arány (%)"),
        aspect="auto",
    )
    fig_heat.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        xaxis_title="",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    # Rövidített tengelycímkék
    short_x = {
        "🚨 VIP Veszélyben – Azonnali Retenció":  "🚨 VIP Veszélyben",
        "⚠️  Lemorzsolódó – Win-Back Kampány":     "⚠️ Win-Back",
        "💎 VIP Stabil – Lojalitás Program":       "💎 VIP Stabil",
        "✅ Stabil – Standard Kommunikáció":        "✅ Stabil",
    }
    fig_heat.update_xaxes(
        ticktext=[short_x.get(c, c) for c in cross_pct.columns],
        tickvals=list(range(len(cross_pct.columns))),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Értékek: az adott RFM szegmens ügyféleinek százaléka az egyes akciókategóriákban")

with col_bubble:
    st.subheader("💰 Churn kockázat vs. Ügyfélérték")
    st.caption("X: churn valószínűség | Y: monetary (£, log) | Méret: frequency")

    bubble_df = filtered.copy()
    clip_val = bubble_df["monetary_total"].quantile(0.99)
    bubble_df = bubble_df[bubble_df["monetary_total"] <= clip_val]

    fig_bub = px.scatter(
        bubble_df,
        x="churn_proba",
        y="monetary_total",
        color="rfm_segment",
        size="frequency",
        size_max=18,
        opacity=0.55,
        log_y=True,
        color_discrete_map=SEGMENT_COLORS,
        labels={
            "churn_proba":    "Churn valószínűség",
            "monetary_total": "Monetary – £ (log)",
            "rfm_segment":    "RFM Szegmens",
            "frequency":      "Frequency",
        },
        hover_data={
            "churn_proba":    ":.1%",
            "monetary_total": ":,.0f",
            "recency_days":   True,
            "return_ratio":   ":.1%",
        },
    )
    fig_bub.add_vline(
        x=threshold, line_dash="dash", line_color="red",
        line_width=1.5,
        annotation_text=f"Küszöb {threshold:.3f}",
        annotation_position="top right",
    )
    fig_bub.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bub, use_container_width=True)

st.divider()

# ── 3. sor: TOP VIP Veszélyben lista ──
st.subheader("🚨 VIP Veszélyben – Prioritizált akciólista")

tab1, tab2 = st.tabs(["🚨 VIP Veszélyben (top 30)", "⚠️ Win-Back Kampány céllistája"])

with tab1:
    vip_list = (
        df[df["action_dynamic"] == "🚨 VIP Veszélyben – Azonnali Retenció"]
        .sort_values("churn_proba", ascending=False)
        .head(30)
        [["monetary_total", "frequency", "recency_days", "return_ratio",
          "rfm_segment", "churn_proba"]]
        .copy()
    )
    vip_list.index.name = "Customer ID"

    st.dataframe(
        vip_list.reset_index().style.format({
            "monetary_total": "£{:,.0f}",
            "churn_proba":    "{:.1%}",
            "recency_days":   "{:.0f} nap",
            "return_ratio":   "{:.1%}",
            "frequency":      "{:.0f}",
        })
        .background_gradient(subset=["churn_proba"], cmap="Reds")
        .background_gradient(subset=["monetary_total"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"💡 **{len(df[df['action_dynamic'] == '🚨 VIP Veszélyben – Azonnali Retenció']):,} ügyfél** "
        f"esik ebbe a kategóriába.  \n"
        f"Becsült veszélyeztetett bevétel: "
        f"**£{df[df['action_dynamic'] == '🚨 VIP Veszélyben – Azonnali Retenció']['monetary_total'].sum():,.0f}**"
    )

with tab2:
    winback_list = (
        df[df["action_dynamic"] == "⚠️  Lemorzsolódó – Win-Back Kampány"]
        .sort_values("churn_proba", ascending=False)
        .head(30)
        [["monetary_total", "frequency", "recency_days", "return_ratio",
          "rfm_segment", "churn_proba"]]
        .copy()
    )
    winback_list.index.name = "Customer ID"

    st.dataframe(
        winback_list.reset_index().style.format({
            "monetary_total": "£{:,.0f}",
            "churn_proba":    "{:.1%}",
            "recency_days":   "{:.0f} nap",
            "return_ratio":   "{:.1%}",
            "frequency":      "{:.0f}",
        })
        .background_gradient(subset=["churn_proba"], cmap="Oranges"),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Letöltés ──
st.subheader("⬇️ Exportálás kampányrendszerbe")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    export_vip = (
        df[df["action_dynamic"] == "🚨 VIP Veszélyben – Azonnali Retenció"]
        .sort_values("churn_proba", ascending=False)
        [["monetary_total", "frequency", "recency_days", "return_ratio",
          "rfm_segment", "churn_proba", "action_dynamic"]]
        .copy()
    )
    export_vip.index.name = "Customer ID"
    csv_vip = export_vip.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"🚨 VIP Veszélyben lista ({len(export_vip):,} ügyfél)",
        data=csv_vip,
        file_name="vip_veszélyben_akciólista.csv",
        mime="text/csv",
    )

with col_dl2:
    export_all = filtered[
        ["monetary_total", "frequency", "recency_days", "return_ratio",
         "rfm_segment", "churn_proba", "action_dynamic"]
    ].copy()
    export_all.index.name = "Customer ID"
    csv_all = export_all.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Teljes szűrt lista ({len(export_all):,} ügyfél)",
        data=csv_all,
        file_name="churn_predictions_filtered.csv",
        mime="text/csv",
    )

st.caption(
    f"⚙️ Modell: XGBoost (A: csak RFM pipeline) | PR-AUC: 0.8107 | "
    f"Optimális threshold: {BEST_THRESHOLD:.3f} | "
    f"Features: recency_days, frequency, monetary_total, monetary_avg, return_ratio"
)
