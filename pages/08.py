# ============================================================
# pages/08.py
# ============================================================
# Üzleti Érték & Bevételi Hatás – Pareto-elemzés, szegmens
# bevételi részesedés, veszélyeztetett bevétel, piac-térkép,
# visszaküldési eróziós hatás és kampánytervező összefoglaló
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

from config import (
    CUSTOMER_SEGMENTS_PARQUET,
    CHURN_PREDICTIONS_PARQUET,
    CLEANED_PARQUET,
)

st.set_page_config(
    page_title="Üzleti Érték & Bevételi Hatás",
    page_icon="💼",
    layout="wide",
)

# ── Konstansok ──
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

ACTION_ORDER = [
    "🚨 VIP Veszélyben – Azonnali Retenció",
    "⚠️  Lemorzsolódó – Win-Back Kampány",
    "💎 VIP Stabil – Lojalitás Program",
    "✅ Stabil – Standard Kommunikáció",
]

BEST_THRESHOLD = 0.419


# ── Adatbetöltés ──
@st.cache_data
def load_merged() -> pd.DataFrame:
    seg = pd.read_parquet(CUSTOMER_SEGMENTS_PARQUET)
    churn = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
    merged = seg.join(
        churn[["churn_proba", "churn_pred", "actual_churn", "action"]],
        how="inner",
    )
    return merged


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    df = pd.read_parquet(CLEANED_PARQUET)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["revenue"] = df["Quantity"] * df["Price"]
    return df


try:
    df = load_merged()
    cleaned = load_cleaned()
except FileNotFoundError as e:
    st.error(f"❌ Hiányzó adatfájl: {e}\n\nFuttasd le az összes notebookot!")
    st.stop()


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 💼 Üzleti Érték")
    st.markdown("---")

    all_segments = sorted(df["Segment"].unique())
    selected_segments = st.multiselect(
        "Szegmensek szűrése",
        options=all_segments,
        default=all_segments,
    )

    st.markdown("**Churn küszöb**")
    threshold = st.slider(
        "Küszöbérték",
        min_value=0.10,
        max_value=0.90,
        value=BEST_THRESHOLD,
        step=0.01,
        format="%.2f",
        help="Ennél magasabb churn_proba esetén az ügyfél veszélyeztetettnek számít.",
    )

    st.markdown("---")
    st.caption("Adatforrás: Online Retail II (UCI)")


# ── Levezetett oszlopok ──
dff = df[df["Segment"].isin(selected_segments)].copy()
dff["at_risk"] = dff["churn_proba"] >= threshold
dff["net_revenue"] = dff["monetary_total"] * (1 - dff["return_ratio"])
dff["return_erosion"] = dff["monetary_total"] - dff["net_revenue"]


# ── KPI KÁRTYÁK ──
st.title("💼 Üzleti Érték & Bevételi Hatás")
st.markdown(
    "Pareto-elemzés, szegmens bevételi részesedés, veszélyeztetett összegek, "
    "piac-térkép, visszaküldési eróziós hatás és kampánytervező összefoglaló."
)
st.markdown("---")

total_rev       = dff["monetary_total"].sum()
net_total_rev   = dff["net_revenue"].sum()
at_risk_rev     = dff.loc[dff["at_risk"], "monetary_total"].sum()
erosion_total   = dff["return_erosion"].sum()
pct_at_risk     = at_risk_rev / total_rev * 100 if total_rev > 0 else 0
pct_erosion     = erosion_total / total_rev * 100 if total_rev > 0 else 0

# Pareto: hány % ügyfél adja a bevétel 80%-át
sorted_df       = dff.sort_values("monetary_total", ascending=False).copy()
sorted_df["cum_rev_pct"] = sorted_df["monetary_total"].cumsum() / total_rev * 100
pareto_80_pct   = (sorted_df["cum_rev_pct"] < 80).sum() / len(sorted_df) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Bruttó bevétel",    f"£{total_rev:,.0f}")
c2.metric(
    "🔥 Veszélyeztetett bevétel",
    f"£{at_risk_rev:,.0f}",
    delta=f"−{pct_at_risk:.1f}% a teljes bevételből",
    delta_color="inverse",
)
c3.metric(
    "↩️ Visszaküldési eróziós hatás",
    f"£{erosion_total:,.0f}",
    delta=f"−{pct_erosion:.1f}% veszteség",
    delta_color="inverse",
)
c4.metric(
    "📐 Pareto-arány",
    f"{pareto_80_pct:.0f}% ügyfél",
    delta="adja a bevétel 80%-át",
    delta_color="off",
)

st.markdown("---")


# ══════════════════════════════════════════════════════════
# 1. sor: Szegmens bevételi kördiagram + Veszélyeztetett bevétel
# ══════════════════════════════════════════════════════════
col_pie, col_risk = st.columns(2)

with col_pie:
    st.subheader("🍩 Szegmens bevételi részesedés")
    st.caption("Bruttó bevétel megoszlása szegmensek szerint")

    seg_rev = (
        dff.groupby("Segment")["monetary_total"]
        .sum()
        .reset_index()
        .sort_values("monetary_total", ascending=False)
    )

    fig_pie = px.pie(
        seg_rev,
        names="Segment",
        values="monetary_total",
        hole=0.45,
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig_pie.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Bevétel: £%{value:,.0f}<br>Arány: %{percent}<extra></extra>",
    )
    fig_pie.update_layout(
        height=360,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(
            text=f"£{total_rev/1e6:.1f}M",
            x=0.5, y=0.5, font_size=18, showarrow=False,
        )],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_risk:
    st.subheader("⚠️ Veszélyeztetett bevétel szegmensenként")
    st.caption(f"Biztonságos vs. veszélyeztetett összeg (küszöb: {threshold:.2f})")

    safe_rev = dff[~dff["at_risk"]].groupby("Segment")["monetary_total"].sum()
    risk_rev = dff[dff["at_risk"]].groupby("Segment")["monetary_total"].sum()
    seg_stacked = pd.DataFrame({"safe": safe_rev, "risk": risk_rev}).fillna(0).reset_index()
    seg_stacked = seg_stacked.sort_values("risk", ascending=True)

    fig_risk = go.Figure()
    fig_risk.add_bar(
        name="Biztonságos",
        y=seg_stacked["Segment"],
        x=seg_stacked["safe"],
        orientation="h",
        marker_color=[SEGMENT_COLORS.get(s, "#aaa") for s in seg_stacked["Segment"]],
        opacity=0.35,
        hovertemplate="<b>%{y}</b><br>Biztonságos: £%{x:,.0f}<extra></extra>",
    )
    fig_risk.add_bar(
        name="Veszélyeztetett",
        y=seg_stacked["Segment"],
        x=seg_stacked["risk"],
        orientation="h",
        marker_color=[SEGMENT_COLORS.get(s, "#aaa") for s in seg_stacked["Segment"]],
        opacity=1.0,
        hovertemplate="<b>%{y}</b><br>Veszélyeztetett: £%{x:,.0f}<extra></extra>",
    )
    fig_risk.update_layout(
        barmode="stack",
        xaxis_title="Bevétel (£)",
        yaxis_title="",
        height=360,
        margin=dict(t=10, b=40, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════
# 2. sor: Pareto-elemzés + Országos bevételi térkép
# ══════════════════════════════════════════════════════════
col_pareto, col_country = st.columns(2)

with col_pareto:
    st.subheader("📐 Pareto-elemzés – Bevétel koncentráció")
    st.caption("A legértékesebb ügyfelek kumulatív bevételi részesedése")

    pareto_df = dff.sort_values("monetary_total", ascending=False).copy().reset_index()
    pareto_df["customer_pct"] = np.arange(1, len(pareto_df) + 1) / len(pareto_df) * 100
    pareto_df["rev_cum_pct"]  = pareto_df["monetary_total"].cumsum() / total_rev * 100

    fig_pareto = go.Figure()
    fig_pareto.add_scatter(
        x=pareto_df["customer_pct"],
        y=pareto_df["rev_cum_pct"],
        mode="lines",
        name="Kumulatív bevétel %",
        line=dict(color="#009E73", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,158,115,0.12)",
    )
    # 80/20 referenciavonal
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="#E69F00", line_width=1.5,
                         annotation_text="80%", annotation_position="left")
    fig_pareto.add_vline(x=pareto_80_pct, line_dash="dash", line_color="#d62728", line_width=1.5,
                         annotation_text=f"{pareto_80_pct:.0f}%",
                         annotation_position="bottom right")
    fig_pareto.add_annotation(
        x=pareto_80_pct + 5,
        y=70,
        text=f"<b>Top {pareto_80_pct:.0f}% ügyfél<br>= 80% bevétel</b>",
        showarrow=False,
        font=dict(size=12, color="#d62728"),
        align="left",
    )
    fig_pareto.update_layout(
        xaxis_title="Ügyfelek (kumulatív %)",
        yaxis_title="Bevétel (kumulatív %)",
        height=360,
        margin=dict(t=10, b=40, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_country:
    st.subheader("🌍 Top piacok bevétel szerint")
    st.caption("Teljes tranzakciós bevétel országonként (top 10)")

    country_rev = (
        cleaned.groupby("Country")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    country_rev["pct"] = country_rev["revenue"] / country_rev["revenue"].sum() * 100
    country_rev_sorted = country_rev.sort_values("revenue", ascending=True)

    fig_country = px.bar(
        country_rev_sorted,
        x="revenue",
        y="Country",
        orientation="h",
        color="revenue",
        color_continuous_scale="Teal",
        text=country_rev_sorted["revenue"].apply(lambda v: f"£{v/1e3:.0f}k"),
        labels={"revenue": "Bevétel (£)", "Country": ""},
    )
    fig_country.update_traces(textposition="outside")
    fig_country.update_layout(
        height=360,
        margin=dict(t=10, b=40, l=10, r=10),
        coloraxis_showscale=False,
        xaxis_title="Bevétel (£)",
    )
    st.plotly_chart(fig_country, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════
# 3. sor: Visszaküldési eróziós hatás + Kampánytervező
# ══════════════════════════════════════════════════════════
col_erosion, col_campaign = st.columns([1, 1.1])

with col_erosion:
    st.subheader("↩️ Visszaküldési eróziós hatás szegmensenként")
    st.caption("Mennyit von le a tényleges bevételből a visszaküldési arány?")

    erosion_stats = (
        dff.groupby("Segment")
        .agg(
            gross=("monetary_total", "sum"),
            net=("net_revenue", "sum"),
        )
        .reset_index()
    )
    erosion_stats["erosion"] = erosion_stats["gross"] - erosion_stats["net"]
    erosion_stats["erosion_pct"] = erosion_stats["erosion"] / erosion_stats["gross"] * 100
    erosion_stats = erosion_stats.sort_values("erosion", ascending=True)

    fig_erosion = go.Figure()
    fig_erosion.add_bar(
        name="Nettó bevétel",
        y=erosion_stats["Segment"],
        x=erosion_stats["net"],
        orientation="h",
        marker_color=[SEGMENT_COLORS.get(s, "#aaa") for s in erosion_stats["Segment"]],
        opacity=0.9,
        hovertemplate="<b>%{y}</b><br>Nettó bevétel: £%{x:,.0f}<extra></extra>",
    )
    fig_erosion.add_bar(
        name="Visszaküldési veszteség",
        y=erosion_stats["Segment"],
        x=erosion_stats["erosion"],
        orientation="h",
        marker_color="#d62728",
        opacity=0.7,
        hovertemplate="<b>%{y}</b><br>Visszaküldési veszteség: £%{x:,.0f}<extra></extra>",
    )
    # Százalékos annotációk
    for _, row in erosion_stats.iterrows():
        fig_erosion.add_annotation(
            x=row["gross"] * 1.02,
            y=row["Segment"],
            text=f"−{row['erosion_pct']:.1f}%",
            showarrow=False,
            font=dict(color="#d62728", size=11),
            xanchor="left",
        )
    fig_erosion.update_layout(
        barmode="stack",
        xaxis_title="Bevétel (£)",
        yaxis_title="",
        height=360,
        margin=dict(t=10, b=40, l=10, r=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_erosion, use_container_width=True)

with col_campaign:
    st.subheader("💰 Kampányakció – Bevételi tét szegmensenként")
    st.caption("Összesített bruttó bevétel szegmens × kampányakció bontásban")

    heatmap_data = (
        dff.groupby(["Segment", "action"])["monetary_total"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=ACTION_ORDER, fill_value=0)
    )
    # Csak azok a szegmensek, amik szerepelnek
    heatmap_data = heatmap_data.reindex(
        [s for s in all_segments if s in heatmap_data.index]
    )

    short_action = {
        "🚨 VIP Veszélyben – Azonnali Retenció":  "🚨 VIP Veszélyben",
        "⚠️  Lemorzsolódó – Win-Back Kampány":     "⚠️ Win-Back",
        "💎 VIP Stabil – Lojalitás Program":       "💎 VIP Stabil",
        "✅ Stabil – Standard Kommunikáció":        "✅ Stabil",
    }
    heatmap_display = heatmap_data.copy()
    heatmap_display.columns = [short_action.get(c, c) for c in heatmap_display.columns]

    fig_heat = px.imshow(
        heatmap_display.values,
        x=heatmap_display.columns.tolist(),
        y=heatmap_display.index.tolist(),
        color_continuous_scale="YlOrRd",
        text_auto=False,
        labels=dict(color="Bevétel (£)"),
        aspect="auto",
    )
    # Cellák szöveges értékei
    for i, seg_name in enumerate(heatmap_display.index):
        for j, act_name in enumerate(heatmap_display.columns):
            val = heatmap_display.iloc[i, j]
            fig_heat.add_annotation(
                x=j, y=i,
                text=f"£{val/1e3:.0f}k" if val >= 1000 else f"£{val:.0f}",
                showarrow=False,
                font=dict(size=11, color="black"),
            )
    fig_heat.update_layout(
        height=360,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════
# 4. Kampánytervező összefoglaló táblázat
# ══════════════════════════════════════════════════════════
st.subheader("📋 Kampánytervező összefoglaló")
st.caption(
    "Szegmens × kampányakció kombinációk: ügyfelek, bruttó bevétel, "
    "átl. churn kockázat és átl. visszaküldési arány."
)

summary = (
    dff.groupby(["Segment", "action"])
    .agg(
        Ügyfelek=("monetary_total", "count"),
        Összbevétel=("monetary_total", "sum"),
        Nettó_bevétel=("net_revenue", "sum"),
        Átl_churn=("churn_proba", "mean"),
        Átl_visszaküldés=("return_ratio", "mean"),
    )
    .reset_index()
)
summary["action"] = pd.Categorical(summary["action"], categories=ACTION_ORDER, ordered=True)
summary = summary.sort_values(["action", "Összbevétel"], ascending=[True, False])
summary = summary.rename(columns={
    "Segment":           "Szegmens",
    "action":            "Kampányakció",
    "Összbevétel":       "Összbevétel (£)",
    "Nettó_bevétel":     "Nettó bevétel (£)",
    "Átl_churn":         "Átl. Churn %",
    "Átl_visszaküldés":  "Átl. Visszaküldés %",
})
summary["Összbevétel (£)"]     = summary["Összbevétel (£)"].round(0).astype(int)
summary["Nettó bevétel (£)"]   = summary["Nettó bevétel (£)"].round(0).astype(int)
summary["Átl. Churn %"]        = (summary["Átl. Churn %"] * 100).round(1)
summary["Átl. Visszaküldés %"] = (summary["Átl. Visszaküldés %"] * 100).round(1)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Átl. Churn %": st.column_config.ProgressColumn(
            "Átl. Churn %",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Átl. Visszaküldés %": st.column_config.ProgressColumn(
            "Átl. Visszaküldés %",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Összbevétel (£)": st.column_config.NumberColumn(
            "Összbevétel (£)", format="£%d",
        ),
        "Nettó bevétel (£)": st.column_config.NumberColumn(
            "Nettó bevétel (£)", format="£%d",
        ),
    },
)
