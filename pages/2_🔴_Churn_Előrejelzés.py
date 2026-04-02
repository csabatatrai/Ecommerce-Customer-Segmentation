"""
pages/2_🔴_Churn_Előrejelzés.py
XGBoost Churn Prediction – üzleti akciótérkép és VIP-lista
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Előrejelzés",
    page_icon="🔴",
    layout="wide",
)

# ── Konstansok ────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció": "#e74c3c",
    "💎 VIP Stabil – Lojalitás Program":     "#2ecc71",
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back": "#e67e22",
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": "#3498db",
}

ACTION_ADVICE = {
    "🚨 VIP Veszélyben – Azonnali Retenció": [
        "Személyes account manager megkeresés (ha B2B ügyfél)",
        "Exkluzív visszatérési kupon (15–20% kedvezmény)",
        "Win-back email sorozat (3 üzenet, 2 hetes intervallummal)",
        "NPS felmérés küldése (proaktív panaszkezelés)",
    ],
    "💎 VIP Stabil – Lojalitás Program": [
        "VIP hűségkártya vagy pontgyűjtő program ajánlata",
        "Korai hozzáférés új termékekhez / exkluzív ajánlatok",
        "Születésnapi / évfordulós személyes üzenet",
        "Upselling & cross-selling: prémium termékcsaládok",
    ],
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back": [
        "Alacsony értékű win-back kupon (5–10% kedvezmény)",
        "Szezonális vagy akciós hírlevél küldése",
        "Automatikus reaktivációs e-mail trigger (6 hónap inaktivitás után)",
    ],
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": [
        "Általános hírlevél és promóciók",
        "Kosárelhagyás emlékeztető e-mail",
        "Termékajánló algoritmus aktiválása (collaborative filtering)",
    ],
}

SEGMENT_COLORS = {
    "VIP Bajnokok":         "#2ecc71",
    "Lemorzsolódó / Alvó":  "#e67e22",
    "Elvesztett / Inaktív": "#e74c3c",
    "Új / Ígéretes":        "#3498db",
    "Ismeretlen":           "#95a5a6",
    "N/A":                  "#95a5a6",
}

# ── Adat betöltő ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Churn előrejelzések betöltése...")
def load_predictions() -> pd.DataFrame:
    candidates = [
        Path("data/processed/churn_predictions.parquet"),
        Path("../data/processed/churn_predictions.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)

    # ── Szintetikus demo adatok ───────────────────────────────────────────────
    st.warning(
        "⚠️ **Demo mód** – `churn_predictions.parquet` nem található. "
        "Szintetikus adatokkal dolgozunk.",
        icon="🔔",
    )
    rng = np.random.default_rng(42)
    n = 5243
    recency = rng.integers(0, 650, n)
    frequency = rng.integers(1, 50, n)
    monetary = np.exp(rng.uniform(4, 13, n))
    return_ratio = rng.uniform(0, 0.4, n)
    monetary_avg = monetary / frequency

    # Churn valószínűség szimulálása (recency-alapú logika)
    churn_log = (
        0.012 * recency
        - 0.05 * np.log1p(frequency)
        - 0.03 * np.log1p(monetary)
        + 0.8 * return_ratio
        + rng.normal(0, 0.3, n)
    )
    churn_proba = 1 / (1 + np.exp(-churn_log))
    churn_pred = (churn_proba >= 0.5).astype(int)
    actual_churn = (churn_proba + rng.normal(0, 0.15, n) >= 0.5).astype(int)

    monetary_median = np.median(monetary)
    high_value = monetary > monetary_median
    high_churn = churn_proba > 0.5

    def _action(hv, hc):
        if hv and hc:
            return "🚨 VIP Veszélyben – Azonnali Retenció"
        elif hv and not hc:
            return "💎 VIP Stabil – Lojalitás Program"
        elif not hv and hc:
            return "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
        else:
            return "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

    actions = [_action(h, c) for h, c in zip(high_value, high_churn)]
    rfm_segs = rng.choice(
        ["VIP Bajnokok", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív", "Új / Ígéretes"],
        size=n, p=[0.10, 0.25, 0.45, 0.20],
    )

    df = pd.DataFrame(
        {
            "recency_days":   recency,
            "frequency":      frequency,
            "monetary_total": monetary,
            "monetary_avg":   monetary_avg,
            "return_ratio":   return_ratio,
            "churn_proba":    churn_proba,
            "churn_pred":     churn_pred,
            "actual_churn":   actual_churn,
            "rfm_segment":    rfm_segs,
            "action":         actions,
        },
        index=[str(12346 + i) for i in range(n)],
    )
    df.index.name = "Customer ID"
    return df


# ── Fő layout ─────────────────────────────────────────────────────────────────
st.title("🔴 Churn Előrejelzés – XGBoost Pipeline")
st.caption(
    "Modell: XGBoost (A: Csak RFM) | 5-fold StratifiedKFold CV | "
    "Metrika: PR-AUC, F1, Recall | Churn arány: 55.7%"
)
st.markdown("---")

df = load_predictions()

# ── Oldalsáv ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Szűrők")
    threshold = st.slider(
        "Churn küszöbérték (threshold)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Ennél magasabb churn_proba → lemorzsolódónak minősített",
    )
    selected_actions = st.multiselect(
        "Akció kategória",
        options=list(ACTION_COLORS.keys()),
        default=list(ACTION_COLORS.keys()),
    )
    monetary_min = st.number_input(
        "Min. nettó bevétel (£)", min_value=0, value=0, step=100
    )

# Küszöb újraszámítás
monetary_median = df["monetary_total"].median()
df = df.copy()
df["churn_pred_dyn"] = (df["churn_proba"] >= threshold).astype(int)

def assign_action(row):
    hv = row["monetary_total"] > monetary_median
    hc = row["churn_proba"] >= threshold
    if hv and hc:   return "🚨 VIP Veszélyben – Azonnali Retenció"
    elif hv:        return "💎 VIP Stabil – Lojalitás Program"
    elif hc:        return "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
    else:           return "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

df["action_dyn"] = df.apply(assign_action, axis=1)

mask = (
    df["action_dyn"].isin(selected_actions)
    & (df["monetary_total"] >= monetary_min)
)
filtered = df[mask].copy()

# ── KPI kártyák ───────────────────────────────────────────────────────────────
st.subheader("📌 Kulcsmutatók")
total = len(df)
churned = (df["churn_proba"] >= threshold).sum()
vip_at_risk = ((df["monetary_total"] > monetary_median) & (df["churn_proba"] >= threshold)).sum()
avg_proba = df["churn_proba"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Összes ügyfél", f"{total:,}")
c2.metric("Lemorzsolódó (előrejelzett)", f"{churned:,}", f"{churned/total*100:.1f}%")
c3.metric("VIP Veszélyben 🚨", f"{vip_at_risk:,}")
c4.metric("Átl. churn valószínűség", f"{avg_proba:.1%}")
c5.metric("Threshold", f"{threshold:.0%}")

st.markdown("---")

# ── Akció-eloszlás + Churn proba hisztogram ───────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Akció-kategóriák eloszlása")
    action_counts = df["action_dyn"].value_counts().reset_index()
    action_counts.columns = ["Akció", "Ügyfelek száma"]
    action_counts["Arány (%)"] = (action_counts["Ügyfelek száma"] / len(df) * 100).round(1)
    fig_bar = px.bar(
        action_counts,
        x="Ügyfelek száma",
        y="Akció",
        orientation="h",
        color="Akció",
        color_discrete_map=ACTION_COLORS,
        text="Arány (%)",
        title="Ügyfelek megoszlása akció-kategóriák szerint",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(showlegend=False, height=350, xaxis_title="Ügyfelek száma", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Churn valószínűség eloszlása")
    fig_hist = px.histogram(
        df,
        x="churn_proba",
        color="action_dyn",
        color_discrete_map=ACTION_COLORS,
        nbins=40,
        title="Churn valószínűség eloszlása szegmensenként",
        labels={"churn_proba": "Churn valószínűség", "action_dyn": "Akció"},
    )
    fig_hist.add_vline(
        x=threshold, line_dash="dash", line_color="black",
        annotation_text=f"Threshold = {threshold:.0%}",
        annotation_position="top right",
    )
    fig_hist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ── Churn × RFM Szegmens kereszttábla ────────────────────────────────────────
if "rfm_segment" in df.columns:
    st.subheader("🔀 RFM Szegmens × Churn Kockázat")
    cross_tab = pd.crosstab(
        df["rfm_segment"],
        df["action_dyn"],
        margins=True,
        margins_name="Összesen",
    )
    pct_tab = cross_tab.div(cross_tab["Összesen"], axis=0).mul(100).round(1)

    tab1, tab2 = st.tabs(["Abszolút értékek", "Arányok (%)"])
    with tab1:
        st.dataframe(cross_tab, use_container_width=True)
    with tab2:
        st.dataframe(
            pct_tab.style.background_gradient(
                subset=[c for c in pct_tab.columns if "Veszélyben" in c or "Win-Back" in c],
                cmap="Reds",
            ).format("{:.1f}%"),
            use_container_width=True,
        )
    st.markdown("---")

# ── Churn térkép: Monetary vs Churn Proba ────────────────────────────────────
st.subheader("🗺️ Churn kockázati térkép – Értékesség vs Lemorzsolódás")
sample = filtered.sample(min(len(filtered), 2000), random_state=42)
fig_risk = px.scatter(
    sample,
    x="monetary_total",
    y="churn_proba",
    color="action_dyn",
    color_discrete_map=ACTION_COLORS,
    opacity=0.65,
    log_x=True,
    labels={
        "monetary_total": "Nettó bevétel – log skála (£)",
        "churn_proba": "Churn valószínűség",
        "action_dyn": "Akció",
    },
    title=f"Churn kockázati mátrix (minta: {len(sample):,} ügyfél)",
    hover_data=["recency_days", "frequency"],
)
fig_risk.add_hline(y=threshold, line_dash="dot", line_color="grey",
                   annotation_text=f"Threshold = {threshold:.0%}")
fig_risk.add_vline(x=monetary_median, line_dash="dot", line_color="grey",
                   annotation_text=f"Medián bevétel = £{monetary_median:,.0f}")
fig_risk.update_layout(height=480, legend_title="Akció")
st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# ── VIP Veszélyben – TOP 30 ───────────────────────────────────────────────────
st.subheader("🚨 VIP Veszélyben – TOP 30 legmagasabb kockázatú ügyfél")
vip_risk_df = (
    df[df["action_dyn"] == "🚨 VIP Veszélyben – Azonnali Retenció"]
    .sort_values("churn_proba", ascending=False)
    .head(30)
)

if len(vip_risk_df):
    display_vip = vip_risk_df[
        ["monetary_total", "frequency", "recency_days", "return_ratio", "churn_proba"]
    ].rename(
        columns={
            "monetary_total": "Nettó bevétel (£)",
            "frequency": "Vásárlások száma",
            "recency_days": "Recency (nap)",
            "return_ratio": "Return arány",
            "churn_proba": "Churn valószínűség",
        }
    )
    st.dataframe(
        display_vip.style
        .background_gradient(subset=["Churn valószínűség"], cmap="Reds")
        .background_gradient(subset=["Nettó bevétel (£)"], cmap="Greens")
        .format({
            "Nettó bevétel (£)": "£{:,.0f}",
            "Churn valószínűség": "{:.1%}",
            "Return arány": "{:.1%}",
        }),
        use_container_width=True,
        height=340,
    )
    # Akciótanácsok
    with st.expander("💡 Javasolt akciók a VIP Veszélyben szegmensre"):
        for tip in ACTION_ADVICE["🚨 VIP Veszélyben – Azonnali Retenció"]:
            st.markdown(f"• {tip}")
else:
    st.info("Nincs ügyfél ebben a kategóriában a jelenlegi szűrők mellett.")

st.markdown("---")

# ── Modell teljesítmény ────────────────────────────────────────────────────────
st.subheader("📈 Modell teljesítmény (keresztvalidáció)")
col_a, col_b = st.columns(2)

cv_data = pd.DataFrame({
    "Modell": ["A: Csak RFM", "B: RFM + K-Means OHE"],
    "F1 (átlag)": [0.7502, 0.7499],
    "F1 (szórás)": [0.0114, 0.0156],
    "Recall (átlag)": [0.745, 0.743],
    "Recall (szórás)": [0.0264, 0.0340],
})

with col_a:
    st.markdown("**5-fold StratifiedKFold CV eredmények:**")
    st.dataframe(
        cv_data.set_index("Modell").style
        .background_gradient(subset=["F1 (átlag)", "Recall (átlag)"], cmap="Blues"),
        use_container_width=True,
    )

with col_b:
    fig_cv = go.Figure()
    for _, row in cv_data.iterrows():
        fig_cv.add_trace(go.Bar(
            name=row["Modell"],
            x=["F1 Score", "Recall"],
            y=[row["F1 (átlag)"], row["Recall (átlag)"]],
            error_y=dict(
                type="data",
                array=[row["F1 (szórás)"], row["Recall (szórás)"]],
                visible=True,
            ),
        ))
    fig_cv.update_layout(
        barmode="group",
        title="CV metrikák összehasonlítása (±szórás)",
        yaxis=dict(range=[0.6, 0.85]),
        height=300,
        legend_title="Modell",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

st.caption(
    "🏆 **Nyertes modell:** A – Csak RFM | "
    "Az imbalanced osztályok (55.7% churn) miatt PR-AUC és Recall a fő metrikák."
)

# ── Letöltés ──────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📥 Szűrt előrejelzések letöltése"):
    dl_cols = [c for c in [
        "recency_days", "frequency", "monetary_total", "return_ratio",
        "churn_proba", "churn_pred_dyn", "rfm_segment", "action_dyn"
    ] if c in filtered.columns]
    csv = filtered[dl_cols].reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ CSV letöltése (szűrt churn előrejelzések)",
        data=csv,
        file_name="churn_elorejelzesek.csv",
        mime="text/csv",
    )
    st.dataframe(filtered[dl_cols].head(200), use_container_width=True)
