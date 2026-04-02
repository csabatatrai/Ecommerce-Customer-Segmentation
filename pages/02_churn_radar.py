"""
pages/02_churn_radar.py
────────────────────────
Churn Kockázati Radar – XGBoost előrejelzések, VIP veszélyben lista,
Precision-Recall görbe, churn eloszlás
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Konfiguráció ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Radar",
    page_icon="🚨",
    layout="wide",
)

BG_DARK  = "#0F1117"
BG_CARD  = "#1A1D27"
ACCENT   = "#E8472A"
AMBER    = "#F0A500"
TEAL     = "#00C9A7"
BLUE     = "#2AB5E8"
TEXT_PRI = "#F0F0F0"
TEXT_SEC = "#9FA3B1"

SEGMENT_COLORS = {
    "VIP Bajnokok":          "#E8472A",
    "Új / Ígéretes":         "#2AB5E8",
    "Lemorzsolódó / Alvó":   "#F0A500",
    "Elvesztett / Inaktív":  "#8A8A8A",
}

ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció":           "#E8472A",
    "💎 VIP Stabil – Lojalitás Program":               "#00C9A7",
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back":     "#F0A500",
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": "#8A8A8A",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG_DARK};
    color: {TEXT_PRI};
}}
.stApp {{ background-color: {BG_DARK}; }}

.page-header {{
    display: flex; align-items: flex-end; gap: 20px;
    padding: 32px 0 24px; border-bottom: 1px solid #2A2D3A; margin-bottom: 32px;
}}
.page-header h1 {{
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: {TEXT_PRI}; margin: 0; line-height: 1;
}}
.page-header .sub {{ font-size: 0.9rem; color: {TEXT_SEC}; padding-bottom: 4px; }}

.kpi-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A; border-radius: 12px;
    padding: 20px 22px; position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, {ACCENT});
}}
.kpi-label {{ font-size: 0.75rem; text-transform: uppercase;
              letter-spacing: .08em; color: {TEXT_SEC}; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'Syne', sans-serif;
              font-size: 2rem; font-weight: 800; color: {TEXT_PRI}; line-height: 1; }}
.kpi-delta {{ font-size: 0.78rem; color: {TEXT_SEC}; margin-top: 4px; }}

.section-title {{
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: {TEXT_PRI}; margin: 0 0 14px; letter-spacing: .03em;
}}
.chart-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A;
    border-radius: 14px; padding: 22px 20px; margin-bottom: 20px;
}}
.risk-badge {{
    display: inline-block; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ── Adatbetöltés ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_predictions():
    p = Path("data/processed/churn_predictions.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


df = load_predictions()

# ── Demo adatok ───────────────────────────────────────────────────────────────
if df is None:
    rng  = np.random.default_rng(42)
    n    = 5243
    segs = rng.choice(
        ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"],
        size=n, p=[0.09, 0.25, 0.28, 0.38]
    )
    monetary = rng.exponential(2000, n).clip(10, 450000)
    recency  = rng.integers(0, 646, n)
    freq     = rng.integers(1, 100, n)
    ret_ratio = rng.uniform(0, 0.5, n)

    # churn valószínűség: recency és alacsony freq → magasabb churn
    churn_proba = np.clip(
        0.3 + 0.0007 * recency - 0.008 * freq + rng.normal(0, 0.15, n), 0.01, 0.99
    )
    churn_pred  = (churn_proba >= 0.5).astype(int)
    actual_churn = rng.binomial(1, churn_proba)
    monetary_median = np.median(monetary)

    def assign_action(mon, cp):
        if mon > monetary_median and cp >= 0.5:
            return "🚨 VIP Veszélyben – Azonnali Retenció"
        elif mon > monetary_median:
            return "💎 VIP Stabil – Lojalitás Program"
        elif cp >= 0.5:
            return "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
        else:
            return "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

    actions = [assign_action(m, cp) for m, cp in zip(monetary, churn_proba)]

    df = pd.DataFrame({
        "Customer ID":    [f"C{i:05d}" for i in range(n)],
        "recency_days":   recency,
        "frequency":      freq,
        "monetary_total": monetary,
        "monetary_avg":   monetary / np.maximum(freq, 1),
        "return_ratio":   ret_ratio,
        "churn_proba":    churn_proba,
        "churn_pred":     churn_pred,
        "actual_churn":   actual_churn,
        "rfm_segment":    segs,
        "action":         actions,
    })
    st.info("ℹ️  Demo mód: `data/processed/churn_predictions.parquet` nem található.", icon="🔔")

# ── Oszlopnév normalizáció ────────────────────────────────────────────────────
rename_map = {}
for col in df.columns:
    lc = col.lower()
    if "segment" in lc and col != "rfm_segment":
        rename_map[col] = "rfm_segment"
    elif lc == "action" or lc == "akció":
        rename_map[col] = "action"
df = df.rename(columns=rename_map)

if "rfm_segment" not in df.columns:
    df["rfm_segment"] = "Ismeretlen"
if "action" not in df.columns:
    df["action"] = "Ismeretlen"

# ── FEJLÉC ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div><h1>🚨 Churn Radar</h1></div>
  <div class="sub">XGBoost churn előrejelzés · F1 ≈ 0.75 · Recall ≈ 0.745</div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR SZŰRŐ ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Szűrők")
    churn_thresh = st.slider("Churn küszöb (threshold)", 0.1, 0.9, 0.5, 0.05)
    min_rev = st.number_input("Min. bevétel (£)", value=0, step=100)
    show_segs = st.multiselect(
        "Szegmens szűrő",
        options=sorted(df["rfm_segment"].unique()),
        default=sorted(df["rfm_segment"].unique()),
    )

df_f = df[
    df["rfm_segment"].isin(show_segs) &
    (df["monetary_total"] >= min_rev)
].copy()
df_f["churn_pred_thresh"] = (df_f["churn_proba"] >= churn_thresh).astype(int)

# ── KPI sátor ─────────────────────────────────────────────────────────────────
churn_rate    = df_f["churn_pred_thresh"].mean() * 100
vip_at_risk   = df_f[df_f["action"] == "🚨 VIP Veszélyben – Azonnali Retenció"]
vip_risk_rev  = vip_at_risk["monetary_total"].sum()
avg_churn_p   = df_f["churn_proba"].mean() * 100
high_risk_n   = (df_f["churn_proba"] > 0.8).sum()

kpis = [
    ("Becsült churn arány",     f"{churn_rate:.1f}%",         f"küszöb: {churn_thresh:.2f}",   ACCENT),
    ("VIP veszélyben (bevétel)", f"£{vip_risk_rev/1e3:.0f}K",  f"{len(vip_at_risk)} ügyfél",    AMBER),
    ("Átl. churn valószínűség", f"{avg_churn_p:.1f}%",         "a szűrt halmazon",              BLUE),
    ("Kritikus kockázat (>80%)", f"{high_risk_n:,}",            "azonnali figyelmet igényel",   "#FF6B6B"),
]

cols = st.columns(len(kpis))
for col, (label, val, delta, color) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# ── SOR 1: Scatter + Action Donut ─────────────────────────────────────────────
col_l, col_r = st.columns([1.7, 1])

with col_l:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ügyfél kockázati térkép (Bevétel × Churn valószínűség)</div>', unsafe_allow_html=True)

    sample = df_f.sample(min(3000, len(df_f)), random_state=42)

    fig_map = go.Figure()

    # Veszélyzóna háttér
    fig_map.add_shape(
        type="rect", x0=churn_thresh, x1=1, y0=0, y1=sample["monetary_total"].max(),
        fillcolor="rgba(232,71,42,0.05)", line=dict(width=0), layer="below",
    )
    fig_map.add_vline(x=churn_thresh, line=dict(color=ACCENT, width=1.5, dash="dash"),
                      annotation_text=f"Threshold ({churn_thresh})",
                      annotation_font_color=ACCENT, annotation_font_size=11)
    fig_map.add_hline(y=sample["monetary_total"].median(),
                      line=dict(color="#2A2D3A", width=1, dash="dot"))

    for seg in sample["rfm_segment"].unique():
        sub = sample[sample["rfm_segment"] == seg]
        fig_map.add_trace(go.Scatter(
            x=sub["churn_proba"], y=sub["monetary_total"],
            mode="markers",
            name=seg,
            marker=dict(
                color=SEGMENT_COLORS.get(seg, "#888"),
                size=6, opacity=0.65,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{seg}</b><br>"
                "Churn: %{x:.1%}<br>Bevétel: £%{y:,.0f}<extra></extra>"
            ),
        ))

    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRI, family="DM Sans"),
        xaxis=dict(title="Churn valószínűség", tickformat=".0%",
                   showgrid=True, gridcolor="#2A2D3A", zeroline=False),
        yaxis=dict(title="Nettó bevétel (£, log)", type="log",
                   showgrid=True, gridcolor="#2A2D3A", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT_SEC)),
        margin=dict(l=0, r=0, t=10, b=0), height=370,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Akció-szegmensek</div>', unsafe_allow_html=True)

    action_counts = df_f["action"].value_counts().reset_index()
    action_counts.columns = ["action","count"]

    short_labels = {
        "🚨 VIP Veszélyben – Azonnali Retenció":           "🚨 VIP Veszélyben",
        "💎 VIP Stabil – Lojalitás Program":               "💎 VIP Stabil",
        "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back":     "⚠️ Win-Back",
        "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": "✅ Stabil",
    }
    action_counts["short"] = action_counts["action"].map(short_labels).fillna(action_counts["action"])

    fig_donut = go.Figure(go.Pie(
        labels=action_counts["short"],
        values=action_counts["count"],
        hole=0.55,
        marker=dict(
            colors=[ACTION_COLORS.get(a, "#666") for a in action_counts["action"]],
            line=dict(color=BG_DARK, width=3),
        ),
        textfont=dict(family="DM Sans", size=10),
        hovertemplate="<b>%{label}</b><br>%{value:,} ügyfél (%{percent})<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{len(df_f):,}</b><br><span style='font-size:10px'>ügyfél</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=TEXT_PRI, family="Syne"),
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRI, family="DM Sans"),
        legend=dict(orientation="v", x=1, y=0.5,
                    font=dict(size=10, color=TEXT_SEC), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=10, t=10, b=10), height=370,
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── CHURN ELOSZLÁS szegmensenként ─────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Churn valószínűség eloszlása szegmensenként</div>', unsafe_allow_html=True)

fig_violin = go.Figure()
for seg in sorted(df_f["rfm_segment"].unique()):
    sub = df_f[df_f["rfm_segment"] == seg]["churn_proba"]
    color = SEGMENT_COLORS.get(seg, "#888")
    fig_violin.add_trace(go.Violin(
        y=sub, name=seg,
        fillcolor=color.replace("#", "rgba(").replace(")", ",0.25)") if color.startswith("#") else color,
        line_color=color, line_width=2,
        meanline_visible=True,
        box_visible=True,
        box_fillcolor=color,
        points=False,
    ))

fig_violin.add_hline(y=churn_thresh, line=dict(color=ACCENT, width=1.5, dash="dash"),
                     annotation_text=f"Threshold {churn_thresh}",
                     annotation_font_color=ACCENT, annotation_font_size=11)
fig_violin.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="DM Sans"),
    yaxis=dict(title="Churn valószínűség", showgrid=True, gridcolor="#2A2D3A",
               range=[0, 1], tickformat=".0%"),
    xaxis=dict(showgrid=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=11)),
    margin=dict(l=0, r=0, t=10, b=0), height=330,
)
st.plotly_chart(fig_violin, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── VIP VESZÉLYBEN LISTA ──────────────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚨 VIP Veszélyben – Legsürgősebb 30 ügyfél (bevétel × kockázat)</div>', unsafe_allow_html=True)

vip_sorted = df_f[
    df_f["churn_proba"] >= churn_thresh
].sort_values(["monetary_total", "churn_proba"], ascending=[False, False]).head(30)

if len(vip_sorted) == 0:
    st.warning("Nincs a szűrésnek megfelelő ügyfél ebben a kockázati tartományban.")
else:
    display_cols = {
        "Customer ID": "Ügyfél",
        "rfm_segment": "Szegmens",
        "monetary_total": "Bevétel (£)",
        "frequency": "Vásárlás (db)",
        "recency_days": "Inaktivitás (nap)",
        "churn_proba": "Churn valószínűség",
        "return_ratio": "Visszaküldési arány",
    }
    avail_cols = [c for c in display_cols if c in vip_sorted.columns]
    show_df = vip_sorted[avail_cols].rename(columns=display_cols)

    fmt = {
        "Bevétel (£)": "£{:,.0f}",
        "Churn valószínűség": "{:.1%}",
        "Visszaküldési arány": "{:.1%}",
    }
    fmt = {k: v for k, v in fmt.items() if k in show_df.columns}

    st.dataframe(
        show_df.style
        .format(fmt)
        .background_gradient(subset=["Churn valószínűség"], cmap="Reds")
        .background_gradient(subset=["Bevétel (£)"], cmap="YlOrBr"),
        use_container_width=True, hide_index=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── AJÁNLÁSOK ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#1A1D27;border:1px solid #2A2D3A;border-left:4px solid {ACCENT};
            border-radius:10px;padding:18px 22px;margin-top:8px;">
  <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:10px;color:{TEXT_PRI}">
    💡 Javasolt akciók a VIP Veszélyben szegmensre
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;color:{TEXT_SEC};font-size:0.88rem;line-height:1.7">
    <div>
      <b style="color:{TEXT_PRI}">1. Személyes megkeresés</b><br>
      Account manager telefonhívás / email a top 20 VIP-nek.
    </div>
    <div>
      <b style="color:{TEXT_PRI}">2. Win-back kupon</b><br>
      Exkluzív 15–20%-os visszatérési kedvezmény, limitált érvénnyel.
    </div>
    <div>
      <b style="color:{TEXT_PRI}">3. Email sorozat</b><br>
      3 üzenet, 2 hetes intervallummal (értékalapú kommunikáció).
    </div>
    <div>
      <b style="color:{TEXT_PRI}">4. NPS felmérés</b><br>
      Proaktív panaszkezelés és elégedettség-mérés kiküldése.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
