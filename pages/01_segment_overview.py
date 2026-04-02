"""
pages/01_segment_overview.py
────────────────────────────
RFM Szegmens Áttekintő – üzleti KPI-ok, szegmens-profil, értékesség
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Konfiguráció ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFM Szegmentáció",
    page_icon="🎯",
    layout="wide",
)

# ── Szín- és stílusdefiníciók ─────────────────────────────────────────────────
SEGMENT_COLORS = {
    "VIP Bajnokok":          "#E8472A",   # tűzvörös
    "Új / Ígéretes":         "#2AB5E8",   # ég kék
    "Lemorzsolódó / Alvó":   "#F0A500",   # borostyán
    "Elvesztett / Inaktív":  "#8A8A8A",   # palaszürke
}

BG_DARK  = "#0F1117"
BG_CARD  = "#1A1D27"
ACCENT   = "#E8472A"
TEXT_PRI = "#F0F0F0"
TEXT_SEC = "#9FA3B1"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG_DARK};
    color: {TEXT_PRI};
}}

.stApp {{ background-color: {BG_DARK}; }}

/* ── fejléc ── */
.page-header {{
    display: flex; align-items: flex-end; gap: 20px;
    padding: 32px 0 24px;
    border-bottom: 1px solid #2A2D3A;
    margin-bottom: 32px;
}}
.page-header h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    color: {TEXT_PRI}; margin: 0; line-height: 1;
}}
.page-header .sub {{
    font-size: 0.9rem; color: {TEXT_SEC};
    padding-bottom: 4px;
}}

/* ── KPI kártyák ── */
.kpi-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 160px;
    background: {BG_CARD};
    border: 1px solid #2A2D3A;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, {ACCENT});
}}
.kpi-label {{ font-size: 0.75rem; text-transform: uppercase;
              letter-spacing: .08em; color: {TEXT_SEC}; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'Syne', sans-serif;
              font-size: 2rem; font-weight: 800; color: {TEXT_PRI}; line-height: 1; }}
.kpi-delta {{ font-size: 0.78rem; color: {TEXT_SEC}; margin-top: 4px; }}

/* ── szekció-fejléc ── */
.section-title {{
    font-family: 'Syne', sans-serif; font-size: 1.05rem;
    font-weight: 700; color: {TEXT_PRI};
    margin: 0 0 14px; letter-spacing: .03em;
}}

/* ── kártya-konténer ── */
.chart-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A;
    border-radius: 14px; padding: 22px 20px; margin-bottom: 20px;
}}

/* ── szegmens-badge ── */
.seg-badge {{
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 500;
}}

/* Streamlit gombok, selectbox */
div[data-testid="stSelectbox"] > div {{ background: {BG_CARD}; border-color: #2A2D3A; }}
</style>
""", unsafe_allow_html=True)

# ── Adatbetöltés ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_segments():
    p = Path("data/processed/customer_segments.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)

    # Oszlopnév-normalizáció
    rename = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "_")
        if "segment" in lc or "szegmens" in lc:
            rename[col] = "Segment"
        elif "cluster" in lc:
            rename[col] = "cluster"
    df = df.rename(columns=rename)

    # Ha még nincs Segment, próbáljuk a 'cluster' alapján rekonstruálni
    if "Segment" not in df.columns and "cluster" in df.columns:
        mapping = {
            0: "VIP Bajnokok",
            1: "Új / Ígéretes",
            2: "Lemorzsolódó / Alvó",
            3: "Elvesztett / Inaktív",
        }
        df["Segment"] = df["cluster"].map(mapping).fillna("Ismeretlen")

    return df


@st.cache_data(show_spinner=False)
def load_predictions():
    p = Path("data/processed/churn_predictions.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


df_seg  = load_segments()
df_pred = load_predictions()

# ── Demo adatok, ha nincs parquet ─────────────────────────────────────────────
if df_seg is None:
    rng = np.random.default_rng(42)
    n   = 5243
    segs = rng.choice(
        ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"],
        size=n, p=[0.09, 0.25, 0.28, 0.38]
    )
    df_seg = pd.DataFrame({
        "Customer ID":   [f"C{i:05d}" for i in range(n)],
        "recency_days":  rng.integers(0, 646, n),
        "frequency":     rng.integers(1, 100, n),
        "monetary_total": rng.exponential(2000, n).clip(10, 450000),
        "return_ratio":  rng.uniform(0, 0.5, n),
        "monetary_avg":  rng.exponential(300, n).clip(10, 13000),
        "Segment":       segs,
        "cluster":       rng.integers(0, 4, n),
    })
    st.info("ℹ️  Demo mód: `data/processed/customer_segments.parquet` nem található.", icon="🔔")

# ── FEJLÉC ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div>
    <h1>🎯 RFM Szegmentáció</h1>
  </div>
  <div class="sub">K-Means alapú ügyfélcsoportosítás · Megfigyelési ablak: 2009-12-01 → 2011-09-09</div>
</div>
""", unsafe_allow_html=True)

# ── KPI SÁTOR ─────────────────────────────────────────────────────────────────
total_customers  = len(df_seg)
total_revenue    = df_seg["monetary_total"].sum()
avg_basket       = df_seg["monetary_avg"].mean()
avg_recency      = df_seg["recency_days"].mean()
vip_count        = (df_seg["Segment"] == "VIP Bajnokok").sum()
vip_revenue      = df_seg[df_seg["Segment"] == "VIP Bajnokok"]["monetary_total"].sum()
vip_rev_pct      = vip_revenue / total_revenue * 100

kpi_data = [
    ("Egyedi ügyfelek",     f"{total_customers:,}",       "szegmentált ügyfél",         ACCENT),
    ("Teljes nettó bevétel", f"£{total_revenue/1e6:.2f}M", "megfigyelési ablakban",      "#2AB5E8"),
    ("Átlag kosárérték",    f"£{avg_basket:,.0f}",        "ügyfélenkénti átlag",         "#F0A500"),
    ("Átlag inaktivitás",   f"{avg_recency:.0f} nap",     "utolsó vásárlás óta",         "#8A8A8A"),
    ("VIP bevétel-arány",   f"{vip_rev_pct:.1f}%",        f"{vip_count} VIP ügyfél adja","#E8472A"),
]

cols = st.columns(len(kpi_data))
for col, (label, val, delta, color) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# ── SZŰRŐ ─────────────────────────────────────────────────────────────────────
all_segs = sorted(df_seg["Segment"].unique())
with st.sidebar:
    st.markdown("### 🔧 Szűrők")
    selected_segs = st.multiselect(
        "Szegmensek megjelenítése",
        options=all_segs, default=all_segs
    )
    min_mon, max_mon = int(df_seg["monetary_total"].min()), int(df_seg["monetary_total"].max())
    mon_range = st.slider("Monetáris érték (£)", min_mon, min(max_mon, 50000), (min_mon, min(max_mon, 50000)))

df_f = df_seg[
    df_seg["Segment"].isin(selected_segs) &
    df_seg["monetary_total"].between(*mon_range)
]

# ── SOROK: PIE + SNAKE ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Szegmens-megoszlás</div>', unsafe_allow_html=True)

    seg_counts = df_f.groupby("Segment").agg(
        count=("monetary_total", "count"),
        revenue=("monetary_total", "sum"),
    ).reset_index()

    fig_pie = go.Figure(go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["count"],
        hole=0.52,
        marker=dict(
            colors=[SEGMENT_COLORS.get(s, "#666") for s in seg_counts["Segment"]],
            line=dict(color=BG_DARK, width=3),
        ),
        textfont=dict(family="DM Sans", size=12),
        hovertemplate="<b>%{label}</b><br>Ügyfelek: %{value:,}<br>Arány: %{percent}<extra></extra>",
    ))
    fig_pie.add_annotation(
        text=f"<b>{len(df_f):,}</b><br><span style='font-size:11px'>ügyfél</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=TEXT_PRI, family="Syne"),
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRI, family="DM Sans"),
        legend=dict(
            orientation="v", x=1, y=0.5,
            font=dict(size=11, color=TEXT_SEC),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=10, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Szegmens-profil (Snake Plot – standardizált RFM)</div>', unsafe_allow_html=True)

    profile = df_f.groupby("Segment")[["recency_days","frequency","monetary_total"]].mean()
    # Standardizálás vizualizációhoz (0-1 skálára normálizáljuk)
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig_snake = go.Figure()
    metrics_display = ["recency_days","frequency","monetary_total"]
    metrics_labels  = ["Recency (napok)","Frequency (számlák)","Monetary (£)"]

    for seg in profile_norm.index:
        color = SEGMENT_COLORS.get(seg, "#888")
        fig_snake.add_trace(go.Scatter(
            x=metrics_labels,
            y=profile_norm.loc[seg, metrics_display].values,
            mode="lines+markers",
            name=seg,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color=BG_DARK, width=2)),
            hovertemplate=f"<b>{seg}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
        ))

    fig_snake.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRI, family="DM Sans"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="#2A2D3A", zeroline=False,
                   title="Normalizált érték", tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_SEC)),
        margin=dict(l=0, r=0, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_snake, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── SZEGMENS TÁBLÁZAT ─────────────────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Szegmens-összefoglaló tábla</div>', unsafe_allow_html=True)

summary = df_f.groupby("Segment").agg(
    Ügyfelek=("monetary_total","count"),
    Bevétel_GBP=("monetary_total","sum"),
    Átlag_bevétel=("monetary_total","mean"),
    Átlag_recency=("recency_days","mean"),
    Átlag_frequency=("frequency","mean"),
    Átlag_return_ratio=("return_ratio","mean"),
).reset_index()
summary["Bevétel arány (%)"] = (summary["Bevétel_GBP"] / summary["Bevétel_GBP"].sum() * 100).round(1)
summary = summary.sort_values("Bevétel_GBP", ascending=False)

# Formázott megjelenítés
display_df = summary.rename(columns={
    "Bevétel_GBP": "Nettó bevétel (£)",
    "Átlag_bevétel": "Átl. értékesség (£)",
    "Átlag_recency": "Átl. inaktivitás (nap)",
    "Átlag_frequency": "Átl. vásárlás (db)",
    "Átlag_return_ratio": "Átl. visszaküldési arány",
})

st.dataframe(
    display_df.style
    .format({
        "Nettó bevétel (£)": "£{:,.0f}",
        "Átl. értékesség (£)": "£{:,.0f}",
        "Átl. inaktivitás (nap)": "{:.0f}",
        "Átl. vásárlás (db)": "{:.1f}",
        "Átl. visszaküldési arány": "{:.1%}",
        "Bevétel arány (%)": "{:.1f}%",
    })
    .background_gradient(subset=["Nettó bevétel (£)"], cmap="Reds"),
    use_container_width=True, hide_index=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── SCATTER: Recency vs Monetary ──────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Recency × Monetary értéktérkép (szegmensenként)</div>', unsafe_allow_html=True)

sample = df_f.sample(min(2000, len(df_f)), random_state=42)
fig_scatter = px.scatter(
    sample,
    x="recency_days", y="monetary_total",
    color="Segment",
    color_discrete_map=SEGMENT_COLORS,
    size="frequency",
    size_max=18,
    opacity=0.72,
    labels={
        "recency_days": "Utolsó vásárlás óta (nap)",
        "monetary_total": "Nettó bevétel (£)",
        "frequency": "Vásárlás száma",
    },
    hover_data={"recency_days": True, "monetary_total": ":.0f", "frequency": True},
)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="DM Sans"),
    xaxis=dict(showgrid=True, gridcolor="#2A2D3A", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#2A2D3A", zeroline=False,
               type="log", title="Nettó bevétel (£, log skála)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=11)),
    margin=dict(l=0, r=0, t=10, b=0),
    height=380,
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── INSIGHT BOX ───────────────────────────────────────────────────────────────
vip_row = summary[summary["Segment"] == "VIP Bajnokok"]
if not vip_row.empty:
    vip_pct = vip_row["Bevétel arány (%)"].values[0]
    vip_cnt = vip_row["Ügyfelek"].values[0]
    vip_cust_pct = vip_cnt / len(df_f) * 100
    st.markdown(f"""
    <div style="background:#1A1D27;border:1px solid #2A2D3A;border-left:4px solid {ACCENT};
                border-radius:10px;padding:18px 22px;margin-top:8px;">
      <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:8px;color:{TEXT_PRI}">
        💡 Üzleti Insight
      </div>
      <div style="color:{TEXT_SEC};font-size:0.9rem;line-height:1.6">
        A <b style="color:{ACCENT}">VIP Bajnokok</b> szegmens az ügyfelek
        <b style="color:{TEXT_PRI}">{vip_cust_pct:.1f}%</b>-át teszi ki,
        mégis a teljes bevétel <b style="color:{TEXT_PRI}">{vip_pct:.1f}%</b>-át generálja.
        Legmagasabb visszaküldési arányuk (≈16%) tipikus B2B/B2C VIP viselkedés —
        ők ismerik legjobban a visszaküldési politikát. Lojalitásprogrammal megőrzésük
        kiemelt prioritás.
      </div>
    </div>
    """, unsafe_allow_html=True)
