# ============================================================
# brutal_report.py  –  BRUTÁLISAN ŐSZINTE VÁSÁRLÓI RIPORT
# ============================================================
# Futtatás:  streamlit run brutal_report.py
# Stílus:    Noir editorial · ipari grid · vörös tinta a fehér lapon
# Szándék:   Az adatok nem hazudnak. Ez a riport sem.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Oldal konfig ────────────────────────────────────────────
st.set_page_config(
    page_title="BRUTALREPORT · Customer Autopsy",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════
#  GLOBÁLIS CSS  –  Noir Editorial stílus
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;600&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');

/* ── Reset ── */
html, body, [class*="css"], .stApp {
    background-color: #0a0a0a !important;
    color: #e8e0d0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none; }
div[data-testid="stVerticalBlock"] > div { padding: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #cc1a2e; }

/* ── Masthead ── */
.masthead {
    border-bottom: 3px solid #cc1a2e;
    padding: 32px 60px 20px 60px;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: end;
    gap: 20px;
    background:
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 59px,
            rgba(204,26,46,0.06) 59px,
            rgba(204,26,46,0.06) 60px
        );
}
.masthead-left {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    line-height: 1.8;
}
.masthead-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 7vw, 6rem);
    letter-spacing: 0.04em;
    color: #e8e0d0;
    text-align: center;
    line-height: 0.9;
}
.masthead-title span { color: #cc1a2e; }
.masthead-right {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    line-height: 1.8;
    text-align: right;
}
.issue-bar {
    background: #cc1a2e;
    padding: 5px 60px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #fff;
}

/* ── Section divider ── */
.section-rule {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 0;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: #cc1a2e;
    padding: 12px 60px 4px 60px;
}

/* ── Verdict cards ── */
.verdicts {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid #1e1e1e;
}
.verdict-card {
    padding: 28px 32px;
    border-right: 1px solid #1e1e1e;
    position: relative;
    overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #cc1a2e;
}
.verdict-card:last-child { border-right: none; }
.verdict-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    line-height: 1;
    color: #e8e0d0;
}
.verdict-number.red { color: #cc1a2e; }
.verdict-number.dim { color: #444; }
.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    margin-top: 6px;
}
.verdict-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #333;
    margin-top: 12px;
    line-height: 1.5;
}

/* ── Headline statement ── */
.headline-statement {
    padding: 40px 60px;
    border-bottom: 1px solid #1e1e1e;
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 60px;
    align-items: start;
}
.hs-dropcap {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-style: italic;
    color: #cc1a2e;
    float: left;
    line-height: 0.8;
    margin-right: 12px;
    margin-top: 8px;
}
.hs-body {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    line-height: 1.75;
    color: #c8c0b0;
    font-style: italic;
}
.hs-pullquote {
    border-left: 3px solid #cc1a2e;
    padding-left: 20px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    line-height: 1.3;
    color: #e8e0d0;
    letter-spacing: 0.03em;
}
.hs-pullquote span { color: #cc1a2e; }

/* ── Chart grid ── */
.chart-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid #1e1e1e;
}
.chart-cell {
    padding: 24px 32px 0 32px;
    border-right: 1px solid #1e1e1e;
}
.chart-cell:last-child { border-right: none; }
.chart-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.06em;
    color: #e8e0d0;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 8px;
    margin-bottom: 4px;
}
.chart-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Full-width chart ── */
.chart-full {
    padding: 24px 60px 0 60px;
    border-bottom: 1px solid #1e1e1e;
}

/* ── Obituary table ── */
.obit-section {
    padding: 32px 60px;
    border-bottom: 1px solid #1e1e1e;
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 60px;
}
.obit-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.05em;
    color: #e8e0d0;
    border-bottom: 2px solid #cc1a2e;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.obit-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    line-height: 1.9;
    color: #555;
}
.obit-text strong { color: #cc1a2e; font-weight: 600; }

/* ── Obit row ── */
.obit-row {
    border-bottom: 1px solid #1a1a1a;
    padding: 10px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    line-height: 1.7;
    color: #444;
}
.obit-row-id {
    color: #cc1a2e;
    font-size: 0.85rem;
    font-weight: bold;
}
.obit-row-val {
    color: #e8e0d0;
    font-weight: bold;
}
.obit-row-churn {
    color: #cc1a2e;
    font-weight: bold;
}
.obit-row-action {
    color: #333;
}
.obit-col-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #cc1a2e;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

/* ── Footer ── */
.brutal-footer {
    padding: 20px 60px;
    border-top: 3px solid #1e1e1e;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brutal-footer-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    color: #2a2a2a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* ── Streamlit plot containers ── */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  ADAT BETÖLTÉS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_or_simulate():
    try:
        from config import (
            CUSTOMER_SEGMENTS_PARQUET, CHURN_PREDICTIONS_PARQUET,
            READY_FOR_RFM_PARQUET
        )
        seg   = pd.read_parquet(CUSTOMER_SEGMENTS_PARQUET)
        churn = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
        raw   = pd.read_parquet(READY_FOR_RFM_PARQUET,
                                columns=["InvoiceDate", "Price", "Quantity", "Customer ID"])

        # Szegmens oszlop normalizálás
        for col in ["Segment", "rfm_segment", "Szegmens"]:
            if col in seg.columns:
                seg = seg.rename(columns={col: "segment_label"})
                break
        if "segment_label" not in seg.columns:
            seg["segment_label"] = "Ismeretlen"

        for col in ["Segment", "rfm_segment", "Szegmens"]:
            if col in churn.columns:
                churn = churn.rename(columns={col: "segment_label"})
                break
        if "segment_label" not in churn.columns:
            churn["segment_label"] = "Ismeretlen"

        if "Customer ID" not in seg.columns:
            seg = seg.reset_index().rename(columns={"index": "Customer ID"})
        if "Customer ID" not in churn.columns:
            churn = churn.reset_index().rename(columns={"index": "Customer ID"})
        if "monetary_total" not in churn.columns and "monetary_total" in seg.columns:
            churn = churn.join(seg.set_index("Customer ID")[["monetary_total"]], on="Customer ID", how="left")

        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
        raw["revenue"] = raw["Price"] * raw["Quantity"]
        ts = raw.groupby(raw["InvoiceDate"].dt.to_period("M").dt.to_timestamp())["revenue"].sum().reset_index()
        ts.columns = ["month", "revenue"]

        return seg, churn, ts, False

    except Exception:
        return _simulate()


def _simulate():
    rng = np.random.default_rng(99)
    n = 4200
    seg_names = ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"]
    probs     = [0.11, 0.26, 0.27, 0.36]
    labels    = np.random.choice(seg_names, n, p=probs)

    monetary = np.where(labels == "VIP Bajnokok",          rng.uniform(4000, 30000, n),
               np.where(labels == "Új / Ígéretes",         rng.uniform(600,  5000,  n),
               np.where(labels == "Lemorzsolódó / Alvó",   rng.uniform(150,  1800,  n),
                                                            rng.uniform(30,   400,   n))))
    recency  = np.where(labels == "VIP Bajnokok",          rng.integers(3,  40,  n),
               np.where(labels == "Új / Ígéretes",         rng.integers(15, 90,  n),
               np.where(labels == "Lemorzsolódó / Alvó",   rng.integers(70, 220, n),
                                                            rng.integers(160, 400, n))))
    freq     = np.where(labels == "VIP Bajnokok",          rng.integers(18, 70, n),
               np.where(labels == "Új / Ígéretes",         rng.integers(5,  18, n),
               np.where(labels == "Lemorzsolódó / Alvó",   rng.integers(2,  7,  n),
                                                            rng.integers(1,  3,  n))))
    churn_p  = np.where(labels == "VIP Bajnokok",          rng.uniform(0.02, 0.20, n),
               np.where(labels == "Új / Ígéretes",         rng.uniform(0.12, 0.42, n),
               np.where(labels == "Lemorzsolódó / Alvó",   rng.uniform(0.48, 0.88, n),
                                                            rng.uniform(0.68, 0.99, n))))

    seg = pd.DataFrame({
        "Customer ID":   np.arange(10000, 10000 + n),
        "segment_label": labels,
        "monetary_total":monetary,
        "recency_days":  recency,
        "frequency":     freq,
    })
    churn = seg.copy()
    churn["churn_proba"] = churn_p

    dates  = pd.date_range("2009-12-01", "2011-11-30", freq="MS")
    trend  = np.linspace(0.75, 1.25, len(dates))
    sea    = 1 + 0.4 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 - np.pi / 2)
    noise  = rng.uniform(0.88, 1.12, len(dates))
    rev    = 210_000 * trend * sea * noise
    rev[np.array([d.month for d in dates]) == 11] *= 1.55
    rev[np.array([d.month for d in dates]) == 12] *= 2.0
    ts = pd.DataFrame({"month": dates, "revenue": rev})

    return seg, churn, ts, True


seg, churn, ts, IS_DEMO = load_or_simulate()

# ── Alap metrikák ────────────────────────────────────────────
total_customers  = len(seg)
total_revenue    = seg["monetary_total"].sum()
churn_rate_pct   = (churn["churn_proba"] >= 0.5).mean() * 100
vip_count        = (seg["segment_label"] == "VIP Bajnokok").sum()
vip_revenue_pct  = (
    seg[seg["segment_label"] == "VIP Bajnokok"]["monetary_total"].sum()
    / total_revenue * 100
)
lost_count       = (seg["segment_label"] == "Elvesztett / Inaktív").sum()
lost_revenue     = (
    seg[seg["segment_label"] == "Elvesztett / Inaktív"]["monetary_total"].sum()
)
vip_at_risk      = churn[
    (churn["churn_proba"] >= 0.5) & (churn["monetary_total"] >= 2000)
].shape[0]
rev_at_risk      = churn[churn["churn_proba"] >= 0.5]["monetary_total"].sum()
avg_recency      = seg["recency_days"].mean()

import datetime
today_str = datetime.date.today().strftime("%d %B %Y").upper()


# ════════════════════════════════════════════════════════════
#  PLOTLY ALAP LAYOUT
# ════════════════════════════════════════════════════════════
NOIR = dict(
    font=dict(family="'IBM Plex Mono', monospace", color="#888"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=16, b=16, l=8, r=8),
    xaxis=dict(gridcolor="#161616", linecolor="#222", zerolinecolor="#222",
               tickfont=dict(size=9, color="#444")),
    yaxis=dict(gridcolor="#161616", linecolor="#222", zerolinecolor="#222",
               tickfont=dict(size=9, color="#444")),
)

RED  = "#cc1a2e"
BONE = "#e8e0d0"
DIM  = "#2a2a2a"
COAL = "#161616"


# ════════════════════════════════════════════════════════════
#  MASTHEAD
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="masthead">
    <div class="masthead-left">
        Kiadó: Data Intelligence Bureau<br>
        Forrás: Online Retail II · UCI / Kaggle<br>
        Időszak: 2009–2011 · UK<br>
        {'⬛ BEMUTATÓ MÓD' if IS_DEMO else '⬛ ÉLŐADATOK'}
    </div>
    <div class="masthead-title">CUSTOMER<br><span>AUTOPSY</span></div>
    <div class="masthead-right">
        {today_str}<br>
        Vol. III · Iss. 01<br>
        {total_customers:,} ügyfél boncolva<br>
        Bizalmas · Üzleti használatra
    </div>
</div>
<div class="issue-bar">
    ▸ &nbsp; Az igazság nem fáj. A nemtudás igen. &nbsp; ◂ &nbsp;&nbsp;&nbsp;
    Churn ráta: {churn_rate_pct:.1f}% &nbsp;·&nbsp;
    VIP-ek a bevétel {vip_revenue_pct:.0f}%-át termelik &nbsp;·&nbsp;
    {vip_at_risk} VIP ügyfél ma éjjel eltűnhet &nbsp;·&nbsp;
    £{rev_at_risk:,.0f} veszélyben
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  VERDICT CARDS
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="verdicts">
    <div class="verdict-card">
        <div class="verdict-number">{total_customers:,}</div>
        <div class="verdict-label">Ügyfél a rendszerben</div>
        <div class="verdict-sub">
            {vip_count:,} VIP · {lost_count:,} elveszett<br>
            Átl. recency: {avg_recency:.0f} nap
        </div>
    </div>
    <div class="verdict-card">
        <div class="verdict-number red">£{total_revenue/1e6:.1f}M</div>
        <div class="verdict-label">Összbevétel</div>
        <div class="verdict-sub">
            Top {vip_revenue_pct:.0f}% VIP-től<br>
            Elveszett: £{lost_revenue/1e3:.0f}K
        </div>
    </div>
    <div class="verdict-card">
        <div class="verdict-number red">{churn_rate_pct:.1f}%</div>
        <div class="verdict-label">Churn valószínűség ≥ 50%</div>
        <div class="verdict-sub">
            {vip_at_risk} VIP veszélyben<br>
            £{rev_at_risk/1e3:.0f}K forog kockán
        </div>
    </div>
    <div class="verdict-card">
        <div class="verdict-number dim">{lost_count:,}</div>
        <div class="verdict-label">Már-már elveszett</div>
        <div class="verdict-sub">
            Lemorzsolódó + Inaktív<br>
            Reaktiváció: most vagy soha
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HEADLINE STATEMENT
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="headline-statement">
    <div>
        <div class="hs-body">
            <span class="hs-dropcap">A</span>
            z adatok ritkán hazudnak — és ez a riport sem fog.
            Az ügyfélbázis <strong style="color:#cc1a2e">{vip_revenue_pct:.0f}%-a</strong>
            a bevétel oroszlánrészét termeli, miközben a fennmaradó
            <strong style="color:#cc1a2e">{100-vip_revenue_pct:.0f}%</strong>
            az infrastruktúra költségeit emészti. Jelenleg
            <strong style="color:#cc1a2e">{churn_rate_pct:.1f}%</strong>
            ügyfél valószínűsíthetően nem tér vissza — köztük
            <strong style="color:#cc1a2e">{vip_at_risk} VIP profil</strong>,
            akik nélkül a havi bevétel érezhetően megcsappan.
            Az XGBoost modell megnevezte őket. Rajtunk áll, mit kezdünk velük.
        </div>
    </div>
    <div>
        <div class="hs-pullquote">
            A VIP-ek<br>
            <span>{vip_revenue_pct:.0f}%</span>-ot<br>
            termelnek.<br>
            Te tudtad?
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CHART ROW 1:  Bevételi trend  +  Churn eloszlás
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ I. — Bevétel és kockázat időben</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("""
    <div class="chart-cell" style="border-right:none;padding-left:60px">
        <div class="chart-title">HAVI BEVÉTEL — A KÖNYÖRTELEN TREND</div>
        <div class="chart-subtitle">2009–2011 · mozgóátlag 3 hó · szezonális kitörések jelölve</div>
    </div>
    """, unsafe_allow_html=True)

    ts["MA3"]  = ts["revenue"].rolling(3).mean()
    ts["MA6"]  = ts["revenue"].rolling(6).mean()
    ts["peak"] = ts["revenue"] > ts["revenue"].quantile(0.90)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Bar(
        x=ts["month"], y=ts["revenue"],
        marker_color=[RED if p else "#1a1a1a" for p in ts["peak"]],
        marker_line_width=0,
        name="Havi bevétel",
        hovertemplate="<b>%{x|%Y %b}</b><br>£%{y:,.0f}<extra></extra>",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["month"], y=ts["MA3"],
        line=dict(color=BONE, width=1.5, dash="solid"),
        mode="lines", name="3 hó MA",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["month"], y=ts["MA6"],
        line=dict(color="#444", width=1, dash="dot"),
        mode="lines", name="6 hó MA",
    ))
    fig_ts.update_layout(**NOIR, height=280,
                          legend=dict(orientation="h", y=1.12, x=0,
                                      font=dict(size=9, color="#444"),
                                      bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_ts, use_container_width=True)

with col_r:
    st.markdown("""
    <div class="chart-cell" style="border-left:1px solid #1e1e1e;border-right:none;padding-right:60px">
        <div class="chart-title">CHURN VALÓSZÍNŰSÉG — SZEGMENSENKÉNT</div>
        <div class="chart-subtitle">Violin eloszlás · minden pont egy ügyfél · vörös = ≥ 50%</div>
    </div>
    """, unsafe_allow_html=True)

    seg_order = ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"]
    seg_colors = ["#1a1a1a", "#222", RED, RED]

    fig_viol = go.Figure()
    for i, (seg_name, color) in enumerate(zip(seg_order, seg_colors)):
        sub = churn[churn["segment_label"] == seg_name]["churn_proba"]
        if len(sub) == 0:
            continue
        fig_viol.add_trace(go.Violin(
            x=[seg_name] * len(sub),
            y=sub,
            name=seg_name,
            fillcolor=color,
            line_color="#333" if i < 2 else RED,
            opacity=0.85,
            box_visible=True,
            meanline_visible=True,
            meanline_color=BONE,
            box_line_color="#555",
            points=False,
            showlegend=False,
        ))
    fig_viol.add_hline(y=0.5, line_color=RED, line_dash="dash", line_width=1,
                       annotation_text="Küszöb 0.50", annotation_font_color=RED,
                       annotation_font_size=9)
    layout_v = {**NOIR}
    layout_v["height"] = 280
    layout_v["yaxis"] = dict(**NOIR["yaxis"], range=[0, 1], tickformat=".0%")
    fig_viol.update_layout(**layout_v)
    st.plotly_chart(fig_viol, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  CHART ROW 2:  Pareto  +  Scatter kockázati térkép
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ II. — A 80/20 törvény és a kockázati térkép</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="chart-cell" style="border-right:none;padding-left:60px">
        <div class="chart-title">PARETO — KI TERMELI A PÉNZT?</div>
        <div class="chart-subtitle">Kumulált bevétel az ügyfelek rangsorában · 80% vonal jelölve</div>
    </div>
    """, unsafe_allow_html=True)

    sorted_seg = seg.sort_values("monetary_total", ascending=False).reset_index(drop=True)
    sorted_seg["cum_rev"]  = sorted_seg["monetary_total"].cumsum()
    sorted_seg["cum_pct"]  = sorted_seg["cum_rev"] / sorted_seg["monetary_total"].sum() * 100
    sorted_seg["cust_pct"] = (sorted_seg.index + 1) / len(sorted_seg) * 100

    pareto_x = sorted_seg[sorted_seg["cum_pct"] >= 80]["cust_pct"].iloc[0]

    fig_par = go.Figure()
    mask = sorted_seg["cust_pct"] <= pareto_x
    fig_par.add_trace(go.Scatter(
        x=sorted_seg[mask]["cust_pct"],
        y=sorted_seg[mask]["cum_pct"],
        mode="lines", line=dict(color=RED, width=2),
        fill="tozeroy", fillcolor="rgba(204,26,46,0.15)",
        name="Top ügyfelek", showlegend=False,
        hovertemplate="Top %{x:.1f}% → %{y:.1f}%<extra></extra>",
    ))
    fig_par.add_trace(go.Scatter(
        x=sorted_seg[~mask]["cust_pct"],
        y=sorted_seg[~mask]["cum_pct"],
        mode="lines", line=dict(color="#333", width=1.5),
        fill="tozeroy", fillcolor="rgba(30,30,30,0.5)",
        name="Többiek", showlegend=False,
    ))
    fig_par.add_vline(x=pareto_x, line_color=RED, line_dash="dash", line_width=1)
    fig_par.add_hline(y=80, line_color="#333", line_dash="dot", line_width=1)
    fig_par.add_annotation(
        x=pareto_x + 2, y=45,
        text=f"<b>TOP {pareto_x:.1f}%</b><br>→ 80% bevétel",
        showarrow=False,
        font=dict(size=10, color=RED, family="IBM Plex Mono"),
        align="left",
    )
    layout_p = {**NOIR, "height": 280}
    layout_p["xaxis"] = dict(**NOIR["xaxis"], title="Ügyfelek (kumulált %)", ticksuffix="%")
    layout_p["yaxis"] = dict(**NOIR["yaxis"], title="Bevétel (kumulált %)", ticksuffix="%")
    fig_par.update_layout(**layout_p)
    st.plotly_chart(fig_par, use_container_width=True)

with col_b:
    st.markdown("""
    <div class="chart-cell" style="border-left:1px solid #1e1e1e;border-right:none;padding-right:60px">
        <div class="chart-title">KOCKÁZATI TÉRKÉP — CHURN vs ÉRTÉK</div>
        <div class="chart-subtitle">Minden pont egy ügyfél · vörös = magas kockázat · jobb felső = 🚨</div>
    </div>
    """, unsafe_allow_html=True)

    sample_c = churn.sample(min(2000, len(churn)), random_state=42)
    colors_c  = [RED if p >= 0.5 else "#1e1e1e" for p in sample_c["churn_proba"]]
    opacities = [0.8 if p >= 0.5 else 0.4 for p in sample_c["churn_proba"]]

    fig_scat = go.Figure()
    low = sample_c[sample_c["churn_proba"] < 0.5]
    fig_scat.add_trace(go.Scatter(
        x=low["churn_proba"], y=low["monetary_total"],
        mode="markers",
        marker=dict(color="#1e1e1e", size=4, line=dict(color="#333", width=0.5)),
        name="Alacsony kockázat",
        hovertemplate="Churn: %{x:.2f}<br>£%{y:,.0f}<extra></extra>",
    ))
    high = sample_c[sample_c["churn_proba"] >= 0.5]
    fig_scat.add_trace(go.Scatter(
        x=high["churn_proba"], y=high["monetary_total"],
        mode="markers",
        marker=dict(color=RED, size=5, opacity=0.75,
                    line=dict(color="rgba(204,26,46,0.3)", width=1)),
        name="Magas kockázat",
        hovertemplate="Churn: %{x:.2f}<br>£%{y:,.0f}<extra></extra>",
    ))
    fig_scat.add_vline(x=0.5, line_color="#2a2a2a", line_dash="dash")
    fig_scat.add_annotation(x=0.75, y=churn["monetary_total"].quantile(0.97),
                             text="🚨 AZONNALI AKCIÓ",
                             font=dict(size=9, color=RED, family="IBM Plex Mono"),
                             showarrow=False)
    layout_s = {**NOIR, "height": 280}
    layout_s["yaxis"] = {
        **NOIR["yaxis"],
        "type": "log",
        "title": "Összbevétel £ (log)",
        "tickfont": dict(size=9, color="#444")
    }
    layout_s["xaxis"] = dict(**NOIR["xaxis"], title="Churn valószínűség",
                              tickformat=".0%")
    layout_s["legend"] = dict(orientation="h", y=1.1, x=0,
                               font=dict(size=9, color="#444"),
                               bgcolor="rgba(0,0,0,0)")
    fig_scat.update_layout(**layout_s)
    st.plotly_chart(fig_scat, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  FULL-WIDTH: Snake Plot újragondolva — "Ujjlenyomat"
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ III. — A szegmensek ujjlenyomata</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

st.markdown('<div class="chart-full"><div class="chart-title">SZEGMENS-UJJLENYOMAT — RFM PROFIL SZEMTŐL SZEMBEN</div><div class="chart-subtitle">Standardizált eltérés az átlagtól · negatív recency = frissebb = jobb · radar formátum</div></div>', unsafe_allow_html=True)

seg_order_full = ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"]
seg_col_map    = {
    "VIP Bajnokok":          BONE,
    "Új / Ígéretes":        "#555",
    "Lemorzsolódó / Alvó":  RED,
    "Elvesztett / Inaktív": "#333",
}

dims = ["recency_days", "frequency", "monetary_total"]
dim_labels = ["Recency<br>(napok)", "Frequency<br>(vásárlás)", "Monetary<br>(£ össz)"]

rfm_mean = seg[dims].mean()
rfm_std  = seg[dims].std().replace(0, 1)

def hex_to_rgba(hex_color, alpha=0.08):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

fig_radar = go.Figure()
for seg_name in seg_order_full:
    sub = seg[seg["segment_label"] == seg_name][dims]
    if len(sub) == 0:
        continue
    avg = sub.mean()
    std_vals = ((avg - rfm_mean) / rfm_std).values
    std_vals[0] = -std_vals[0]
    closed = list(std_vals) + [std_vals[0]]
    closed_labels = dim_labels + [dim_labels[0]]

    base_color = seg_col_map.get(seg_name, "#333333")
    fill_c = hex_to_rgba(base_color, 0.08) if "#" in base_color else "rgba(30,30,30,0.1)"

    fig_radar.add_trace(go.Scatterpolar(
        r=closed,
        theta=closed_labels,
        fill="toself",
        fillcolor=fill_c,
        line=dict(color=base_color, width=2),
        name=seg_name,
        hovertemplate=f"<b>{seg_name}</b><br>%{{theta}}: %{{r:.2f}} σ<extra></extra>",
    ))

fig_radar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        gridshape="linear",
        angularaxis=dict(
            tickfont=dict(size=10, color="#666", family="IBM Plex Mono"),
            linecolor="#222",
            gridcolor="#1a1a1a",
        ),
        radialaxis=dict(
            tickfont=dict(size=8, color="#333"),
            gridcolor="#1a1a1a",
            linecolor="#222",
            tickformat=".1f",
        ),
    ),
    legend=dict(
        orientation="h", y=-0.15, x=0.5, xanchor="center",
        font=dict(size=10, color="#666", family="IBM Plex Mono"),
        bgcolor="rgba(0,0,0,0)",
    ),
    font=dict(family="IBM Plex Mono", color="#666"),
    height=420,
    margin=dict(t=20, b=60, l=80, r=80),
)
with st.container():
    st.markdown('<div style="padding: 0 60px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CHART ROW 3:  Recency hisztogram  +  Bubble szegmens
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ IV. — Az idő és az elveszett esélyek</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

col_x, col_y = st.columns([3, 2])

with col_x:
    st.markdown("""
    <div class="chart-cell" style="border-right:none;padding-left:60px">
        <div class="chart-title">RECENCY HISZTOGRAM — MIKOR VÁSÁROLT UTOLJÁRA?</div>
        <div class="chart-subtitle">Napok az utolsó vásárlás óta · szegmensenként rétegezve · több nap = több baj</div>
    </div>
    """, unsafe_allow_html=True)

    fig_hist = go.Figure()
    hist_colors = {
        "VIP Bajnokok":          BONE,
        "Új / Ígéretes":        "#444",
        "Lemorzsolódó / Alvó":  RED,
        "Elvesztett / Inaktív": "#2a2a2a",
    }
    for sname in seg_order_full:
        sub = seg[seg["segment_label"] == sname]["recency_days"]
        if len(sub) == 0:
            continue
        fig_hist.add_trace(go.Histogram(
            x=sub, nbinsx=50,
            name=sname,
            marker_color=hist_colors.get(sname, "#333"),
            opacity=0.85,
        ))
    layout_h = {**NOIR, "height": 280, "barmode": "stack"}
    layout_h["xaxis"] = dict(**NOIR["xaxis"], title="Recency (napok)")
    layout_h["yaxis"] = dict(**NOIR["yaxis"], title="Ügyfelek száma")
    layout_h["legend"] = dict(orientation="h", y=1.1, x=0,
                               font=dict(size=9, color="#444"),
                               bgcolor="rgba(0,0,0,0)")
    fig_hist.update_layout(**layout_h)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_y:
    st.markdown("""
    <div class="chart-cell" style="border-left:1px solid #1e1e1e;border-right:none;padding-right:60px">
        <div class="chart-title">SZEGMENS BUBORÉK</div>
        <div class="chart-subtitle">X: átl. recency · Y: átl. monetary · méret: ügyfélszám</div>
    </div>
    """, unsafe_allow_html=True)

    bubble = seg.groupby("segment_label").agg(
        avg_recency=("recency_days", "mean"),
        avg_monetary=("monetary_total", "mean"),
        count=("Customer ID", "count"),
    ).reset_index()

    fig_bub = go.Figure()
    for _, row in bubble.iterrows():
        sname = row["segment_label"]
        col_b2 = hist_colors.get(sname, "#333")
        fig_bub.add_trace(go.Scatter(
            x=[row["avg_recency"]],
            y=[row["avg_monetary"]],
            mode="markers+text",
            marker=dict(
                size=row["count"] / bubble["count"].max() * 80 + 15,
                color=col_b2,
                line=dict(color="#444" if col_b2 == BONE else col_b2, width=1),
                opacity=0.85,
            ),
            text=[sname.split(" ")[0]],
            textposition="middle center",
            textfont=dict(size=8, color=BONE if col_b2 != BONE else "#000",
                          family="IBM Plex Mono"),
            name=sname,
            hovertemplate=f"<b>{sname}</b><br>Recency: %{{x:.0f}} nap<br>Átl: £%{{y:,.0f}}<br>{row['count']:,} ügyfél<extra></extra>",
            showlegend=False,
        ))
    layout_b = {**NOIR, "height": 280}
    layout_b["xaxis"] = dict(**NOIR["xaxis"], title="Átl. recency (nap)")
    layout_b["yaxis"] = dict(**NOIR["yaxis"], title="Átl. monetary £", type="log")
    fig_bub.update_layout(**layout_b)
    st.plotly_chart(fig_bub, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  OBITUARY SECTION
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ V. — Gyászhirdetések és döntések</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

top_vip_risk = (
    churn[churn["churn_proba"] >= 0.5]
    .nlargest(5, "monetary_total")[["Customer ID", "monetary_total", "churn_proba", "recency_days"]]
    if "recency_days" in churn.columns
    else churn[churn["churn_proba"] >= 0.5].nlargest(5, "monetary_total")[["Customer ID", "monetary_total", "churn_proba"]]
)

avg_vip_monetary = churn[churn["monetary_total"] >= 2000]["monetary_total"].mean()

# ── A két oszlop Streamlit col-okkal renderelve ──────────────
obit_col_left, obit_col_right = st.columns([1, 2])

with obit_col_left:
    st.markdown(f"""
    <div style="padding: 32px 32px 32px 60px;">
        <div class="obit-header">VIP<br>VESZÉLYBEN</div>
        <div class="obit-text">
            A következő ügyfelek <strong>ma még jelen vannak</strong>
            a rendszerben — holnap már lehet, hogy nem.<br><br>
            Churn valószínűségük meghaladja az
            <strong>50%-ot</strong>, és bevételük alapján
            ők tartják fent a cég bevételének
            <strong>jelentős részét</strong>.<br><br>
            Minden elveszett VIP ügyfél átlagosan
            <strong>£{avg_vip_monetary:,.0f}</strong>
            éves bevétel-kiesést jelent.<br><br>
            Az idő most van.
        </div>
    </div>
    """, unsafe_allow_html=True)

with obit_col_right:
    st.markdown("""
    <div style="padding: 32px 60px 32px 0;">
        <div class="obit-col-header">Legértékesebb veszélyeztetett ügyfelek</div>
    """, unsafe_allow_html=True)

    for _, row in top_vip_risk.iterrows():
        cust_id    = int(row["Customer ID"])
        monetary   = row["monetary_total"]
        churn_prob = row["churn_proba"]
        rec_str    = f" &nbsp;·&nbsp; {int(row['recency_days'])} napja nem vásárolt" if "recency_days" in row.index else ""

        st.markdown(
            f'<div class="obit-row">'
            f'<span class="obit-row-id">#{cust_id}</span>'
            f' &nbsp;&nbsp;·&nbsp;&nbsp; '
            f'Összbevétel: <span class="obit-row-val">£{monetary:,.0f}</span>'
            f' &nbsp;&nbsp;·&nbsp;&nbsp; '
            f'Churn: <span class="obit-row-churn">{churn_prob:.0%}</span>'
            f'{rec_str}'
            f'<br>'
            f'<span class="obit-row-action">⬛ Javasolt akció: személyes megkeresés, exkluzív visszatérési ajánlat, dedikált account manager</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  FULL-WIDTH: Bevételi koncentráció heatmap
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">§ VI. — Ahol a pénz valóban van</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

st.markdown('<div class="chart-full"><div class="chart-title">BEVÉTELI KONCENTRÁCIÓ — SZEGMENS × RECENCY BUCKET</div><div class="chart-subtitle">Minél sötétebb, annál kevesebb pénz · vörös = ahol a bevétel koncentrálódik</div></div>', unsafe_allow_html=True)

seg2 = seg.copy()
seg2["recency_bucket"] = pd.cut(
    seg2["recency_days"],
    bins=[0, 30, 60, 120, 180, 365, 9999],
    labels=["0–30 nap\n(Friss)", "31–60", "61–120", "121–180", "181–365", "365+\n(Eltűnt)"],
)
heat_data = seg2.groupby(["segment_label", "recency_bucket"])["monetary_total"].sum().unstack(fill_value=0)
heat_data = heat_data.reindex([s for s in seg_order_full if s in heat_data.index])

fig_heat = go.Figure(go.Heatmap(
    z=heat_data.values,
    x=[str(c) for c in heat_data.columns],
    y=heat_data.index.tolist(),
    colorscale=[
        [0.0,  "#0a0a0a"],
        [0.2,  "#1a0a0a"],
        [0.5,  "#6b0f1a"],
        [0.8,  "#b5132a"],
        [1.0,  "#cc1a2e"],
    ],
    text=[[f"£{v/1000:.0f}K" for v in row] for row in heat_data.values],
    texttemplate="%{text}",
    textfont=dict(size=10, family="IBM Plex Mono", color=BONE),
    hovertemplate="<b>%{y}</b> · %{x}<br>£%{z:,.0f}<extra></extra>",
    showscale=False,
))
NOIR_heat = {k: v for k, v in NOIR.items() if k not in ("margin", "xaxis", "yaxis")}
fig_heat.update_layout(
    **NOIR_heat,
    height=220,
    margin=dict(t=8, b=8, l=160, r=40),
    xaxis=dict(gridcolor="#161616", linecolor="#222", zerolinecolor="#222",
               tickfont=dict(size=9, color="#444", family="IBM Plex Mono")),
    yaxis=dict(gridcolor="#161616", linecolor="#222", zerolinecolor="#222",
               tickfont=dict(size=9, color="#888", family="IBM Plex Mono")),
)
with st.container():
    st.markdown('<div style="padding: 0 60px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="brutal-footer">
    <div class="brutal-footer-text">
        Customer Autopsy · Data Intelligence Bureau · {today_str}
    </div>
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;color:#1e1e1e;letter-spacing:0.1em;">
        BRUTÁLISAN ŐSZINTE · ADATALAPÚ · MEGKERÜLHETETLEN
    </div>
    <div class="brutal-footer-text">
        XGBoost · K-Means · RFM · SHAP · Online Retail II · UCI
    </div>
</div>
""", unsafe_allow_html=True)