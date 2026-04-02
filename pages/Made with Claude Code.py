"""
Customer Intelligence Dashboard
================================
Streamlit egyldalas vizualizáció – RFM Szegmentáció & Churn Előrejelzés

Indítás:
    streamlit run dashboard.py

Adatok:
    A script automatikusan keresi a projekt által generált CSV/Parquet fájlokat.
    Ha nem találja, szintetikus demo adatokkal indul.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# BRAND COLORS & PALETTA
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "accent":   "#A81022",
    "accent_l": "#C4242E",
    "accent_d": "#7A0B19",
    "accent_m": "#E8424E",
    "bg":       "#0A0A0A",
    "surf":     "#111111",
    "surf2":    "#181818",
    "surf3":    "#222222",
    "border":   "#2D2D2D",
    "text":     "#FFFFFF",
    "muted":    "#888888",
    "muted2":   "#555555",
    "green":    "#4CAF82",
    "orange":   "#E8A838",
    "blue":     "#4A90D9",
}

# Szegmens szín-sorrend (Champions → Lost)
SEG_COLORS = {
    "Champions":    C["accent"],
    "Loyal":        "#C43040",
    "Promising":    "#5C4A00",
    "At Risk":      "#C47820",
    "Lost":         "#3A3A3A",
}

SEGMENT_ORDER = ["Champions", "Loyal", "Promising", "At Risk", "Lost"]

# Plotly sorozat-paletta (5 szín)
PALETTE = [C["accent"], "#C43040", "#9B6B10", "#C47820", "#4A4A4A"]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Alap ── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
.stApp {{
    background-color: {C["bg"]};
    color: {C["text"]};
}}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: #0E0E0E;
    border-right: 1px solid {C["border"]};
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {C["muted"]} !important;
    font-size: 12px;
}}

/* ── KPI kártya ── */
.kpi-card {{
    background: {C["surf"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 18px 22px 14px 22px;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: {C["accent"]};
    border-radius: 10px 10px 0 0;
}}
.kpi-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: {C["muted"]};
    margin-bottom: 10px;
}}
.kpi-value {{
    font-size: 30px;
    font-weight: 800;
    color: {C["text"]};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px;
    color: {C["muted"]};
    margin-top: 6px;
}}
.kpi-delta-pos {{ color: {C["green"]}; font-size: 12px; font-weight: 600; margin-top: 4px; }}
.kpi-delta-neg {{ color: {C["accent"]}; font-size: 12px; font-weight: 600; margin-top: 4px; }}

/* ── Szekció fejléc ── */
.sec-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 4px 0;
}}
.sec-bar {{
    width: 4px;
    height: 28px;
    background: {C["accent"]};
    border-radius: 2px;
    flex-shrink: 0;
}}
.sec-title {{
    font-size: 16px;
    font-weight: 700;
    color: {C["text"]};
    letter-spacing: 0.2px;
}}
.sec-desc {{
    font-size: 12px;
    color: {C["muted"]};
    margin-bottom: 16px;
    padding-left: 16px;
}}

/* ── Chart wrapper ── */
.chart-box {{
    background: {C["surf"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 10px;
}}

/* ── Insight kártya ── */
.insight-card {{
    background: {C["surf2"]};
    border: 1px solid {C["border"]};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}}
.insight-card .i-seg {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.insight-card .i-val {{
    font-size: 22px;
    font-weight: 800;
    color: {C["text"]};
}}
.insight-card .i-desc {{
    font-size: 11px;
    color: {C["muted"]};
    margin-top: 2px;
}}

/* ── Demo badge ── */
.demo-badge {{
    background: {C["surf3"]};
    border: 1px solid {C["accent_d"]};
    color: {C["accent_m"]};
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    display: inline-block;
}}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {{
    background-color: {C["accent_d"]} !important;
    border: none !important;
}}

/* ── Elrejtjük a Streamlit fejlécet ── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* ── Dataframe stílus ── */
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}
[data-testid="stDataFrameResizable"] {{ background: {C["surf"]} !important; }}

/* ── Elválasztó ── */
hr {{ border-color: {C["border"]} !important; margin: 8px 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADATBETÖLTÉS
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

RFM_PATHS = [
    BASE / "data" / "processed" / "rfm_segments.csv",
    BASE / "data" / "processed" / "customer_segments.csv",
    BASE / "data" / "rfm_segments.csv",
    BASE / "data" / "customer_segments.csv",
    BASE / "outputs" / "rfm_segments.csv",
    BASE / "outputs" / "customer_segments.csv",
    BASE / "data" / "processed" / "rfm_segments.parquet",
    BASE / "data" / "customer_segments.parquet",
]

CHURN_PATHS = [
    BASE / "data" / "processed" / "churn_predictions.csv",
    BASE / "data" / "churn_predictions.csv",
    BASE / "outputs" / "churn_predictions.csv",
    BASE / "data" / "processed" / "churn_predictions.parquet",
]

RFM_REQ   = {"customer_id", "recency", "frequency", "monetary"}
CHURN_REQ = {"customer_id", "churn_probability"}


def _load_file(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None


def _try_paths(paths: list, required: set) -> pd.DataFrame | None:
    for p in paths:
        if not p.exists():
            continue
        df = _load_file(p)
        if df is None:
            continue
        df.columns = df.columns.str.lower().str.strip()
        if required.issubset(set(df.columns)):
            return df
    return None


def _rfm_to_segments(df: pd.DataFrame) -> pd.Series:
    """Automatikus szegmens-hozzárendelés R/F/M értékekből."""
    r = pd.qcut(df["recency"],   5, labels=[5,4,3,2,1], duplicates="drop").astype(float)
    f = pd.qcut(df["frequency"], 5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
    m = pd.qcut(df["monetary"],  5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
    score = r.fillna(3) + f.fillna(3) + m.fillna(3)
    return pd.cut(score, bins=[0, 5, 8, 10, 12, 16],
                  labels=["Lost","At Risk","Promising","Loyal","Champions"],
                  include_lowest=True).astype(str)


def _demo_data(n: int = 3800, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Szintetikus, de realisztikus e-commerce adatok."""
    rng = np.random.default_rng(seed)
    segs = {
        "Champions": dict(w=0.11, r_mu=14,  r_s=7,   f_mu=19, f_s=5,  m_mu=870, m_s=310, ca=1, cb=9),
        "Loyal":     dict(w=0.22, r_mu=34,  r_s=14,  f_mu=10, f_s=3,  m_mu=430, m_s=155, ca=2, cb=7),
        "Promising": dict(w=0.24, r_mu=58,  r_s=20,  f_mu=5,  f_s=2,  m_mu=225, m_s=100, ca=3, cb=5),
        "At Risk":   dict(w=0.27, r_mu=98,  r_s=27,  f_mu=3,  f_s=2,  m_mu=175, m_s=85,  ca=6, cb=3),
        "Lost":      dict(w=0.16, r_mu=185, r_s=42,  f_mu=1,  f_s=1,  m_mu=65,  m_s=38,  ca=9, cb=2),
    }
    rows = []
    cid = 10000
    for seg, p in segs.items():
        cnt = int(n * p["w"])
        for _ in range(cnt):
            rows.append({
                "customer_id":       cid,
                "recency":           max(1,  int(rng.normal(p["r_mu"], p["r_s"]))),
                "frequency":         max(1,  int(rng.normal(p["f_mu"], p["f_s"]))),
                "monetary":          max(5., round(float(rng.normal(p["m_mu"], p["m_s"])), 2)),
                "segment":           seg,
                "churn_probability": round(float(np.clip(rng.beta(p["ca"], p["cb"]), 0, 1)), 4),
            })
            cid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.drop(columns=["churn_probability"]), df[["customer_id","churn_probability"]]


@st.cache_data(show_spinner="⏳  Adatok betöltése...")
def load_data():
    rfm   = _try_paths(RFM_PATHS,   RFM_REQ)
    churn = _try_paths(CHURN_PATHS, CHURN_REQ)
    demo  = rfm is None

    if demo:
        rfm, churn = _demo_data()
    else:
        # Szegmens oszlop normalizálása
        for col in ["segment","cluster","cluster_label","segment_label","customer_segment"]:
            if col in rfm.columns and col != "segment":
                rfm = rfm.rename(columns={col: "segment"})
                break
        if "segment" not in rfm.columns:
            rfm["segment"] = _rfm_to_segments(rfm)

        if churn is None:
            churn = rfm[["customer_id"]].copy()
            # becsléses fallback
            r_norm = (rfm["recency"] - rfm["recency"].min()) / rfm["recency"].ptp()
            churn["churn_probability"] = (r_norm * 0.6 + np.random.beta(2,5,len(rfm))*0.4).clip(0,1).round(4)

        # Churn oszlop normalizálása
        for col in ["churn_prob","churn_score","churn_probability","probability"]:
            if col in churn.columns and col != "churn_probability":
                churn = churn.rename(columns={col: "churn_probability"})
                break

    merged = rfm.merge(churn[["customer_id","churn_probability"]], on="customer_id", how="left")
    merged["churn_probability"] = merged["churn_probability"].fillna(0.3)
    merged["segment"] = merged["segment"].astype(str).str.strip()
    return merged, demo


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY TÉMA SEGÉDFÜGGVÉNY
# ═══════════════════════════════════════════════════════════════════════════════

def theme(fig, title: str = "", height: int = None) -> go.Figure:
    kw = dict(
        paper_bgcolor=C["surf"],
        plot_bgcolor=C["surf"],
        font=dict(family="Inter, Segoe UI, sans-serif", color=C["text"], size=12),
        margin=dict(t=44 if title else 20, b=36, l=52, r=24),
        colorway=PALETTE,
        legend=dict(
            bgcolor=C["surf2"],
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(size=11, color=C["muted"]),
        ),
        hoverlabel=dict(
            bgcolor=C["surf3"],
            bordercolor=C["border"],
            font=dict(color=C["text"], size=12),
        ),
    )
    if title:
        kw["title"] = dict(text=title, font=dict(size=13, color=C["text"], weight=700), x=0.02, y=0.97)
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False,
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["muted"]))
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False,
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["muted"]))
    return fig


def seg_color(name: str) -> str:
    return SEG_COLORS.get(name, C["muted2"])


# ═══════════════════════════════════════════════════════════════════════════════
# KOMPONENSEK
# ═══════════════════════════════════════════════════════════════════════════════

def kpi(label: str, value: str, sub: str = "", delta: float = None, delta_good: bool = True):
    if delta is not None:
        dclass = "kpi-delta-pos" if delta_good else "kpi-delta-neg"
        arrow  = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="{dclass}">{arrow}&nbsp;{abs(delta):.1f}%</div>'
    else:
        delta_html = ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def section(title: str, desc: str = ""):
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-bar"></div>
      <div class="sec-title">{title}</div>
    </div>
    <div class="sec-desc">{desc}</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIZUALIZÁCIÓK
# ═══════════════════════════════════════════════════════════════════════════════

def fig_segment_donut(df: pd.DataFrame) -> go.Figure:
    """Szegmens megoszlás – ügyfél darabszám."""
    counts = (df.groupby("segment")
                .size()
                .reindex([s for s in SEGMENT_ORDER if s in df["segment"].unique()])
                .dropna())
    colors = [seg_color(s) for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{label}</b><br>%{value:,} ügyfél<br>%{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
    ))
    fig.add_annotation(text="Ügyfelek", x=0.5, y=0.54, showarrow=False,
                       font=dict(size=11, color=C["muted"]))
    fig.add_annotation(text=f"{len(df):,}", x=0.5, y=0.42, showarrow=False,
                       font=dict(size=26, color=C["text"], weight=700) if hasattr(dict,"weight") else
                       dict(size=26, color=C["text"]))
    theme(fig, "Szegmens Megoszlás", height=320)
    fig.update_layout(legend=dict(orientation="v", x=1.05, y=0.5))
    return fig


def fig_revenue_donut(df: pd.DataFrame) -> go.Figure:
    """Szegmens megoszlás – bevétel arány."""
    rev = (df.groupby("segment")["monetary"]
             .sum()
             .reindex([s for s in SEGMENT_ORDER if s in df["segment"].unique()])
             .dropna())
    colors = [seg_color(s) for s in rev.index]
    fig = go.Figure(go.Pie(
        labels=rev.index,
        values=rev.values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
    ))
    fig.add_annotation(text="Bevétel", x=0.5, y=0.54, showarrow=False,
                       font=dict(size=11, color=C["muted"]))
    fig.add_annotation(text=f"${rev.sum()/1e6:.1f}M", x=0.5, y=0.42, showarrow=False,
                       font=dict(size=22, color=C["text"]))
    theme(fig, "Bevételi Részesedés", height=320)
    fig.update_layout(legend=dict(orientation="v", x=1.05, y=0.5))
    return fig


def fig_rfm_scatter(df: pd.DataFrame) -> go.Figure:
    """RFM 2D scatter – Recency vs Frequency, méret=Monetary."""
    sample = df.sample(min(1800, len(df)), random_state=1)
    segs_present = [s for s in SEGMENT_ORDER if s in sample["segment"].unique()]

    fig = go.Figure()
    for seg in segs_present:
        sub = sample[sample["segment"] == seg]
        fig.add_trace(go.Scatter(
            x=sub["recency"],
            y=sub["frequency"],
            mode="markers",
            name=seg,
            marker=dict(
                color=seg_color(seg),
                size=np.clip(sub["monetary"] / sub["monetary"].max() * 24, 4, 24),
                opacity=0.75,
                line=dict(color=C["bg"], width=0.5),
            ),
            hovertemplate=(
                f"<b>{seg}</b><br>"
                "Recency: %{x} nap<br>"
                "Frequency: %{y} vásárlás<br>"
                "Monetary: $%{customdata:,.0f}<extra></extra>"
            ),
            customdata=sub["monetary"],
        ))

    # Kvadráns vezérvonalak
    r_med = df["recency"].median()
    f_med = df["frequency"].median()
    for val, axis in [(r_med, "x"), (f_med, "y")]:
        fig.add_shape(type="line",
                      x0=val if axis=="x" else df["recency"].min(),
                      x1=val if axis=="x" else df["recency"].max(),
                      y0=val if axis=="y" else df["frequency"].min(),
                      y1=val if axis=="y" else df["frequency"].max(),
                      line=dict(color=C["muted2"], width=1, dash="dot"))

    theme(fig, "RFM Pozícionálás  ·  pont mérete = Vásárlási Érték", height=400)
    fig.update_xaxes(title="Recency (napok az utolsó vásárlás óta)",
                     autorange="reversed")  # alacsonyabb recency = jobb
    fig.update_yaxes(title="Vásárlások száma (Frequency)")
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Hex színt rgba() formátumba konvertál Plotly kompatibilis módon."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_rfm_box(df: pd.DataFrame, metric: str, label: str, unit: str = "") -> go.Figure:
    """Box-plot egy RFM metrikához szegmensenként."""
    segs = [s for s in SEGMENT_ORDER if s in df["segment"].unique()]
    fig = go.Figure()
    for seg in segs:
        vals = df[df["segment"] == seg][metric]
        color = seg_color(seg)
        fig.add_trace(go.Box(
            y=vals,
            name=seg,
            marker_color=color,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.25),
            boxmean=True,
            hovertemplate=f"<b>{seg}</b><br>{label}: %{{y:.1f}}{unit}<extra></extra>",
        ))
    theme(fig, label, height=280)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=label + (f" ({unit})" if unit else ""))
    return fig


def fig_churn_histogram(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Churn valószínűség eloszlás hisztogram."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["churn_probability"],
        nbinsx=40,
        marker_color=C["surf3"],
        marker_line=dict(color=C["border"], width=0.5),
        name="Minden ügyfél",
        hovertemplate="Churn p: %{x:.2f}<br>Ügyfelek: %{y}<extra></extra>",
    ))
    # Küszöb fölötti sáv piros
    high_risk = df[df["churn_probability"] >= threshold]
    fig.add_trace(go.Histogram(
        x=high_risk["churn_probability"],
        nbinsx=40,
        marker_color=_hex_to_rgba(C["accent"], 0.75),
        marker_line=dict(color=C["accent"], width=0.5),
        name=f"Magas kockázat (≥{threshold:.2f})",
        hovertemplate="Churn p: %{x:.2f}<br>Ügyfelek: %{y}<extra></extra>",
    ))
    # Küszöb vonal
    fig.add_vline(x=threshold, line_dash="dash", line_color=C["accent"],
                  annotation=dict(text=f"Küszöb: {threshold:.2f}",
                                  font=dict(color=C["accent"], size=11),
                                  bgcolor=C["surf3"],
                                  bordercolor=C["accent"],
                                  borderwidth=1,
                                  borderpad=4))
    theme(fig, "Churn Valószínűség Eloszlása", height=320)
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title="Churn Valószínűség")
    fig.update_yaxes(title="Ügyfelek száma")
    return fig


def fig_churn_by_segment(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Churn ráta szegmensenként – vízszintes oszlopdiagram."""
    agg = (df.groupby("segment")
             .agg(churn_rate=("churn_probability", lambda x: (x >= threshold).mean() * 100),
                  avg_prob=("churn_probability", "mean"),
                  count=("customer_id", "count"))
             .reset_index())
    agg = agg.set_index("segment").reindex(
        [s for s in SEGMENT_ORDER if s in agg["segment"].values]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=agg["segment"],
        x=agg["churn_rate"],
        orientation="h",
        marker=dict(
            color=[seg_color(s) for s in agg["segment"]],
            line=dict(color=C["bg"], width=1),
        ),
        text=[f"{v:.1f}%" for v in agg["churn_rate"]],
        textposition="outside",
        textfont=dict(color=C["text"], size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Churn ráta: %{x:.1f}%<br>"
            "Átlag valószínűség: %{customdata:.3f}<extra></extra>"
        ),
        customdata=agg["avg_prob"],
    ))
    theme(fig, f"Churn Ráta Szegmensenként  (küszöb: {threshold:.2f})", height=320)
    fig.update_xaxes(title="Churn Ráta (%)", range=[0, agg["churn_rate"].max() * 1.25])
    fig.update_yaxes(title="")
    return fig


def fig_revenue_at_risk(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Stacked bar – biztonságos vs veszélyeztetett bevétel szegmensenként."""
    def risk_band(p):
        if p >= threshold:
            return "Magas kockázat"
        elif p >= threshold * 0.6:
            return "Mérsékelt kockázat"
        else:
            return "Biztonságos"

    df2 = df.copy()
    df2["risk"] = df2["churn_probability"].apply(risk_band)
    agg = df2.groupby(["segment","risk"])["monetary"].sum().unstack(fill_value=0)
    agg = agg.reindex([s for s in SEGMENT_ORDER if s in agg.index])

    bands = ["Biztonságos", "Mérsékelt kockázat", "Magas kockázat"]
    colors_map = {
        "Biztonságos":        _hex_to_rgba(C["green"],  0.75),
        "Mérsékelt kockázat": _hex_to_rgba(C["orange"], 0.75),
        "Magas kockázat":     _hex_to_rgba(C["accent"], 0.85),
    }
    fig = go.Figure()
    for band in bands:
        if band not in agg.columns:
            continue
        fig.add_trace(go.Bar(
            name=band,
            x=agg.index,
            y=agg[band] / 1e3,
            marker_color=colors_map[band],
            marker_line=dict(color=C["bg"], width=1),
            hovertemplate=f"<b>%{{x}}</b><br>{band}: $%{{y:,.1f}}K<extra></extra>",
        ))
    theme(fig, "Bevételi Kockázat Szegmensenként  (ezer $)", height=340)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Bevétel (ezer $)")
    return fig


def _safe_qcut(series: pd.Series, n: int, prefix: str) -> pd.Series:
    """qcut duplicates-safe változata: mindig pontosan n bint ad vissza."""
    try:
        result = pd.qcut(series, n, labels=[f"{prefix}{i}" for i in range(1, n+1)],
                         duplicates="drop")
        if result.nunique() == n:
            return result
    except ValueError:
        pass
    # Fallback: rank-alapú binning
    ranks = series.rank(method="first")
    return pd.cut(ranks, bins=n, labels=[f"{prefix}{i}" for i in range(1, n+1)])


def fig_churn_rfm_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: átlagos churn valószínűség R-score × F-score rácsban."""
    d = df.copy()
    d["r_bin"] = _safe_qcut(d["recency"],   5, "R")
    d["f_bin"] = _safe_qcut(d["frequency"], 5, "F")
    d = d.dropna(subset=["r_bin","f_bin"])
    pivot = d.pivot_table(values="churn_probability", index="r_bin", columns="f_bin", aggfunc="mean")

    colorscale = [
        [0.0,  C["surf3"]],
        [0.35, C["accent_d"]],
        [0.65, C["accent"]],
        [1.0,  C["accent_m"]],
    ]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale=colorscale,
        zmin=0, zmax=1,
        text=np.round(pivot.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="R: %{y}  F: %{x}<br>Átlag churn: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Churn",
            tickcolor=C["muted"],
            tickfont=dict(color=C["muted"]),
            outlinecolor=C["border"],
            outlinewidth=1,
        ),
    ))
    theme(fig, "Churn Hőtérkép  ·  Recency × Frequency", height=340)
    fig.update_xaxes(title="Frequency kvintilis (F1=legkevesebb)")
    fig.update_yaxes(title="Recency kvintilis (R1=legrégebben vásárolt)")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FŐ APP
# ═══════════════════════════════════════════════════════════════════════════════

df, using_demo = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🎯 Customer Intelligence")
    st.markdown("---")

    all_segs = [s for s in SEGMENT_ORDER if s in df["segment"].unique()]
    sel_segs = st.multiselect(
        "SZEGMENS SZŰRŐ",
        options=all_segs,
        default=all_segs,
        help="Válaszd ki a megjelenítendő szegmenseket",
    )

    threshold = st.slider(
        "CHURN KÜSZÖB",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="E fölötti churn valószínűség esetén az ügyfél 'magas kockázatú'",
        format="%.2f",
    )

    top_n = st.slider(
        "TOP KOCKÁZATOS ÜGYFELEK (TÁBLÁZAT)",
        min_value=10, max_value=100, value=20, step=5,
    )

    st.markdown("---")
    if using_demo:
        st.markdown('<span class="demo-badge">⚡ DEMO ADATOK</span>', unsafe_allow_html=True)
        st.caption("Szintetikus adatok – csatlakoztatsd a projekt CSV/Parquet outputjait a valós megjelenítéshez.")
    else:
        st.markdown('<span class="demo-badge" style="border-color:#4CAF82;color:#4CAF82;">● ÉLŐ ADATOK</span>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.caption("**Elvárt fájlok** az élő adatokhoz:")
    st.caption("`data/processed/rfm_segments.csv`")
    st.caption("`data/processed/churn_predictions.csv`")

# Szűrés a kiválasztott szegmensekre
dff = df[df["segment"].isin(sel_segs)] if sel_segs else df.copy()

# ── FEJLÉC ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
  <div style="font-size:28px;font-weight:800;letter-spacing:-0.5px;color:#FFFFFF;line-height:1.1;">
    Customer Intelligence Dashboard
  </div>
  <div style="font-size:13px;color:#888;margin-top:6px;">
    RFM Szegmentáció &nbsp;·&nbsp; Churn Előrejelzés &nbsp;·&nbsp; Bevételi Kockázat
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI KÁRTYÁK ───────────────────────────────────────────────────────────────
n_total    = len(dff)
n_churn    = int((dff["churn_probability"] >= threshold).sum())
churn_pct  = n_churn / n_total * 100 if n_total else 0
avg_clv    = dff["monetary"].mean()
total_rev  = dff["monetary"].sum()
rev_at_risk = dff[dff["churn_probability"] >= threshold]["monetary"].sum()
top_seg    = dff.groupby("segment").size().idxmax() if n_total else "–"

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Összes ügyfél", f"{n_total:,}",
        sub=f"{len(sel_segs)} szegmens aktív")
with k2:
    kpi("Churn ráta", f"{churn_pct:.1f}%",
        sub=f"{n_churn:,} magas kockázatú ügyfél",
        delta=None)
with k3:
    kpi("Átlag vásárlási érték", f"${avg_clv:,.0f}",
        sub=f"Összes: ${total_rev/1e6:.2f}M")
with k4:
    kpi("Veszélyeztetett bevétel", f"${rev_at_risk/1e3:,.0f}K",
        sub=f"a teljes bevétel {rev_at_risk/total_rev*100:.1f}%-a" if total_rev else "–",
        delta=None)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SZEKCIÓ – RFM SZEGMENTÁCIÓ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
section(
    "① RFM Szegmentáció",
    "Ügyfelek csoportosítása Recency · Frequency · Monetary mutatók alapján"
)

col_l, col_r = st.columns(2, gap="medium")
with col_l:
    with st.container():
        st.plotly_chart(fig_segment_donut(dff), use_container_width=True, config={"displayModeBar": False})
with col_r:
    with st.container():
        st.plotly_chart(fig_revenue_donut(dff), use_container_width=True, config={"displayModeBar": False})

st.plotly_chart(fig_rfm_scatter(dff), use_container_width=True, config={"displayModeBar": False})

# RFM Box-plot sor
b1, b2, b3 = st.columns(3, gap="small")
with b1:
    st.plotly_chart(fig_rfm_box(dff, "recency",   "Recency",   "nap"),
                    use_container_width=True, config={"displayModeBar": False})
with b2:
    st.plotly_chart(fig_rfm_box(dff, "frequency", "Frequency", "db"),
                    use_container_width=True, config={"displayModeBar": False})
with b3:
    st.plotly_chart(fig_rfm_box(dff, "monetary",  "Monetary",  "$"),
                    use_container_width=True, config={"displayModeBar": False})

# Szegmens összefoglaló táblázat
with st.expander("📋  Szegmens összefoglaló táblázat", expanded=False):
    agg_tbl = (dff.groupby("segment")
                  .agg(
                      Ügyfelek      = ("customer_id",      "count"),
                      Átlag_Recency = ("recency",           "mean"),
                      Átlag_Freq    = ("frequency",         "mean"),
                      Átlag_CLV     = ("monetary",          "mean"),
                      Össz_Bevétel  = ("monetary",          "sum"),
                      Churn_Ráta_pct= ("churn_probability", lambda x: (x >= threshold).mean() * 100),
                  )
                  .reindex([s for s in SEGMENT_ORDER if s in dff["segment"].unique()])
                  .round(1))
    agg_tbl["Össz_Bevétel"] = agg_tbl["Össz_Bevétel"].map("${:,.0f}".format)
    agg_tbl["Átlag_CLV"]    = agg_tbl["Átlag_CLV"].map("${:,.0f}".format)
    st.dataframe(agg_tbl, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SZEKCIÓ – CHURN ELŐREJELZÉS
# ─────────────────────────────────────────────────────────────────────────────
section(
    "② Churn Előrejelzés",
    f"Lemorzsolódási kockázat elemzése  ·  Aktív küszöb: {threshold:.2f}"
)

ch_l, ch_r = st.columns(2, gap="medium")
with ch_l:
    st.plotly_chart(fig_churn_histogram(dff, threshold),
                    use_container_width=True, config={"displayModeBar": False})
with ch_r:
    st.plotly_chart(fig_churn_by_segment(dff, threshold),
                    use_container_width=True, config={"displayModeBar": False})

st.plotly_chart(fig_revenue_at_risk(dff, threshold),
                use_container_width=True, config={"displayModeBar": False})

st.plotly_chart(fig_churn_rfm_heatmap(dff),
                use_container_width=True, config={"displayModeBar": False})

# Top at-risk ügyfelek táblázat
st.markdown(f"""
<div class="sec-header" style="margin-top:20px;">
  <div class="sec-bar"></div>
  <div class="sec-title">Top {top_n} Legveszélyeztetettebb Ügyfél</div>
</div>
<div class="sec-desc">Legmagasabb churn valószínűséggel rendelkező ügyfelek</div>
""", unsafe_allow_html=True)

risk_tbl = (dff.nlargest(top_n, "churn_probability")[
    ["customer_id","segment","recency","frequency","monetary","churn_probability"]
].copy())
risk_tbl.columns = ["Ügyfél ID","Szegmens","Recency (nap)","Vásárlások","CLV ($)","Churn P"]
risk_tbl["CLV ($)"]  = risk_tbl["CLV ($)"].map("${:,.0f}".format)
risk_tbl["Churn P"]  = risk_tbl["Churn P"].map("{:.3f}".format)
st.dataframe(
    risk_tbl.reset_index(drop=True),
    use_container_width=True,
    height=min(top_n * 36 + 40, 520),
)

# ─────────────────────────────────────────────────────────────────────────────
# LÁBLÉC
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top:1px solid {C['border']};padding-top:16px;display:flex;
            justify-content:space-between;align-items:center;">
  <span style="color:{C['muted']};font-size:11px;">
    Customer Intelligence Dashboard &nbsp;·&nbsp; E-commerce Ügyfél-szegmentáció
  </span>
  <span style="color:{C['muted2']};font-size:11px;">
    {"Demo adatok · " if using_demo else ""}
    {n_total:,} ügyfél &nbsp;·&nbsp; {len(sel_segs)} szegmens
  </span>
</div>
""", unsafe_allow_html=True)