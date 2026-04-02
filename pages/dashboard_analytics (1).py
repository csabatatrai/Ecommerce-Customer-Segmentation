# ============================================================
# dashboard_analytics.py – Üzleti Analitikai Dashboard
# ============================================================
# Futtatás: streamlit run dashboard_analytics.py
#
# A dashboard a projekt pipeline kimeneteit tölti be
# (customer_segments.parquet, churn_predictions.parquet, rfm_features.parquet).
# Ha a fájlok nem elérhetők, szimulált adatokkal fut bemutató módban.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Oldalbeállítások ────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Customer Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Téma és stílus ──────────────────────────────────────────
BRAND_RED    = "#c0253f"
BRAND_DARK   = "#1a1a2e"
BRAND_CARD   = "#16213e"
ACCENT_GOLD  = "#f0a500"
ACCENT_TEAL  = "#00b4d8"
ACCENT_GREEN = "#2dc653"
TEXT_MUTED   = "#8899aa"

SEGMENT_COLORS = {
    # Valós pipeline szegmensnevek (02_customer_segmentation.ipynb segment_color_map)
    "VIP Bajnokok":          BRAND_RED,
    "Lemorzsolódó / Alvó":  ACCENT_GOLD,
    "Elvesztett / Inaktív": "#4a4e69",
    "Új / Ígéretes":        ACCENT_TEAL,
    # Szimulált / alternatív nevek (fallback)
    "Hűséges Vásárlók":     ACCENT_TEAL,
    "Veszélyben Lévők":     ACCENT_GOLD,
    "Inaktív / Elveszett":  "#4a4e69",
    "Ismeretlen":           "#555566",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color="#cdd3de"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[BRAND_RED, ACCENT_TEAL, ACCENT_GOLD, ACCENT_GREEN, "#9b5de5", "#f15bb5"],
        xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
        yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=40, l=40, r=20),
    )
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {BRAND_DARK};
        color: #cdd3de;
    }}
    .stApp {{ background-color: {BRAND_DARK}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BRAND_CARD};
        border-right: 1px solid #1e2d3d;
    }}

    /* KPI kártya */
    .kpi-card {{
        background: linear-gradient(135deg, {BRAND_CARD} 0%, #0f3460 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }}
    .kpi-value {{
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }}
    .kpi-delta-pos {{ color: {ACCENT_GREEN}; font-size: 0.85rem; font-weight: 600; }}
    .kpi-delta-neg {{ color: {BRAND_RED};   font-size: 0.85rem; font-weight: 600; }}

    /* Section fejléc */
    .section-header {{
        font-family: 'Syne', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 3px solid {BRAND_RED};
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }}

    /* Szegmens badge */
    .badge-vip     {{ background:{BRAND_RED};    color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
    .badge-loyal   {{ background:{ACCENT_TEAL};  color:#000; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
    .badge-risk    {{ background:{ACCENT_GOLD};  color:#000; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}
    .badge-inactive{{ background:#4a4e69;        color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }}

    /* Elválasztó */
    hr {{ border-color: #1e2d3d; margin: 24px 0; }}

    div[data-testid="stMetric"] {{
        background: {BRAND_CARD};
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #1e3a5f;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
    [data-testid="stMetricValue"] {{ color: #fff !important; font-family: 'Syne', sans-serif; font-size: 1.8rem !important; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.82rem !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ADAT BETÖLTÉS / SZIMULÁCIÓ
# ═══════════════════════════════════════════════════════════

def _normalize_segment_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    A pipeline különböző notebookokban eltérő névvel menti a szegmens oszlopot:
      - 02_customer_segmentation: 'Segment'
      - 03_churn_prediction join után: 'rfm_segment'
      - esetleg: 'segment_label', 'Szegmens'
    Ez a függvény egységesíti 'segment_label' névvé.
    """
    candidates = ["Segment", "rfm_segment", "segment_label", "Szegmens"]
    for col in candidates:
        if col in df.columns and col != "segment_label":
            df = df.rename(columns={col: "segment_label"})
            break
    # Ha egyik sem létezik, generálunk egy placeholder oszlopot
    if "segment_label" not in df.columns:
        df["segment_label"] = "Ismeretlen"
    return df


def _normalize_churn_df(df: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    A churn parquet-ből hiányozhat a szegmenscímke (rfm_segment join opcionális volt).
    Ha nincs, joinoljuk a customer_segments-ből.
    """
    df = _normalize_segment_col(df)

    # Ha az összes érték 'Ismeretlen', próbáljuk pótolni a seg_df-ből
    if (df["segment_label"] == "Ismeretlen").all() and seg_df is not None:
        seg_norm = _normalize_segment_col(seg_df.copy())[["segment_label"]]
        df = df.join(seg_norm, how="left", rsuffix="_from_seg")
        if "segment_label_from_seg" in df.columns:
            df["segment_label"] = df["segment_label_from_seg"].fillna("Ismeretlen")
            df.drop(columns=["segment_label_from_seg"], inplace=True)

    # monetary_total: ha a churn df-ben nincs, próbáljuk meg joinolni
    if "monetary_total" not in df.columns and seg_df is not None:
        if "monetary_total" in seg_df.columns:
            df = df.join(seg_df[["monetary_total"]], how="left", rsuffix="_seg")

    # recency_days: ha nincs, próbáljuk meg joinolni
    if "recency_days" not in df.columns and seg_df is not None:
        if "recency_days" in seg_df.columns:
            df = df.join(seg_df[["recency_days"]], how="left", rsuffix="_seg")

    # return_ratio: opcionális
    if "return_ratio" not in df.columns:
        df["return_ratio"] = 0.0

    return df


@st.cache_data(show_spinner=False)
def load_data():
    """
    Pipeline kimeneteket tölt be, rugalmasan kezelve a valós oszlopneveket.
    Ha a fájlok nem elérhetők, szimulált adatokkal fut.
    """
    try:
        from config import CUSTOMER_SEGMENTS_PARQUET, CHURN_PREDICTIONS_PARQUET, RFM_FEATURES_PARQUET

        seg   = pd.read_parquet(CUSTOMER_SEGMENTS_PARQUET)
        churn = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
        rfm   = pd.read_parquet(RFM_FEATURES_PARQUET)

        seg   = _normalize_segment_col(seg)
        churn = _normalize_churn_df(churn, seg)
        rfm   = _normalize_segment_col(rfm)

        # Biztosítjuk, hogy a seg-ben is legyenek az RFM alaposzlopok
        for col in ["recency_days", "frequency", "monetary_total", "return_ratio"]:
            if col not in seg.columns and col in rfm.columns:
                seg = seg.join(rfm[[col]], how="left", rsuffix="_rfm")

        if "return_ratio" not in seg.columns:
            seg["return_ratio"] = 0.0

        # Customer ID: ha index-ben van, visszahozzuk oszlopba
        if "Customer ID" not in seg.columns:
            seg = seg.reset_index().rename(columns={"index": "Customer ID"})
        if "Customer ID" not in churn.columns:
            churn = churn.reset_index().rename(columns={"index": "Customer ID"})

        is_demo = False

    except Exception:
        is_demo = True
        seg, churn, rfm = _simulate_data()

    return seg, churn, rfm, is_demo


def _simulate_data():
    """Szimulált adatok, ha a pipeline kimenetei nem elérhetők."""
    rng = np.random.default_rng(42)
    n = 3_800

    # Valós szegmensnevek a 02-es notebookból
    segment_labels = np.random.choice(
        ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"],
        size=n,
        p=[0.12, 0.28, 0.25, 0.35],
    )

    recency   = np.where(segment_labels == "VIP Bajnokok",           rng.integers(1, 30, n),
                np.where(segment_labels == "Új / Ígéretes",          rng.integers(10, 80, n),
                np.where(segment_labels == "Lemorzsolódó / Alvó",    rng.integers(60, 200, n),
                                                                      rng.integers(150, 365, n))))
    frequency = np.where(segment_labels == "VIP Bajnokok",           rng.integers(15, 60, n),
                np.where(segment_labels == "Új / Ígéretes",          rng.integers(6, 20, n),
                np.where(segment_labels == "Lemorzsolódó / Alvó",    rng.integers(2, 8, n),
                                                                      rng.integers(1, 4, n))))
    monetary  = np.where(segment_labels == "VIP Bajnokok",           rng.uniform(3000, 25000, n),
                np.where(segment_labels == "Új / Ígéretes",          rng.uniform(800, 5000, n),
                np.where(segment_labels == "Lemorzsolódó / Alvó",    rng.uniform(200, 1500, n),
                                                                      rng.uniform(50, 500, n))))

    return_ratio = np.clip(rng.beta(2, 12, n) + np.where(segment_labels == "VIP Bajnokok", 0.04, 0), 0, 0.5)

    churn_proba = np.where(segment_labels == "VIP Bajnokok",          rng.uniform(0.02, 0.22, n),
                  np.where(segment_labels == "Új / Ígéretes",         rng.uniform(0.10, 0.40, n),
                  np.where(segment_labels == "Lemorzsolódó / Alvó",   rng.uniform(0.45, 0.85, n),
                                                                       rng.uniform(0.65, 0.99, n))))

    cust_ids = np.arange(10000, 10000 + n)

    seg = pd.DataFrame({
        "Customer ID":   cust_ids,
        "segment_label": segment_labels,
        "recency_days":  recency,
        "frequency":     frequency,
        "monetary_total":monetary,
        "return_ratio":  return_ratio,
    })

    churn = pd.DataFrame({
        "Customer ID":   cust_ids,
        "segment_label": segment_labels,
        "monetary_total":monetary,
        "recency_days":  recency,
        "return_ratio":  return_ratio,
        "churn_proba":   churn_proba,
        "churn_pred":    (churn_proba >= 0.5).astype(int),
    })
    churn["action"] = np.where(
        (churn["churn_proba"] >= 0.5) & (churn["monetary_total"] >= 2000),
        "🚨 VIP Veszélyben – Azonnali Retenció",
        np.where(churn["churn_proba"] >= 0.5, "⚠️ Retenció ajánlott",
        np.where(churn["churn_proba"] < 0.20, "✅ Stabil – Upsell lehetőség",
                 "👀 Figyelendő – Soft nurturing"))
    )

    rfm = seg.copy()
    return seg, churn, rfm


# Betöltés
with st.spinner("Adatok betöltése..."):
    df_seg, df_churn, df_rfm, IS_DEMO = load_data()

# Idősor kinyerése
if IS_DEMO:
    rng2 = np.random.default_rng(42)
    dates = pd.date_range("2009-12-01", "2011-11-30", freq="D")
    trend = np.linspace(0.8, 1.2, len(dates))
    seasonal = 1 + 0.35 * np.sin(2 * np.pi * np.arange(len(dates)) / 365 - np.pi / 2)
    noise = rng2.uniform(0.85, 1.15, len(dates))
    revenue_arr = 8000 * trend * seasonal * noise
    revenue_arr[dates.month == 11] *= 1.6
    revenue_arr[dates.month == 12] *= 2.1
    ts_df = pd.DataFrame({"InvoiceDate": dates, "revenue": revenue_arr})
else:
    # Valós adatokból idősor rekonstrukciója a nyers Parquet-ből
    try:
        from config import READY_FOR_RFM_PARQUET
        _raw = pd.read_parquet(READY_FOR_RFM_PARQUET, columns=["InvoiceDate", "Price", "Quantity"])
        _raw["revenue"] = _raw["Price"] * _raw["Quantity"]
        _raw["InvoiceDate"] = pd.to_datetime(_raw["InvoiceDate"])
        ts_df = _raw.groupby(_raw["InvoiceDate"].dt.date)["revenue"].sum().reset_index()
        ts_df.columns = ["InvoiceDate", "revenue"]
        ts_df["InvoiceDate"] = pd.to_datetime(ts_df["InvoiceDate"])
    except Exception:
        ts_df = None


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:24px'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#fff;'>
            📦 Customer<br>Intelligence
        </div>
        <div style='color:{TEXT_MUTED};font-size:0.8rem;margin-top:4px;'>E-commerce · B2B + B2C</div>
    </div>
    """, unsafe_allow_html=True)

    if IS_DEMO:
        st.info("🔬 **Bemutató mód** – Szimulált adatokkal fut. A valós adatokhoz futtasd előbb a pipeline notebookokat.", icon="ℹ️")

    st.markdown("---")
    st.markdown("**Nézet**")
    page = st.radio(
        "",
        ["📊 Összefoglaló",
         "👥 Szegmentáció",
         "⚠️ Churn Előrejelzés",
         "💰 Bevételelemzés",
         "🎯 Akciótér"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Szűrők
    st.markdown("**Szűrők**")
    seg_filter = st.multiselect(
        "Szegmens",
        options=df_seg["segment_label"].unique().tolist(),
        default=df_seg["segment_label"].unique().tolist(),
    )
    churn_threshold = st.slider("Churn küszöb", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.72rem;'>v1.0 · ecommerce-segmentation</div>", unsafe_allow_html=True)


# Szűrt adatok
df_seg_f  = df_seg[df_seg["segment_label"].isin(seg_filter)]
df_churn_f = df_churn[df_churn["segment_label"].isin(seg_filter)]
at_risk   = df_churn_f[df_churn_f["churn_proba"] >= churn_threshold]


# ═══════════════════════════════════════════════════════════
# KPI HELPER
# ═══════════════════════════════════════════════════════════

def kpi(value, label, delta=None, fmt="{:,.0f}", delta_positive_is_good=True):
    val_str = fmt.format(value) if not isinstance(value, str) else value
    delta_html = ""
    if delta is not None:
        good = (delta >= 0) == delta_positive_is_good
        cls  = "kpi-delta-pos" if good else "kpi-delta-neg"
        sign = "▲" if delta > 0 else "▼"
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.1f}%</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{val_str}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 1. OLDAL – ÖSSZEFOGLALÓ
# ═══════════════════════════════════════════════════════════

if page == "📊 Összefoglaló":
    st.markdown(f"""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin-bottom:4px;'>
        Vásárlói Intelligencia Dashboard
    </div>
    <div style='color:{TEXT_MUTED};font-size:0.9rem;margin-bottom:28px;'>
        Online Retail II · 2009–2011 · UK ajándékáru-nagykereskedő
    </div>
    """, unsafe_allow_html=True)

    # ── KPI sor ────────────────────────────────────────────
    total_customers = len(df_seg_f)
    total_revenue   = df_seg_f["monetary_total"].sum()
    churn_rate      = (at_risk.shape[0] / max(len(df_churn_f), 1)) * 100
    vip_count       = df_seg_f[df_seg_f["segment_label"] == "VIP Bajnokok"].shape[0]
    avg_freq        = df_seg_f["frequency"].mean()
    avg_recency     = df_seg_f["recency_days"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(kpi(total_customers, "Ügyfelek száma", +4.2), unsafe_allow_html=True)
    with c2: st.markdown(kpi(total_revenue, "Összbevétel (£)", +8.7, fmt="£{:,.0f}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi(churn_rate, "Churn ráta (%)", -2.1, fmt="{:.1f}%", delta_positive_is_good=False), unsafe_allow_html=True)
    with c4: st.markdown(kpi(vip_count, "VIP Bajnokok", +1.8), unsafe_allow_html=True)
    with c5: st.markdown(kpi(avg_freq, "Átl. vásárlási freq.", None, fmt="{:.1f}x"), unsafe_allow_html=True)
    with c6: st.markdown(kpi(avg_recency, "Átl. recency (nap)", -3.4, delta_positive_is_good=False), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Szegmens donut + Bevétel idősor ────────────────────
    left, right = st.columns([1, 2])

    with left:
        section("Ügyfélszegmensek")
        seg_counts = df_seg_f["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["Szegmens", "Darab"]
        fig_donut = go.Figure(go.Pie(
            labels=seg_counts["Szegmens"],
            values=seg_counts["Darab"],
            hole=0.62,
            marker=dict(colors=[SEGMENT_COLORS.get(s, "#888") for s in seg_counts["Szegmens"]],
                        line=dict(color=BRAND_DARK, width=3)),
            textinfo="percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{value:,} ügyfél (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total_customers:,}</b><br><span style='font-size:11px'>ügyfél</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#fff"),
            align="center",
        )
        fig_donut.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        section("Havi Bevételi Trend (2009–2011)")
        if ts_df is not None:
            ts_monthly = ts_df.copy()
            ts_monthly["month"] = ts_monthly["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
            ts_monthly = ts_monthly.groupby("month")["revenue"].sum().reset_index()
            ts_monthly["MA3"] = ts_monthly["revenue"].rolling(3).mean()

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(
                x=ts_monthly["month"], y=ts_monthly["revenue"],
                name="Havi bevétel",
                marker_color=ACCENT_TEAL,
                opacity=0.45,
                hovertemplate="<b>%{x|%Y %b}</b><br>£%{y:,.0f}<extra></extra>",
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_monthly["month"], y=ts_monthly["MA3"],
                name="3 hó mozgóátlag",
                line=dict(color=BRAND_RED, width=2.5),
                mode="lines",
                hovertemplate="<b>%{x|%Y %b}</b><br>MA3: £%{y:,.0f}<extra></extra>",
            ))
            # JAVÍTÁS: Két lépésben alkalmazzuk a layout frissítést
            fig_ts.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
            fig_ts.update_layout(height=320, legend=dict(orientation="h", y=1.05, x=0))
            st.plotly_chart(fig_ts, use_container_width=True)

    # ── Szegmens bevétel hozzájárulás ──────────────────────
    section("Szegmensenkénti Bevételi Hozzájárulás")
    seg_rev = df_seg_f.groupby("segment_label").agg(
        Bevétel=("monetary_total", "sum"),
        Ügyfelek=("Customer ID", "count"),
        Átl_bevétel=("monetary_total", "mean"),
    ).reset_index().sort_values("Bevétel", ascending=False)
    seg_rev["Bevétel %"] = seg_rev["Bevétel"] / seg_rev["Bevétel"].sum() * 100

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=seg_rev["segment_label"],
        x=seg_rev["Bevétel"],
        orientation="h",
        marker=dict(
            color=[SEGMENT_COLORS.get(s, "#888") for s in seg_rev["segment_label"]],
            line=dict(color="rgba(0,0,0,0)"),
        ),
        text=[f"£{v:,.0f}  ({p:.1f}%)" for v, p in zip(seg_rev["Bevétel"], seg_rev["Bevétel %"])],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>Bevétel: £%{x:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=260,
                          xaxis_title="Összbevétel (£)", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 2. OLDAL – SZEGMENTÁCIÓ
# ═══════════════════════════════════════════════════════════

elif page == "👥 Szegmentáció":
    st.markdown(f"<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin-bottom:24px;'>RFM Szegmentáció</div>", unsafe_allow_html=True)

    # ── 3D scatter – RFM tér ───────────────────────────────
    section("Ügyfelek az RFM Térben (3D)")
    st.caption("Recency vs Frequency vs Monetary · Szegmensenként színezve · Forgatható és zoomolható")

    sample = df_seg_f.sample(min(2000, len(df_seg_f)), random_state=42)
    fig_3d = px.scatter_3d(
        sample,
        x="recency_days", y="frequency", z="monetary_total",
        color="segment_label",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.72,
        size_max=6,
        labels={"recency_days": "Recency (nap)", "frequency": "Frequency", "monetary_total": "Monetary (£)", "segment_label": "Szegmens"},
        hover_data={"Customer ID": True, "recency_days": ":.0f", "frequency": ":.0f", "monetary_total": ":,.0f"},
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=520,
                          scene=dict(
                              bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(backgroundcolor=BRAND_CARD, gridcolor="#1e2d3d", color="#cdd3de"),
                              yaxis=dict(backgroundcolor=BRAND_CARD, gridcolor="#1e2d3d", color="#cdd3de"),
                              zaxis=dict(backgroundcolor=BRAND_CARD, gridcolor="#1e2d3d", color="#cdd3de"),
                          ))
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Snake Plot ─────────────────────────────────────────
    section("Snake Plot – Szegmensprofilok (standardizált RFM)")
    st.caption("Az átlagtól való eltérés · pozitív = jobb, negatív = rosszabb az adott dimenzión")

    rfm_metrics = ["recency_days", "frequency", "monetary_total", "return_ratio"]
    rfm_labels  = ["Recency (nap)", "Frequency", "Monetary (£)", "Return Ráta"]

    snake = df_seg_f.groupby("segment_label")[rfm_metrics].mean()
    snake_std = (snake - snake.mean()) / snake.std()
    # Recency-nél a kisebb a jobb → megfordítjuk
    snake_std["recency_days"] = -snake_std["recency_days"]

    fig_snake = go.Figure()
    for seg, row in snake_std.iterrows():
        fig_snake.add_trace(go.Scatter(
            x=rfm_labels, y=row.values,
            name=seg,
            mode="lines+markers",
            line=dict(color=SEGMENT_COLORS.get(seg, "#888"), width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate=f"<b>{seg}</b><br>%{{x}}: %{{y:.2f}} σ<extra></extra>",
        ))
    fig_snake.add_hline(y=0, line_color="#334155", line_dash="dash", line_width=1)
    
    # JAVÍTÁS: Két lépésben alkalmazzuk a layout frissítést
    fig_snake.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig_snake.update_layout(height=380, yaxis_title="Standardizált eltérés (σ)",
                            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    st.plotly_chart(fig_snake, use_container_width=True)

    # ── Szegmens összefoglaló táblázat ────────────────────
    section("Szegmens Összefoglaló")
    tbl = df_seg_f.groupby("segment_label").agg(
        Ügyfelek=("Customer ID", "count"),
        Átl_Recency=("recency_days", "mean"),
        Átl_Frequency=("frequency", "mean"),
        Átl_Monetary=("monetary_total", "mean"),
        Össz_Bevétel=("monetary_total", "sum"),
        Átl_Return_Ráta=("return_ratio", "mean"),
    ).reset_index().rename(columns={"segment_label": "Szegmens"})
    tbl["Átl_Recency"]      = tbl["Átl_Recency"].round(0).astype(int).astype(str) + " nap"
    tbl["Átl_Frequency"]    = tbl["Átl_Frequency"].round(1)
    tbl["Átl_Monetary"]     = tbl["Átl_Monetary"].map("£{:,.0f}".format)
    tbl["Össz_Bevétel"]     = tbl["Össz_Bevétel"].map("£{:,.0f}".format)
    tbl["Átl_Return_Ráta"]  = (tbl["Átl_Return_Ráta"] * 100).round(1).astype(str) + "%"
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Boxplot – Monetary eloszlás szegmensenként ─────────
    section("Monetary Eloszlás Szegmensenként")
    fig_box = px.box(
        df_seg_f, x="segment_label", y="monetary_total",
        color="segment_label",
        color_discrete_map=SEGMENT_COLORS,
        points="outliers",
        labels={"segment_label": "Szegmens", "monetary_total": "Összbevétel (£)"},
    )
    fig_box.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=360,
                           showlegend=False)
    fig_box.update_yaxes(type="log", title="Összbevétel (£, log skála)")
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 3. OLDAL – CHURN ELŐREJELZÉS
# ═══════════════════════════════════════════════════════════

elif page == "⚠️ Churn Előrejelzés":
    st.markdown(f"<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin-bottom:24px;'>Churn Előrejelzés (XGBoost)</div>", unsafe_allow_html=True)

    # KPI sor
    c1, c2, c3, c4 = st.columns(4)
    n_churn   = (df_churn_f["churn_proba"] >= churn_threshold).sum()
    n_safe    = (df_churn_f["churn_proba"] < churn_threshold).sum()
    vip_risk  = df_churn_f[(df_churn_f["churn_proba"] >= churn_threshold) & (df_churn_f["monetary_total"] >= 2000)].shape[0]
    rev_at_risk = df_churn_f[df_churn_f["churn_proba"] >= churn_threshold]["monetary_total"].sum()

    with c1: st.markdown(kpi(n_churn, f"Lemorzsolódó (≥{churn_threshold:.0%})", fmt="{:,.0f}"), unsafe_allow_html=True)
    with c2: st.markdown(kpi(n_safe,  "Stabil ügyfél", fmt="{:,.0f}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi(vip_risk, "VIP Veszélyben", fmt="{:,.0f}"), unsafe_allow_html=True)
    with c4: st.markdown(kpi(rev_at_risk, "Veszélyeztetett bevétel (£)", fmt="£{:,.0f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Churn valószínűség hisztogram ──────────────────────
    section("Churn Valószínűség Eloszlása")
    fig_hist = go.Figure()
    for seg_name in df_churn_f["segment_label"].unique():
        sub = df_churn_f[df_churn_f["segment_label"] == seg_name]
        fig_hist.add_trace(go.Histogram(
            x=sub["churn_proba"], name=seg_name,
            nbinsx=40,
            marker_color=SEGMENT_COLORS.get(seg_name, "#888"),
            opacity=0.75,
        ))
    fig_hist.add_vline(x=churn_threshold, line_color=ACCENT_GOLD, line_dash="dash",
                       annotation_text=f"Küszöb: {churn_threshold:.0%}",
                       annotation_font_color=ACCENT_GOLD)
    
    # JAVÍTÁS: Két lépésben alkalmazzuk a layout frissítést
    fig_hist.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig_hist.update_layout(height=340, barmode="overlay", xaxis_title="Churn valószínűség",
                           yaxis_title="Ügyfelek száma", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter: Monetary vs Churn proba ───────────────────
    section("Ügyfelek Kockázati Térképe")
    st.caption("X: churn valószínűség · Y: összbevétel · Méret: vásárlási frekvencia · Szín: szegmens")

    sample_c = df_churn_f.sample(min(1500, len(df_churn_f)), random_state=7)
    fig_scatter = px.scatter(
        sample_c,
        x="churn_proba", y="monetary_total",
        color="segment_label",
        size="recency_days",
        size_max=18,
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.65,
        labels={
            "churn_proba":    "Churn valószínűség",
            "monetary_total": "Összbevétel (£)",
            "segment_label":  "Szegmens",
            "recency_days":   "Recency (nap)",
        },
        hover_data={"Customer ID": True, "churn_proba": ":.2f", "monetary_total": ":,.0f"},
    )
    # Kvadráns vonalak
    fig_scatter.add_vline(x=churn_threshold, line_color="#334155", line_dash="dot")
    fig_scatter.add_hline(y=df_churn_f["monetary_total"].quantile(0.75), line_color="#334155", line_dash="dot")
    # Kvadráns feliratok
    for xt, yt, txt in [
        (0.1, df_churn_f["monetary_total"].quantile(0.85), "💎 VIP Stabil"),
        (0.75, df_churn_f["monetary_total"].quantile(0.85), "🚨 VIP Veszélyben"),
        (0.1, df_churn_f["monetary_total"].quantile(0.15), "✅ Kis kockázat"),
        (0.75, df_churn_f["monetary_total"].quantile(0.15), "⚠️ Magas kockázat"),
    ]:
        fig_scatter.add_annotation(x=xt, y=yt, text=txt,
                                    showarrow=False, font=dict(size=10, color=TEXT_MUTED))
    fig_scatter.update_yaxes(type="log")
    
    # JAVÍTÁS: Két lépésben alkalmazzuk a layout frissítést
    fig_scatter.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig_scatter.update_layout(height=440, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── TOP 20 VIP veszélyben ──────────────────────────────
    section("🚨 TOP 20 VIP Veszélyben")
    vip_tbl = df_churn_f.nlargest(20, "monetary_total")[
        ["Customer ID", "segment_label", "monetary_total", "churn_proba", "recency_days", "return_ratio"]
    ].copy()
    vip_tbl["churn_proba"]   = (vip_tbl["churn_proba"] * 100).round(1).astype(str) + "%"
    vip_tbl["monetary_total"] = vip_tbl["monetary_total"].map("£{:,.0f}".format)
    vip_tbl["return_ratio"]  = (vip_tbl["return_ratio"] * 100).round(1).astype(str) + "%"
    vip_tbl["recency_days"]  = vip_tbl["recency_days"].round(0).astype(int).astype(str) + " nap"
    vip_tbl = vip_tbl.rename(columns={
        "Customer ID": "Ügyfél ID", "segment_label": "Szegmens",
        "monetary_total": "Összbevétel", "churn_proba": "Churn %",
        "recency_days": "Recency", "return_ratio": "Return ráta",
    })
    st.dataframe(vip_tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# 4. OLDAL – BEVÉTELELEMZÉS
# ═══════════════════════════════════════════════════════════

elif page == "💰 Bevételelemzés":
    st.markdown(f"<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin-bottom:24px;'>Bevételelemzés</div>", unsafe_allow_html=True)

    # ── Bevételi idősor ────────────────────────────────────
    section("Bevételi Trend és Szezonalitás")
    if ts_df is not None:
        ts_monthly = ts_df.copy()
        ts_monthly["month"] = ts_monthly["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
        ts_monthly = ts_monthly.groupby("month")["revenue"].sum().reset_index()
        ts_monthly["YoY_change"] = ts_monthly["revenue"].pct_change(12) * 100
        ts_monthly["MA3"] = ts_monthly["revenue"].rolling(3).mean()

        fig_ts2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.7, 0.3],
                                 subplot_titles=["Havi bevétel (£)", "Éves változás (%)"],
                                 vertical_spacing=0.08)
        fig_ts2.add_trace(go.Bar(
            x=ts_monthly["month"], y=ts_monthly["revenue"],
            marker_color=ACCENT_TEAL, opacity=0.5, name="Bevétel",
            hovertemplate="<b>%{x|%Y %b}</b><br>£%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        fig_ts2.add_trace(go.Scatter(
            x=ts_monthly["month"], y=ts_monthly["MA3"],
            line=dict(color=BRAND_RED, width=2.5), name="3 hó MA",
            hovertemplate="£%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        fig_ts2.add_trace(go.Bar(
            x=ts_monthly["month"], y=ts_monthly["YoY_change"],
            marker_color=[ACCENT_GREEN if v >= 0 else BRAND_RED for v in ts_monthly["YoY_change"].fillna(0)],
            name="YoY %", opacity=0.8,
            hovertemplate="%{y:.1f}%<extra></extra>",
        ), row=2, col=1)
        fig_ts2.add_hline(y=0, row=2, col=1, line_color="#334155")
        layout_dict = PLOTLY_TEMPLATE["layout"].to_plotly_json()
        layout_dict["height"] = 480
        layout_dict["showlegend"] = True
        fig_ts2.update_layout(**layout_dict)
        fig_ts2.update_yaxes(gridcolor="#1e2d3d")
        st.plotly_chart(fig_ts2, use_container_width=True)

    # ── Pareto analízis (80/20) ─────────────────────────────
    section("Pareto-analízis: Ügyfelek vs Bevétel")
    st.caption("Hány százalékuk termeli a bevétel 80%-át?")

    sorted_customers = df_seg_f.sort_values("monetary_total", ascending=False).reset_index(drop=True)
    sorted_customers["cumrev"] = sorted_customers["monetary_total"].cumsum()
    sorted_customers["cumrev_pct"] = sorted_customers["cumrev"] / sorted_customers["monetary_total"].sum() * 100
    sorted_customers["cust_pct"] = (sorted_customers.index + 1) / len(sorted_customers) * 100

    # Pareto pont
    pareto_idx = sorted_customers[sorted_customers["cumrev_pct"] >= 80].index[0]
    pareto_cust_pct = sorted_customers.loc[pareto_idx, "cust_pct"]

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=sorted_customers["cust_pct"], y=sorted_customers["cumrev_pct"],
        mode="lines", name="Kumulált bevétel %",
        line=dict(color=BRAND_RED, width=3),
        fill="tozeroy", fillcolor="rgba(192,37,63,0.12)",
        hovertemplate="Top %{x:.1f}% ügyfél → %{y:.1f}% bevétel<extra></extra>",
    ))
    fig_pareto.add_hline(y=80, line_color=ACCENT_GOLD, line_dash="dash")
    fig_pareto.add_vline(x=pareto_cust_pct, line_color=ACCENT_GOLD, line_dash="dash",
                          annotation_text=f"Top {pareto_cust_pct:.1f}% → 80% bevétel",
                          annotation_font_color=ACCENT_GOLD, annotation_yshift=-20)
    fig_pareto.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=380,
                              xaxis_title="Ügyfelek (kumulált %)",
                              yaxis_title="Bevétel (kumulált %)")
    st.plotly_chart(fig_pareto, use_container_width=True)

    # ── Bevételi koncentráció ─────────────────────────────
    section("Bevételi Koncentráció (Szegmens × Recency bucket)")
    df_seg_f2 = df_seg_f.copy()
    df_seg_f2["recency_bucket"] = pd.cut(
        df_seg_f2["recency_days"],
        bins=[0, 30, 90, 180, 365, 9999],
        labels=["0–30 nap", "31–90", "91–180", "181–365", "365+ nap"],
    )
    heat_data = df_seg_f2.groupby(["segment_label", "recency_bucket"])["monetary_total"].sum().unstack(fill_value=0)

    fig_heat = px.imshow(
        heat_data.values,
        x=heat_data.columns.astype(str),
        y=heat_data.index,
        color_continuous_scale=[[0, BRAND_DARK], [0.5, "#0f3460"], [1, BRAND_RED]],
        aspect="auto",
        text_auto=False,
        labels=dict(x="Recency bucket", y="Szegmens", color="Bevétel (£)"),
    )
    fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=280)
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 5. OLDAL – AKCIÓTÉR
# ═══════════════════════════════════════════════════════════

elif page == "🎯 Akciótér":
    st.markdown(f"<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin-bottom:24px;'>Üzleti Akciótér</div>", unsafe_allow_html=True)

    # ── Prioritizált akciólista ────────────────────────────
    section("Prioritizált Ügyfél Akciólista")
    st.caption(f"Churn küszöb: {churn_threshold:.0%} · Szűrt szegmensek: {', '.join(seg_filter)}")

    action_df = df_churn_f.copy()

    # A valós pipeline 'action' oszlopát használjuk, ha létezik
    # Egyébként újrageneráljuk a churn_proba és monetary_total alapján
    if "action" not in action_df.columns or action_df["action"].isna().all():
        action_df["action"] = np.where(
            (action_df["churn_proba"] >= churn_threshold) & (action_df["monetary_total"] >= 2000),
            "🚨 VIP Veszélyben – Azonnali Retenció",
            np.where(action_df["churn_proba"] >= churn_threshold, "⚠️ Retenció ajánlott",
            np.where(action_df["churn_proba"] < 0.20, "✅ Stabil – Upsell lehetőség",
                     "👀 Figyelendő – Soft nurturing"))
        )

    action_df["Prioritás"] = action_df["action"]

    # Összefoglaló
    action_summary = action_df["Prioritás"].value_counts().reset_index()
    action_summary.columns = ["Prioritás", "Ügyfelek"]
    priority_colors = {
        "🚨 VIP Veszélyben – Azonnali Retenció": BRAND_RED,
        "⚠️ Retenció ajánlott":                   ACCENT_GOLD,
        "👀 Figyelendő – Soft nurturing":           ACCENT_TEAL,
        "✅ Stabil – Upsell lehetőség":             ACCENT_GREEN,
    }
    fig_act = go.Figure(go.Bar(
        y=action_summary["Prioritás"],
        x=action_summary["Ügyfelek"],
        orientation="h",
        marker_color=[priority_colors.get(p, "#888") for p in action_summary["Prioritás"]],
        text=action_summary["Ügyfelek"].astype(str) + " ügyfél",
        textposition="outside",
    ))
    fig_act.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=250,
                           xaxis_title="Ügyfelek száma", yaxis_title="",
                           showlegend=False)
    st.plotly_chart(fig_act, use_container_width=True)

    # Szűrhető táblázat
    section("Letölthető Akciólista")
    priority_sel = st.multiselect(
        "Prioritás szűrő",
        options=action_df["Prioritás"].unique().tolist(),
        default=["🚨 VIP Veszélyben – Azonnali Retenció", "⚠️ Retenció ajánlott"],
    )
    export_df = action_df[action_df["Prioritás"].isin(priority_sel)][
        ["Customer ID", "segment_label", "monetary_total", "churn_proba", "recency_days", "Prioritás"]
    ].sort_values("monetary_total", ascending=False).copy()

    export_df["churn_proba"]    = (export_df["churn_proba"] * 100).round(1).astype(str) + "%"
    export_df["monetary_total"] = export_df["monetary_total"].map("£{:,.0f}".format)
    export_df["recency_days"]   = export_df["recency_days"].round(0).astype(int).astype(str) + " nap"
    export_df = export_df.rename(columns={
        "Customer ID": "Ügyfél ID", "segment_label": "Szegmens",
        "monetary_total": "Összbevétel", "churn_proba": "Churn %",
        "recency_days": "Recency",
    })
    st.dataframe(export_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Exportálás CSV-be",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="akciolista.csv",
        mime="text/csv",
    )

    # ── Üzleti ajánlások ─────────────────────────────────
    section("📋 Szegmensenkénti Üzleti Ajánlások")

    recs = {
        "VIP Bajnokok": {
            "color": BRAND_RED,
            "icon": "💎",
            "actions": [
                "Személyre szabott VIP hűségprogram (early access, exkluzív kedvezmények)",
                "Dedikált account manager vagy priority support",
                "Éves vásárlói visszatekintő riport küldése",
                "Visszaküldési arány figyelése – magas return ráta normális, de kiugrósnál vizsgálni",
            ],
            "goal": "Megtartás + referral aktiválás",
        },
        "Új / Ígéretes": {
            "color": ACCENT_TEAL,
            "icon": "🤝",
            "actions": [
                "Upsell kampányok a következő szint (VIP) felé",
                "Bundle ajánlatok a rendelési frekvencia növeléséhez",
                "Email nurturing: top-selling termékek, újdonságok",
                "Szezonális előrendelési lehetőségek",
            ],
            "goal": "Upgrade VIP-be + keresztértékesítés",
        },
        "Lemorzsolódó / Alvó": {
            "color": ACCENT_GOLD,
            "icon": "⚠️",
            "actions": [
                "Win-back email sorozat (3 levél, 2 hetes időközzel)",
                "Személyre szabott kedvezmény az utolsó vásárláshoz hasonló kategóriában",
                "SMS/push értesítő: 'Hiányoztál!'",
                "Churn okának felmérése rövid survey-jel",
            ],
            "goal": "Reaktiváció a lemorzsolódás előtt",
        },
        "Elvesztett / Inaktív": {
            "color": "#4a4e69",
            "icon": "🌑",
            "actions": [
                "Utolsó esély kampány: 15-20% kedvezmény időkorlátos ajánlattal",
                "Ha nincs válasz 60 napon belül: szüneteltetés + GDPR-tisztítás",
                "Hasonló profilú (lookalike) ügyfelek keresése a megtartott szegmensekben",
                "Tanulságok integrálása az onboarding folyamatba",
            ],
            "goal": "Alacsony ROI – szelektív reaktiváció",
        },
    }
    for seg_name, rec in recs.items():
        if seg_name.replace("VIP Bajnokok", "VIP Bajnokok") in seg_filter or True:
            with st.expander(f"{rec['icon']} {seg_name} – {rec['goal']}", expanded=False):
                for action in rec["actions"]:
                    st.markdown(f"- {action}")