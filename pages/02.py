"""
pages/04_cluster_3d.py
───────────────────────
K-Means 3D Klaszter Vizualizáció – Streamlit dashboard
Betölti a mentett modellt és adatokat, újraszámolja a dinamikus
segment_map-et, majd interaktív 3D scatter plotot jelenít meg
centroidokkal – pontosan a notebook 4.5-ös cellájának logikájával.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Opcionális imports (joblib csak ha van mentett modell) ────────────────────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ── Konfiguráció ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Klaszter Térkép",
    page_icon="🌐",
    layout="wide",
)

BG_DARK  = "#0F1117"
BG_CARD  = "#1A1D27"
ACCENT   = "#E8472A"
TEXT_PRI = "#F0F0F0"
TEXT_SEC = "#9FA3B1"

# A notebook eredeti colormap-je
SEGMENT_COLOR_MAP = {
    "Lemorzsolódó / Alvó":  "#E69F00",
    "Elvesztett / Inaktív": "#56B4E9",
    "VIP Bajnokok":         "#009E73",
    "Új / Ígéretes":        "#CC79A7",
}

CENTROID_COLOR    = "#FFFFFF"
CENTROID_OUTLINE  = "#000000"

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
    padding: 32px 0 24px; border-bottom: 1px solid #2A2D3A; margin-bottom: 28px;
}}
.page-header h1 {{
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: {TEXT_PRI}; margin: 0; line-height: 1;
}}
.page-header .sub {{ font-size: 0.9rem; color: {TEXT_SEC}; padding-bottom: 4px; }}

.kpi-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A; border-radius: 12px;
    padding: 18px 20px; position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, {ACCENT});
}}
.kpi-label {{ font-size: 0.72rem; text-transform: uppercase;
              letter-spacing: .08em; color: {TEXT_SEC}; margin-bottom: 5px; }}
.kpi-value {{ font-family: 'Syne', sans-serif; font-size: 1.8rem;
              font-weight: 800; color: {TEXT_PRI}; line-height: 1; }}
.kpi-delta {{ font-size: 0.75rem; color: {TEXT_SEC}; margin-top: 4px; }}

.section-title {{
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: {TEXT_PRI}; margin: 0 0 12px; letter-spacing: .03em;
}}
.chart-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A;
    border-radius: 14px; padding: 22px 20px; margin-bottom: 20px;
}}
.info-pill {{
    display: inline-block; background: #2A2D3A; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; color: {TEXT_SEC};
    margin: 2px;
}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ADATBETÖLTÉS – pontosan a notebook logikájával
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_rfm_features():
    """rfm_features.parquet betöltése (a 02-es notebook kimenete)."""
    p = Path("data/processed/rfm_features.parquet")
    if p.exists():
        return pd.read_parquet(p), True
    # Fallback: customer_segments.parquet
    p2 = Path("data/processed/customer_segments.parquet")
    if p2.exists():
        return pd.read_parquet(p2), True
    return None, False


@st.cache_resource(show_spinner=False)
def load_models():
    """KMeans modell + Scaler betöltése joblib-ból."""
    kmeans, scaler = None, None
    if not JOBLIB_OK:
        return kmeans, scaler

    km_path = Path("models/kmeans_rfm.joblib")
    sc_path = Path("models/scaler_rfm.joblib")

    if km_path.exists():
        try:
            kmeans = joblib.load(km_path)
        except Exception:
            pass
    if sc_path.exists():
        try:
            scaler = joblib.load(sc_path)
        except Exception:
            pass
    return kmeans, scaler


def derive_segment_map(rfm_export, labels):
    """
    Dinamikus segment_map levezetése – a notebook 4.4-es cellájának
    pontos másolata, újrafuttatás-biztos RFM-átlag alapján.
    """
    tmp = rfm_export.copy()
    tmp["_cluster"] = labels

    profile = tmp.groupby("_cluster")[
        ["recency_days", "frequency", "monetary_total"]
    ].mean()

    rank_m = profile["monetary_total"].rank(ascending=False)
    rank_f = profile["frequency"].rank(ascending=False)

    vip_idx  = (rank_m + rank_f).idxmin()
    lost_idx = profile["recency_days"].idxmax()

    remaining = [c for c in profile.index if c not in [vip_idx, lost_idx]]
    new_idx   = profile.loc[remaining, "recency_days"].idxmin()
    sleep_idx = [c for c in remaining if c != new_idx][0]

    return {
        vip_idx:   "VIP Bajnokok",
        lost_idx:  "Elvesztett / Inaktív",
        new_idx:   "Új / Ígéretes",
        sleep_idx: "Lemorzsolódó / Alvó",
    }


def build_demo_data():
    """
    Demo adathalmaz generálása, ha nincs parquet.
    Ugyanolyan struktúrát produkál, mint a 02-es notebook kimenete.
    """
    rng = np.random.default_rng(42)
    n   = 5243

    # Nyers RFM értékek
    recency  = rng.integers(0, 646, n).astype(float)
    freq     = rng.integers(1, 60, n).astype(float)
    monetary = rng.exponential(2000, n).clip(10, 450000)
    ret_cnt  = rng.integers(0, 20, n).astype(float)
    ret_ratio = ret_cnt / (freq + ret_cnt)
    mon_avg  = monetary / freq

    # StandardScaler szimulálása (z-score)
    def scale(arr):
        return (arr - arr.mean()) / (arr.std() + 1e-9)

    rec_s = scale(recency)
    frq_s = scale(freq)
    mon_s = scale(monetary)

    df = pd.DataFrame({
        "recency_days":     recency,
        "frequency":        freq,
        "monetary_total":   monetary,
        "return_count":     ret_cnt,
        "monetary_avg":     mon_avg,
        "return_ratio":     ret_ratio,
        "recency_scaled":   rec_s,
        "frequency_scaled": frq_s,
        "monetary_scaled":  mon_s,
    })
    df.index.name = "Customer ID"

    # K-Means szimulálása: 4 szegmens manuális felosztással
    conditions = [
        (rec_s < -0.3) & (frq_s > 0.3) & (mon_s > 0.3),  # VIP
        (rec_s > 0.5),                                       # Elvesztett
        (rec_s < -0.3) & (frq_s <= 0.3),                   # Új/Ígéretes
    ]
    choices  = [2, 1, 3]
    clusters = np.select(conditions, choices, default=0)
    df["cluster"] = clusters

    # Centroidok kiszámítása
    scaled_cols = ["recency_scaled", "frequency_scaled", "monetary_scaled"]
    centroids = (
        df.groupby("cluster")[scaled_cols].mean().values
    )

    return df, clusters, centroids


# ════════════════════════════════════════════════════════════════════════════
# ADATOK ÖSSZERAKÁSA
# ════════════════════════════════════════════════════════════════════════════

rfm_export, data_ok = load_rfm_features()
kmeans, scaler      = load_models()
demo_mode           = not data_ok

SCALED_COLS = ["recency_scaled", "frequency_scaled", "monetary_scaled"]

if demo_mode:
    rfm_export, labels, centroids = build_demo_data()
    segment_map = derive_segment_map(rfm_export, labels)
    st.info("ℹ️  Demo mód: `data/processed/rfm_features.parquet` nem található. "
            "Szimulált adatokat jelenítünk meg.", icon="🔔")

else:
    # Ellenőrzés: van-e már 'cluster' oszlop a parquetban?
    if "cluster" in rfm_export.columns:
        labels = rfm_export["cluster"].values
    elif kmeans is not None and all(c in rfm_export.columns for c in SCALED_COLS):
        labels = kmeans.predict(rfm_export[SCALED_COLS])
        rfm_export["cluster"] = labels
    else:
        # Fallback: KMeans újrafuttatása a scaled oszlopokon
        from sklearn.cluster import KMeans as _KMeans
        _km = _KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = _km.fit_predict(rfm_export[SCALED_COLS])
        rfm_export["cluster"] = labels
        kmeans = _km

    # Centroidok
    if kmeans is not None:
        centroids = kmeans.cluster_centers_
    else:
        centroids = rfm_export.groupby("cluster")[SCALED_COLS].mean().values

    segment_map = derive_segment_map(rfm_export, labels)

# Szegmens-nevek hozzárendelése
rfm_export["Segment"] = rfm_export["cluster"].map(segment_map)


# ════════════════════════════════════════════════════════════════════════════
# FEJLÉC
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
  <div><h1>🌐 3D Klaszter Térkép</h1></div>
  <div class="sub">K-Means K=4 · Skálázott RFM tér · Interaktív centroid vizualizáció</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR – vezérlők
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎛️ Megjelenítés")

    point_opacity = st.slider("Pontok átlátszósága", 0.1, 1.0, 0.6, 0.05)
    point_size    = st.slider("Pontok mérete", 1, 8, 3)
    centroid_size = st.slider("Centroid mérete", 8, 30, 16)

    st.markdown("---")
    st.markdown("### 🎨 Szegmensek")
    show_segs = {}
    for seg, color in SEGMENT_COLOR_MAP.items():
        show_segs[seg] = st.checkbox(
            f"● {seg}", value=True,
            help=f"Szín: {color}"
        )
    show_centroids = st.checkbox("◆ Centroidok", value=True)

    st.markdown("---")
    st.markdown("### 📐 Tengely nézet")
    camera_preset = st.selectbox(
        "Kamera pozíció",
        ["Szabad (interaktív)", "Elölnézet (R×F)", "Oldalnézet (F×M)", "Felülnézet (R×M)"],
    )

    st.markdown("---")
    st.markdown("### 📊 Minta")
    max_points = st.slider(
        "Max. megjelenített pont",
        500, len(rfm_export), min(3000, len(rfm_export)), 250,
    )


# ════════════════════════════════════════════════════════════════════════════
# KPI SÁTOR
# ════════════════════════════════════════════════════════════════════════════

seg_counts  = rfm_export["Segment"].value_counts()
total       = len(rfm_export)
vip_n       = seg_counts.get("VIP Bajnokok", 0)
vip_rev     = rfm_export[rfm_export["Segment"] == "VIP Bajnokok"]["monetary_total"].sum()
lost_n      = seg_counts.get("Elvesztett / Inaktív", 0)

kpis = [
    ("Összes ügyfél",       f"{total:,}",                   "szegmentált",                    ACCENT),
    ("VIP Bajnokok",        f"{vip_n:,}",                   f"bevétel: £{vip_rev/1e3:.0f}K",  "#009E73"),
    ("Elvesztett / Inaktív",f"{lost_n:,}",                  f"{lost_n/total*100:.1f}% arány",  "#56B4E9"),
    ("Centroidok száma",    "4",                             "K-Means K=4 szegmens",           "#E69F00"),
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

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 3D SCATTER – a notebook 4.5-ös cellájának Streamlit-adaptációja
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Ügyfélszegmensek 3D RFM térben (skálázott értékek)</div>',
            unsafe_allow_html=True)

# Mintavétel a teljesítményhez
sample_df = rfm_export.sample(min(max_points, len(rfm_export)), random_state=42)

# Aktív szegmensek szűrése
active_segs  = [s for s, v in show_segs.items() if v]
filtered_df  = sample_df[sample_df["Segment"].isin(active_segs)]

# ── 1. Adatpontok (px.scatter_3d – pontosan mint a notebookban) ───────────
fig = px.scatter_3d(
    filtered_df,
    x="recency_scaled",
    y="frequency_scaled",
    z="monetary_scaled",
    color="Segment",
    opacity=point_opacity,
    color_discrete_map=SEGMENT_COLOR_MAP,
    labels={
        "recency_scaled":   "Recency (Skálázott)",
        "frequency_scaled": "Frequency (Skálázott)",
        "monetary_scaled":  "Monetary (Skálázott)",
    },
    custom_data=["Segment"],
    hover_data={
        "recency_scaled":   ":.3f",
        "frequency_scaled": ":.3f",
        "monetary_scaled":  ":.3f",
        "Segment":          True,
    },
)

fig.update_traces(
    marker=dict(size=point_size, line=dict(width=0)),
    selector=dict(mode="markers"),
)

# ── 2. Centroidok hozzáadása (go.Scatter3d – pontosan mint a notebookban) ─
if show_centroids and len(centroids) > 0:
    # Centroid szegmens-nevek a segment_map alapján
    n_clusters = len(centroids)
    centroid_names = [
        segment_map.get(i, f"Klaszter {i}") for i in range(n_clusters)
    ]
    centroid_colors = [
        SEGMENT_COLOR_MAP.get(segment_map.get(i, ""), "#CCCCCC")
        for i in range(n_clusters)
    ]

    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode="markers+text",
        marker=dict(
            size=centroid_size,
            color=CENTROID_COLOR,
            symbol="diamond",
            opacity=1.0,
            line=dict(width=2.5, color=CENTROID_OUTLINE),
        ),
        text=centroid_names,
        textposition="top center",
        textfont=dict(size=10, color=TEXT_PRI, family="DM Sans"),
        name="Klaszter Középpontok (Centroidok)",
        hovertemplate=(
            "<b>Centroid: %{text}</b><br>"
            "R: %{x:.3f}<br>"
            "F: %{y:.3f}<br>"
            "M: %{z:.3f}<extra></extra>"
        ),
    ))

# ── 3. Kamera preset ──────────────────────────────────────────────────────
camera_map = {
    "Szabad (interaktív)":    dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    "Elölnézet (R×F)":        dict(eye=dict(x=0, y=0, z=2.5)),
    "Oldalnézet (F×M)":       dict(eye=dict(x=2.5, y=0, z=0)),
    "Felülnézet (R×M)":       dict(eye=dict(x=0, y=2.5, z=0)),
}

# ── 4. Layout (a notebook template='plotly_white' -> adaptálva dark theme-re) ─
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    legend_title_text="Ügyfélszegmensek",
    legend=dict(
        bgcolor="rgba(26,29,39,0.9)",
        bordercolor="#2A2D3A",
        borderwidth=1,
        font=dict(color=TEXT_PRI, size=12, family="DM Sans"),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        bgcolor=BG_DARK,
        xaxis=dict(
            title="Recency (Skálázott)",
            backgroundcolor=BG_CARD,
            gridcolor="#2A2D3A",
            showbackground=True,
            zerolinecolor="#3A3D4A",
            tickfont=dict(color=TEXT_SEC, size=9),
            title_font=dict(color=TEXT_SEC, size=11),
        ),
        yaxis=dict(
            title="Frequency (Skálázott)",
            backgroundcolor=BG_CARD,
            gridcolor="#2A2D3A",
            showbackground=True,
            zerolinecolor="#3A3D4A",
            tickfont=dict(color=TEXT_SEC, size=9),
            title_font=dict(color=TEXT_SEC, size=11),
        ),
        zaxis=dict(
            title="Monetary (Skálázott)",
            backgroundcolor=BG_CARD,
            gridcolor="#2A2D3A",
            showbackground=True,
            zerolinecolor="#3A3D4A",
            tickfont=dict(color=TEXT_SEC, size=9),
            title_font=dict(color=TEXT_SEC, size=11),
        ),
        camera=camera_map[camera_preset],
        aspectmode="cube",
    ),
    font=dict(color=TEXT_PRI, family="DM Sans"),
    height=650,
    uirevision="stable",   # megőrzi a kamera pozícióját widgetek változásakor
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CENTROID KOORDINÁTA TÁBLA
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">◆ Klaszter centroidok koordinátái (skálázott RFM tér)</div>',
            unsafe_allow_html=True)

centroid_rows = []
for i, (cx, cy, cz) in enumerate(centroids):
    seg_name = segment_map.get(i, f"Klaszter {i}")
    n_members = (rfm_export["cluster"] == i).sum()
    pct       = n_members / total * 100
    centroid_rows.append({
        "Szegmens":              seg_name,
        "Recency (skálázott)":   round(cx, 4),
        "Frequency (skálázott)": round(cy, 4),
        "Monetary (skálázott)":  round(cz, 4),
        "Ügyfelek (db)":         n_members,
        "Arány (%)":             round(pct, 1),
    })

centroid_df = pd.DataFrame(centroid_rows).sort_values("Ügyfelek (db)", ascending=False)

color_map_display = {
    row["Szegmens"]: SEGMENT_COLOR_MAP.get(row["Szegmens"], "#888")
    for _, row in centroid_df.iterrows()
}

st.dataframe(
    centroid_df.style.format({
        "Recency (skálázott)":   "{:+.4f}",
        "Frequency (skálázott)": "{:+.4f}",
        "Monetary (skálázott)":  "{:+.4f}",
        "Arány (%)":             "{:.1f}%",
    }).background_gradient(
        subset=["Monetary (skálázott)"], cmap="RdYlGn"
    ).background_gradient(
        subset=["Recency (skálázott)"], cmap="RdYlGn_r"
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SZEGMENS STATISZTIKA (nyers értékek – üzleti olvashatóság)
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Szegmens-profil (nyers RFM átlagok)</div>',
            unsafe_allow_html=True)

raw_cols = [c for c in ["recency_days","frequency","monetary_total","return_ratio"]
            if c in rfm_export.columns]

profile_tbl = rfm_export.groupby("Segment")[raw_cols].mean().round(2).reset_index()
profile_tbl = profile_tbl.rename(columns={
    "recency_days":   "Átl. inaktivitás (nap)",
    "frequency":      "Átl. vásárlás (db)",
    "monetary_total": "Átl. bevétel (£)",
    "return_ratio":   "Átl. visszaküldési arány",
})

fmt2 = {}
if "Átl. inaktivitás (nap)" in profile_tbl.columns:
    fmt2["Átl. inaktivitás (nap)"] = "{:.0f}"
if "Átl. vásárlás (db)" in profile_tbl.columns:
    fmt2["Átl. vásárlás (db)"] = "{:.1f}"
if "Átl. bevétel (£)" in profile_tbl.columns:
    fmt2["Átl. bevétel (£)"] = "£{:,.0f}"
if "Átl. visszaküldési arány" in profile_tbl.columns:
    fmt2["Átl. visszaküldési arány"] = "{:.1%}"

st.dataframe(
    profile_tbl.style.format(fmt2).background_gradient(
        subset=["Átl. bevétel (£)"] if "Átl. bevétel (£)" in profile_tbl.columns else [],
        cmap="YlOrRd",
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# METODOLÓGIA MEGJEGYZÉS
# ════════════════════════════════════════════════════════════════════════════

model_source = "betöltött `kmeans_rfm.joblib`" if (kmeans is not None and not demo_mode) else "szimulált (demo)"
scaler_source = "betöltött `scaler_rfm.joblib`" if (scaler is not None and not demo_mode) else "z-score (szimulált)"

st.markdown(f"""
<div style="background:{BG_CARD};border:1px solid #2A2D3A;border-left:4px solid #56B4E9;
            border-radius:10px;padding:18px 22px;margin-top:4px;">
  <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:10px;color:{TEXT_PRI}">
    📐 Metodológiai megjegyzések
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;
              color:{TEXT_SEC};font-size:0.86rem;line-height:1.7">
    <div>
      <b style="color:{TEXT_PRI}">Modell:</b> {model_source}<br>
      <b style="color:{TEXT_PRI}">Scaler:</b> {scaler_source}<br>
      <b style="color:{TEXT_PRI}">Klaszterszám:</b> K=4 (üzleti döntés, K=2 automatikus javaslattal szemben)
    </div>
    <div>
      <b style="color:{TEXT_PRI}">Segment map:</b> dinamikus, RFM-átlag alapján (újrafuttatás-biztos)<br>
      <b style="color:{TEXT_PRI}">Silhouette (K=4):</b> 0.3779 · <b style="color:{TEXT_PRI}">Davies-Bouldin:</b> 0.9254<br>
      <b style="color:{TEXT_PRI}">Megfigyelési ablak:</b> 2009-12-01 → 2011-09-09
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
