import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ügyfélszegmentáció Dashboard",
    page_icon="🎯",
    layout="wide",
)

# ── Konstansok és stílus ───────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#2ecc71",
    "Lemorzsolódó / Alvó":    "#e67e22",
    "Elvesztett / Inaktív":   "#e74c3c",
    "Új / Ígéretes":          "#3498db",
}

# ── Adat betöltő (Gyorsítótárazva) ────────────────────────────────────────────
@st.cache_data(show_spinner="Adatok betöltése...")
def load_data():
    """Betölti a customer_segments.parquet fájlt."""
    # Útvonal keresése (kompatibilis lokális és telepített futtatással is)
    file_path = Path("data/processed/customer_segments.parquet")
    if not file_path.exists():
        file_path = Path("../data/processed/customer_segments.parquet")
        
    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        st.error("⚠️ Nem található a customer_segments.parquet fájl!")
        st.stop()
        
    # Szegmens névoszlop egységesítése
    if "Segment" not in df.columns and "cluster" in df.columns:
        # Ha a Segment oszlop valamiért hiányzik, de a cluster megvan
        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols):
            df = df.rename(columns={str_cols[0]: "Segment"})
            
    return df

# ── Fő UI ─────────────────────────────────────────────────────────────────────
st.title("🎯 Interaktív RFM Ügyfélszegmentáció Dashboard")
st.markdown("Fedezd fel az ügyfélbázis szerkezetét a **Recency (Újdonság)**, **Frequency (Gyakoriság)** és **Monetary (Pénzérték)** mutatók alapján.")

df = load_data()

# ── 1. KPI Kártyák ────────────────────────────────────────────────────────────
st.subheader("📌 Főbb Mutatók (KPI-ok)")

total_customers = len(df)
avg_basket_value = df['monetary_avg'].mean()
num_segments = df['Segment'].nunique()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(label="Aktív Vásárlók Száma", value=f"{total_customers:,} fő")
kpi2.metric(label="Átlagos Kosárérték", value=f"£{avg_basket_value:,.2f}")
kpi3.metric(label="Azonosított Szegmensek", value=f"{num_segments} db")

st.markdown("---")

# ── 2. Sor: Donut Chart & Radar Chart ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Szegmensek Aránya")
    
    # Adat-előkészítés a Donut charthoz
    seg_counts = df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Szegmens', 'Ügyfelek Száma']
    
    fig_donut = px.pie(
        seg_counts,
        names="Szegmens",
        values="Ügyfelek Száma",
        hole=0.45,
        color="Szegmens",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig_donut.update_traces(textinfo="percent+label", textposition="inside", textfont_size=14)
    fig_donut.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
    
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("Radar Chart: Szegmens vs Globális Átlag")
    
    # Szegmens kiválasztása a Radarhoz
    selected_segment = st.selectbox("Válassz szegmenst a profilozáshoz:", df['Segment'].unique())
    
    # 0-1 (MinMax) skálázás a vizuális ábrázolhatóság kedvéért
    features = ['recency_days', 'frequency', 'monetary_total']
    labels = ['Recency (Napok)', 'Frequency (Tranzakciók)', 'Monetary (Költés)']
    
    # Globális és Szegmens átlagok kiszámítása
    global_means = df[features].mean()
    segment_means = df[df['Segment'] == selected_segment][features].mean()
    
    # Normalizálás a Radar ábrához (hogy egy skálán jól mutassanak)
    max_vals = df[features].max()
    min_vals = df[features].min()
    
    def normalize(val, col):
        # Recency esetében a kisebb érték a jobb, de a radaron ábrázolhatjuk az inverzét is.
        # Itt maradunk a tiszta normalizált értéknél (0-1).
        return (val - min_vals[col]) / (max_vals[col] - min_vals[col])

    global_norm = [normalize(global_means[f], f) for f in features]
    segment_norm = [normalize(segment_means[f], f) for f in features]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=global_norm + [global_norm[0]], # Körbezárás miatt
        theta=labels + [labels[0]],
        fill='toself',
        name='Globális Átlag',
        line_color='gray',
        opacity=0.4
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=segment_norm + [segment_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name=selected_segment,
        line_color=SEGMENT_COLORS.get(selected_segment, '#333'),
        opacity=0.8
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, max(max(global_norm), max(segment_norm)) + 0.1])),
        showlegend=True,
        margin=dict(t=30, b=10, l=10, r=10)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── 3. Sor: 3D Interaktív Scatter Plot ────────────────────────────────────────
st.subheader("🌌 3D RFM Szegmens Térkép")
st.markdown("Forgasd és nagyítsd a diagramot! Az adatpontok csökkentve (sample) jelennek meg a böngésző tehermentesítése érdekében.")

# A nagy adatmennyiség miatt érdemes egy sample-t venni (pl. 2000 elem), hogy az interakció fluid maradjon
sample_size = min(len(df), 2500)
df_sample = df.sample(n=sample_size, random_state=42)

# Eldöntjük, hogy a skálázott vagy a nyers értékeket használjuk-e a tengelyeken.
# A log vizualizáció miatt a nyers értékek is jók, de a modell a skálázottat használta a 02-es notebookban.
x_col = 'recency_scaled' if 'recency_scaled' in df.columns else 'recency_days'
y_col = 'frequency_scaled' if 'frequency_scaled' in df.columns else 'frequency'
z_col = 'monetary_scaled' if 'monetary_scaled' in df.columns else 'monetary_total'

fig_3d = px.scatter_3d(
    df_sample,
    x=x_col,
    y=y_col,
    z=z_col,
    color='Segment',
    color_discrete_map=SEGMENT_COLORS,
    hover_data=['recency_days', 'frequency', 'monetary_total'],
    opacity=0.7,
    size_max=5
)

# Vizuális tisztítás
fig_3d.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='Recency',
        yaxis_title='Frequency',
        zaxis_title='Monetary'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=700,
    legend_title_text='Szegmensek'
)

st.plotly_chart(fig_3d, use_container_width=True)