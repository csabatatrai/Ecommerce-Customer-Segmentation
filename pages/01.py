import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# =============================================================================
# E-Commerce Customer Analytics Dashboard
# =============================================================================

# --- Útvonalak beállítása ---
current_dir = Path(__file__).parent
if current_dir.name == "pages":
    sys.path.append(str(current_dir.parent))
import config

# Oldal beállításai
st.set_page_config(page_title="Adatfelfedezés és Trendek", page_icon="📈", layout="wide")

# --- KÖZPONTI STÍLUSOK (CSS) ---
st.markdown("""
    <style>
    .insight-card {
        background: linear-gradient(135deg, #A81022 0%, #7a0b18 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #ff4b4b;
        box-shadow: 0 10px 20px rgba(168, 16, 34, 0.2);
        margin: 15px 0px;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .insight-title {
        display: flex; align-items: center; font-size: 1.4rem;
        font-weight: 700; margin-bottom: 10px; text-transform: uppercase;
        letter-spacing: 1px;
    }
    .insight-body { font-size: 1.1rem; line-height: 1.6; opacity: 0.95; }
    .recommendation-tag {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 5px; padding: 8px 12px; margin-top: 15px;
        font-weight: 600; display: inline-block; border: 1px solid rgba(255, 255, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SEGÉDFÜGGVÉNYEK ---
def business_insight_card(title, insight_text, recommendation):
    """Tiszta HTML generálás a kártyához"""
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">🎯 {title}</div>
        <div class="insight-body">{insight_text}</div>
        <div class="recommendation-tag">💡 JAVASLAT: {recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_parquet(config.READY_FOR_RFM_PARQUET)
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if 'Revenue' not in df.columns:
        price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
        df['Revenue'] = df['Quantity'] * df[price_col]
    return df

# --- ADATBETÖLTÉS ---
with st.spinner("Adatok betöltése..."):
    df = load_data()

st.title("📈 Adatfelfedezés és Trendek")
st.markdown("Elemezd a vásárlási szokásokat és az üzleti trendeket interaktív módon.")

# --- 1. DÁTUMVÁLASZTÓ ---
st.sidebar.header("Szűrési beállítások")
min_date, max_date = df['InvoiceDate'].min().date(), df['InvoiceDate'].max().date()
selected_dates = st.sidebar.date_input("Időszak:", value=[min_date, max_date], min_value=min_date, max_value=max_date)

if len(selected_dates) == 2:
    mask = (df['InvoiceDate'].dt.date >= selected_dates[0]) & (df['InvoiceDate'].dt.date <= selected_dates[1])
    filtered_df = df.loc[mask].copy()
else:
    filtered_df = df.copy()

st.markdown("---")

# --- 2. IDŐSOROS TRENDEK ---
st.subheader("📆 Idősoros trendek: Bevétel és Tranzakciók")
granularity = st.selectbox("Időbeli felbontás:", ["Napi", "Heti", "Havi"])
resample_rules = {"Napi": "D", "Heti": "W-MON", "Havi": "ME"}
time_df = filtered_df.set_index('InvoiceDate').resample(resample_rules[granularity]).agg(
    Bevétel=('Revenue', 'sum'), Tranzakciószám=('Invoice', 'nunique')
).reset_index()

tab_rev, tab_trans = st.tabs(["💰 Bevétel", "🛒 Tranzakciószám"])
with tab_rev:
    st.plotly_chart(px.line(time_df, x='InvoiceDate', y='Bevétel', markers=True, title=f"{granularity} bevétel"), use_container_width=True)
with tab_trans:
    st.plotly_chart(px.line(time_df, x='InvoiceDate', y='Tranzakciószám', markers=True, title=f"{granularity} tranzakció", color_discrete_sequence=['#ff7f0e']), use_container_width=True)

st.markdown("---")

# --- 3. HŐTÉRKÉP (2D & 3D) ---
st.subheader("🔥 Vásárlási intenzitás (Időbeli eloszlás)")

filtered_df['Hour'] = filtered_df['InvoiceDate'].dt.hour
filtered_df['DayOfWeek'] = filtered_df['InvoiceDate'].dt.day_name()
hu_days = ['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']
day_map = dict(zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], hu_days))
filtered_df['DayOfWeek_HU'] = filtered_df['DayOfWeek'].map(day_map)

heatmap_data = filtered_df.groupby(['DayOfWeek_HU', 'Hour'])['Invoice'].nunique().reset_index()
heatmap_pivot = heatmap_data.pivot(index='DayOfWeek_HU', columns='Hour', values='Invoice').fillna(0)
heatmap_pivot = heatmap_pivot.reindex([d for d in hu_days if d in heatmap_pivot.index])

tab_2d, tab_3d = st.tabs(["📊 2D Hőtérkép", "⛰️ 3D Csúcsmodell"])

with tab_2d:
    fig_2d = px.imshow(heatmap_pivot, color_continuous_scale="Viridis", aspect="auto")
    st.plotly_chart(fig_2d, use_container_width=True)

with tab_3d:
    # 3D Felület létrehozása
    fig_3d = go.Figure(data=[go.Surface(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Viridis',
        # Itt töröltem a manuális "db" feliratot, mert a ticksuffix automatikusan odateszi
        hovertemplate='<b>Nap:</b> %{y}<br>' +
                      '<b>Idő:</b> %{x} óra:00<br>' +
                      '<b>Tranzakciók:</b> %{z}<extra></extra>'
    )])

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Óra (0-24)', ticksuffix=' óra'),
            yaxis=dict(title='Hét napja'),
            # A ticksuffix miatt a %{z} értéke már alapból tartalmazni fogja a " db"-ot
            zaxis=dict(title='Vásárlások (db)', ticksuffix=' db'),
            aspectratio=dict(x=1.5, y=1.5, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True
    )
    st.plotly_chart(fig_3d, use_container_width=True)

#! INSIGHT KÁRTYA A HŐTÉRKÉPEK ALATT
business_insight_card(
    title="Vásárlási Csúcsidőszakok",
    insight_text="A 3D modell csúcsai alapján a forgalom kedden és csütörtökön 10:00 és 13:00 között a legintenzívebb. Ez a 'műszak közbeni' vásárlási morálra utal.",
    recommendation="A villámakciókat (Flash Sales) ezekre a csúcsokra érdemes időzíteni az azonnali konverzió érdekében."
)

st.markdown("---")

# --- 4. TOPLISTÁK ---
st.subheader("🏆 Toplisták")
metric_choice = st.selectbox("Rendezési alap:", ["Bevétel (Érték)", "Darabszám (Mennyiség)"])
metric_col = 'Revenue' if metric_choice == "Bevétel (Érték)" else 'Quantity'

col1, col2 = st.columns(2)
with col1:
    top_p = filtered_df.groupby('Description')[metric_col].sum().nlargest(10).reset_index()
    fig_p = px.bar(top_p, y='Description', x=metric_col, orientation='h', title="Top Termékek")
    fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_p, use_container_width=True)

with col2:
    if 'Country' in filtered_df.columns:
        top_c = filtered_df.groupby('Country')[metric_col].sum().nlargest(10).reset_index()
        fig_c = px.bar(top_c, y='Country', x=metric_col, orientation='h', title="Top Országok", color_discrete_sequence=['#2ca02c'])
        fig_c.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_c, use_container_width=True)