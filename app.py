"""
Tesztoldal Streamlithez
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import time

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Streamlit Kézikönyv", page_icon="🛠️", layout="wide")

# Oldal beállításai
st.set_page_config(page_title="Adatfelfedezés és Trendek", page_icon="📈", layout="wide")

#! kell oldal elejére a piros kártyához
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
#! idáig kell dashboard elejére a piros kártyához

st.title("Ez lesz a dashboardom főoldala")
st.markdown("Jó böngészést!")

# Tabok (fülek) használata a rendszerezéshez
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Menü 1", 
    "Menü 2", 
    "Menü 3", 
    "Menü 4", 
    "Menü 5"
])

# =====================================================================
# TAB 1: Szöveg és Adat megjelenítés
# =====================================================================

with st.container(border=True):
    st.subheader("💡 Üzleti Insight")
    st.write("Példa: A negyedéves eladások 15%-kal nőttek a kampány hatására.")
    st.caption("Példa: Forrás: Értékesítési adatbázis, 2024 Q3")

#! PIROS INSIGHT KÁRTYA
business_insight_card(
    title="Vásárlási Csúcsidőszakok",
    insight_text="A 3D modell csúcsai alapján a forgalom kedden és csütörtökön 10:00 és 13:00 között a legintenzívebb. Ez a 'műszak közbeni' vásárlási morálra utal.",
    recommendation="A villámakciókat (Flash Sales) ezekre a csúcsokra érdemes időzíteni az azonnali konverzió érdekében."
)
#! eddig