"""
pages/99_🛠️_Streamlit_Showcase.py
Egy átfogó puskázó oldal a Streamlit képességeinek teszteléséhez.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import time

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Streamlit Kézikönyv", page_icon="🛠️", layout="wide")

st.title("🛠️ Streamlit Képességek és Elemek")
st.markdown("Ezen az oldalon végigkattintgathatod a Streamlit legfontosabb funkcióit. Szemezgess belőle bátran a projektjeidhez!")

# Tabok (fülek) használata a rendszerezéshez
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Szöveg és Adat", 
    "🎛️ Bemeneti Elemek (Widgetek)", 
    "📊 Grafikonok", 
    "🏗️ Elrendezés (Layout)", 
    "🔔 Állapot és Visszajelzés"
])

# =====================================================================
# TAB 1: Szöveg és Adat megjelenítés
# =====================================================================

with st.container(border=True):
    st.subheader("💡 Üzleti Insight")
    st.write("A negyedéves eladások 15%-kal nőttek a kampány hatására.")
    st.caption("Forrás: Értékesítési adatbázis, 2024 Q3")