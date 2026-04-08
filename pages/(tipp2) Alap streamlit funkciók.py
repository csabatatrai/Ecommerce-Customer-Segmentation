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
with tab1:
    st.header("Szövegformázás")
    st.markdown("A `st.markdown()` segítségével **félkövér**, *dőlt*, vagy akár [linkeket](https://streamlit.io) is beilleszthetsz.")
    st.caption("Ez egy `st.caption()` – apró, halványabb szöveg megjegyzésekhez.")
    st.code("print('Ez egy st.code() blokk, kód megjelenítéséhez.')", language="python")
    st.latex(r"E = mc^2 \quad \text{(Ez egy st.latex() blokk matematikai formulákhoz)}")
    
    st.divider() # Vízszintes vonal
    
    st.header("KPI Kártyák (Metrics)")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Bevétel", value="£12,500", delta="£1,200 (Növekedés)")
    m2.metric(label="Lemorzsolódás", value="15%", delta="-2%", delta_color="inverse") # Kisebb az jobb
    m3.metric(label="Aktív ügyfelek", value="1,024", delta="0", delta_color="off")
    
    st.divider()
    
    st.header("Adattáblák")
    df_demo = pd.DataFrame(np.random.randn(5, 3), columns=["A metrika", "B metrika", "C metrika"])
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("`st.dataframe()`")
        st.caption("Interaktív, rendezhető, kereshető táblázat. Akár színezhető is (Pandas Styler).")
        st.dataframe(df_demo.style.highlight_max(axis=0, color='lightgreen'))
        
    with col_t2:
        st.subheader("`st.data_editor()`")
        st.caption("A felhasználó által szerkeszthető táblázat (kattints egy cellába)!")
        edited_df = st.data_editor(df_demo, num_rows="dynamic")

# =====================================================================
# TAB 2: Bemeneti Elemek (Widgetek)
# =====================================================================
with tab2:
    st.header("Interaktív vezérlők")
    
    # Session state példa gombhoz
    if "counter" not in st.session_state:
        st.session_state.counter = 0
        
    w1, w2, w3 = st.columns(3)
    
    with w1:
        st.subheader("Kattintások")
        if st.button("Sima Gomb (st.button)"):
            st.session_state.counter += 1
        st.write(f"Gomb megnyomva: **{st.session_state.counter}** alkalommal.")
        
        st.download_button("Letöltés gomb (st.download_button)", data="Teszt adat", file_name="teszt.txt")
        
        st.checkbox("Jelölőnégyzet (st.checkbox)", value=True)
        st.toggle("Kapcsoló (st.toggle)")
        
    with w2:
        st.subheader("Választók")
        st.radio("Rádiógomb (st.radio)", options=["Első", "Második", "Harmadik"])
        st.selectbox("Legördülő menü (st.selectbox)", options=["Alma", "Körte", "Barack"])
        st.multiselect("Többes választó (st.multiselect)", options=["VIP", "Alvó", "Új", "Lemorzsolódó"], default=["VIP"])
        
    with w3:
        st.subheader("Szöveg és Szám")
        st.text_input("Szöveges bemenet (st.text_input)", placeholder="Írj ide valamit...")
        st.number_input("Szám bemenet (st.number_input)", min_value=0, max_value=100, value=50, step=5)
        st.slider("Csúszka (st.slider)", min_value=0.0, max_value=10.0, value=5.5)
        
    st.divider()
    
    w4, w5 = st.columns(2)
    with w4:
        st.date_input("Dátumválasztó (st.date_input)", value=datetime.date.today())
        st.time_input("Időválasztó (st.time_input)", value=datetime.datetime.now().time())
    with w5:
        st.color_picker("Színválasztó (st.color_picker)", "#2ecc71")
        st.file_uploader("Fájl feltöltő (st.file_uploader)", type=["csv", "parquet", "png"])

# =====================================================================
# TAB 3: Grafikonok
# =====================================================================
with tab3:
    st.header("Beépített és Plotly Grafikonok")
    
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Beépített Vonaldiagram (`st.line_chart`)")
        st.line_chart(chart_data)
        
        st.subheader("Beépített Területdiagram (`st.area_chart`)")
        st.area_chart(chart_data)
        
    with c2:
        st.subheader("Beépített Oszlopdiagram (`st.bar_chart`)")
        st.bar_chart(chart_data)
        
        st.subheader("Plotly Integráció (`st.plotly_chart`)")
        fig = px.scatter(chart_data, x="A", y="B", size=np.abs(chart_data["C"])*10, color="C", title="Plotly Scatter")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 4: Elrendezés (Layout)
# =====================================================================
with tab4:
    st.header("Hogyan rendezd el a tartalmat?")
    
    st.markdown("**1. Oszlopok (`st.columns`)**")
    cols = st.columns([1, 2, 1]) # Arányok: 25% - 50% - 25%
    cols[0].info("Kisebb oszlop (1)")
    cols[1].success("Kétszer olyan széles oszlop (2)")
    cols[2].warning("Kisebb oszlop (1)")
    
    st.markdown("**2. Lenyíló fülek (`st.expander`)**")
    with st.expander("Kattints ide a részletekért (st.expander)"):
        st.write("Ez a tartalom rejtve volt! Ide érdemes tenni a magyarázatokat, hosszú táblázatokat vagy a letöltés gombokat.")
        
    st.markdown("**3. Konténerek (`st.container`)**")
    with st.container(border=True):
        st.write("Ez egy keretes konténer. Összefogja a logikailag együttesen kezelendő elemeket.")
        st.button("Gomb a konténerben")
        
    st.markdown("**4. Popover (Újdonság!)**")
    with st.popover("⚙️ Beállítások megnyitása"):
        st.write("Ez egy felugró panel!")
        st.checkbox("Sötét mód")
        st.slider("Betűméret", 10, 24, 16)

# =====================================================================
# TAB 5: Állapot és Visszajelzés
# =====================================================================
with tab5:
    st.header("Értesítések és Töltés")
    
    st.subheader("Színes üzenetdobozok")
    st.info("`st.info()`: Hasznos információk, semleges.")
    st.success("`st.success()`: Sikeres művelet (pl. modell betöltve).")
    st.warning("`st.warning()`: Figyelmeztetés (pl. demo adatokat használsz).")
    st.error("`st.error()`: Hibaüzenet (pl. nem található a fájl).")
    
    st.divider()
    
    st.subheader("Dinamikus visszajelzések")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("Mutass egy Toast üzenetet"):
            st.toast("✅ Ez egy Toast üzenet, ami hamar eltűnik!", icon="🍞")
            
    with col_btn2:
        if st.button("Indíts el egy töltőképernyőt (Spinner)"):
            with st.spinner("Modell betanítása folyamatban... (3 másodperc)"):
                time.sleep(3)
            st.success("Kész!")
            
    with col_btn3:
        if st.button("Ünnepeljünk! (Balloons)"):
            st.balloons()
            # Esetleg havazás is: st.snow()
            
    st.divider()
    
    st.subheader("Progress bar (Folyamatjelző)")
    if st.button("Folyamatjelző indítása"):
        progress_text = "Adatok betöltése..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=f"Folyamatban... {percent_complete + 1}%")
        time.sleep(0.5)
        my_bar.empty()
        st.success("Adatbetöltés befejeződött!")