"""
E-commerce Customer Analytics – Streamlit Multi-Page App
Belépési pont: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 E-Commerce Customer Analytics Dashboard")
st.markdown("---")

st.markdown("""
### Üdvözöljük az ügyfél-elemzési dashboardon!

Ez az alkalmazás az **Online Retail II** adathalmazon alapuló, három-fázisú gépi tanulási
pipeline üzleti eredményeit mutatja be interaktív vizualizációk és szűrők segítségével.

Az alábbi oldalak érhetők el a bal oldali menüből:

| Oldal | Tartalom |
|---|---|
| 📊 **RFM Szegmentáció** | K-Means alapú ügyfélszegmentáció, szegmensprofil és snake-plot |
| 🔴 **Churn Előrejelzés** | XGBoost churn-modell eredményei, akciótérkép, VIP-lista |
| 🔍 **Ügyfél Kereső** | Egyedi ügyfél RFM-profil és churn-kockázat keresője |

---

**Adatforrás:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) –  
~1 millió tranzakció, 2009–2011, UK alapú webáruház  
**Cutoff dátum:** 2011-09-09 (90 napos célablak)
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 **RFM Szegmentáció**\n\nK-Means (K=4) klaszterezés RFM feature-ök alapján")
with col2:
    st.warning("🔴 **Churn Előrejelzés**\n\nXGBoost modell – 55.7%-os churn arány kezelése")
with col3:
    st.success("🔍 **Ügyfél Kereső**\n\nEgyéni ügyfélprofil és testreszabott akciójavaslat")
