import streamlit as st

st.set_page_config(
    page_title="Vezetői összefoglaló | Ügyfélmegtartás",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/vezeto_osszefoglalo.py", title="Vezetői összefoglaló", icon="📊"),
    st.Page("pages/Ügyfélkereső.py", title="Ügyfélkereső", icon="🔍"),
])
pg.run()
