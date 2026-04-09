import streamlit as st

st.set_page_config(
    page_title="Vezetői összefoglaló | Ügyfélmegtartás",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/executive_summary.py", title="Vezetői összefoglaló"),
    st.Page("pages/marketing_segments.py", title="Kampánytervező"),
    st.Page("pages/customer_search.py", title="Ügyfélkereső"),
])
pg.run()
