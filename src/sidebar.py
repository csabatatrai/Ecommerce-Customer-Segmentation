import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("## E-kereskedelmi vásárlói szegmentáció és churn-elemzés")
        st.markdown("---")
        st.markdown("**Adatforrás**")
        st.markdown(
            "🗃️ [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)"
        )
        st.markdown("---")
        st.markdown("**Módszertan**")
        st.markdown("- RFM szegmentáció\n- XGBoost churn modell")
        st.markdown("---")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Forráskód-181717?logo=github)](https://github.com/csabatatrai/Ecommerce-Customer-Segmentation/)"
            "  \n"
            "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profil-0A66C2?logo=linkedin)](https://www.linkedin.com/in/csabatatrai-datascientist/)"
        )
