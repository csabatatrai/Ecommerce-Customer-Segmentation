import streamlit as st


def render_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #000000;
            }
            [data-testid="stHeader"] {
                background-color: #000000;
            }
            [data-testid="stSidebarNav"] a {
                font-size: 1.1rem;
                padding: 0.65rem 1rem;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                background: rgba(168, 16, 34, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                margin-bottom: 0.4rem;
                color: #ffffff !important;
                transition: background 0.2s ease;
            }
            [data-testid="stSidebarNav"] a:hover {
                background: rgba(168, 16, 34, 0.65);
                border-color: rgba(255, 255, 255, 0.3);
            }
            [data-testid="stSidebarNav"] a[aria-current="page"] {
                background: rgba(168, 16, 34, 0.85);
                border-color: rgba(255, 255, 255, 0.45);
                font-weight: 700;
            }
            [data-testid="stSidebarNav"] a span {
                font-size: 1.1rem;
                text-align: center;
                color: #ffffff !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## E-kereskedelmi vásárlói szegmentáció és churn-elemzés")
        st.markdown("---")
        st.markdown("**Adatforrás**")
        st.markdown(
            "🗃️ [UCI Online Retail II](https://archive-beta.ics.uci.edu/dataset/502/online+retail+ii)"
        )
        st.markdown("---")
        st.markdown("**Módszertan**")
        st.markdown("- RFM szegmentáció\n- XGBoost churn modell")
        st.markdown("---")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Forráskód-A81022?logo=github&style=for-the-badge)](https://github.com/csabatatrai/Ecommerce-Customer-Segmentation/)"
            "  \n\n"
            "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profil-0A66C2?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/csabatatrai-datascientist/)"
        )
