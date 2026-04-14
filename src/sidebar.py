import streamlit as st


def render_sidebar():
    st.markdown("""
        <style>
            /* ── Szövegszín egységesítés sötét háttérhez (light/dark mode) ── */
            .stApp p,
            .stApp li,
            .stApp h1, .stApp h2, .stApp h3,
            .stApp h4, .stApp h5, .stApp h6,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stWidgetLabel"] p,
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"],
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] li {
                color: #c8cfe8 !important;
            }

            /* ── Gombok – sidebar nav linkekkel megegyező stílus ── */
            .stApp .stButton > button,
            .stApp [data-testid^="stBaseButton"] {
                background: rgba(168, 16, 34, 0.35) !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 8px !important;
                color: #ffffff !important;
                transition: background 0.2s ease, border-color 0.2s ease !important;
            }
            .stApp .stButton > button:hover,
            .stApp [data-testid^="stBaseButton"]:hover {
                background: rgba(168, 16, 34, 0.65) !important;
                border-color: rgba(255, 255, 255, 0.3) !important;
            }
            .stApp .stButton > button:active,
            .stApp [data-testid^="stBaseButton"]:active {
                background: rgba(168, 16, 34, 0.85) !important;
                border-color: rgba(255, 255, 255, 0.45) !important;
            }
            .stApp .stButton > button p,
            .stApp [data-testid^="stBaseButton"] p {
                color: #ffffff !important;
            }

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
