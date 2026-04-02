# pages/2_🔮_Lemorzsolodas_es_CLV.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import CHURN_PREDICTIONS_PARQUET, CLV_PREDICTIONS_PARQUET

st.set_page_config(page_title="Lemorzsolódás és CLV", page_icon="🔮", layout="wide")

st.title("🔮 Lemorzsolódási Kockázat és Ügyfélérték (CLV)")
st.markdown("""
Ezen az oldalon látható, hogy mely ügyfeleknél a legmagasabb a lemorzsolódás (**Churn**) valószínűsége, 
illetve mekkora az ő előrejelzett élettartam értékük (**CLV**) a következő hónapokban.
Ezek az adatok kritikusak a célzott marketing- és megtartási (retention) kampányokhoz.
""")

@st.cache_data
def load_predictive_data():
    df_churn = None
    df_clv = None
    
    try:
        df_churn = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
    except FileNotFoundError:
        pass
        
    try:
        df_clv = pd.read_parquet(CLV_PREDICTIONS_PARQUET)
    except FileNotFoundError:
        pass
        
    return df_churn, df_clv

df_churn, df_clv = load_predictive_data()

# Ha mindkettő megvan, érdemes összefűzni őket az Ügyfél ID (Customer ID) alapján
if df_churn is not None and df_clv is not None:
    # Feltételezve, hogy az index a Customer ID, vagy van egy közös oszlop
    df_combined = df_churn.join(df_clv, how='inner')
    
    st.subheader("A 'Veszélyeztetett Érték' (At-Risk Value) mátrix")
    st.markdown("A jobb felső negyedben találhatók a legfontosabb ügyfelek: **Magas értékűek, de nagy a kockázata az elvesztésüknek.**")
    
    # Kockázat vs CLV scatter plot
    churn_prob_col = 'Churn_Probability' if 'Churn_Probability' in df_combined.columns else df_combined.columns[1]
    clv_col = 'CLV' if 'CLV' in df_combined.columns else df_combined.columns[-1]

    fig_scatter = px.scatter(df_combined, x=churn_prob_col, y=clv_col, 
                             color=churn_prob_col, color_continuous_scale='RdYlGn_r',
                             labels={churn_prob_col: "Lemorzsolódási Valószínűség (%)", clv_col: "Várható CLV (Következő 12 hó)"},
                             hover_data=df_combined.columns)
    
    # Veszélyességi vonalak húzása (pl. 50% churn felett és medián CLV felett)
    median_clv = df_combined[clv_col].median()
    fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Magas Kockázat")
    fig_scatter.add_hline(y=median_clv, line_dash="dash", line_color="blue", annotation_text="Átlag feletti érték")
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Top cselekvési lista
    st.subheader("Azonnali cselekvést igénylő VIP ügyfelek")
    st.markdown("Azon ügyfelek listája, akiknek a CLV értéke a felső 20%-ban van, de a lemorzsolódási valószínűségük > 50%.")
    
    top_20_clv_threshold = df_combined[clv_col].quantile(0.8)
    action_list = df_combined[(df_combined[clv_col] >= top_20_clv_threshold) & (df_combined[churn_prob_col] > 0.5)]
    action_list = action_list.sort_values(by=clv_col, ascending=False)
    
    st.dataframe(action_list, use_container_width=True)

elif df_churn is not None:
    st.success("Lemorzsolódási adatok betöltve, de a CLV predikciók hiányoznak.")
    st.subheader("Lemorzsolódási valószínűségek eloszlása")
    churn_prob_col = 'Churn_Probability' if 'Churn_Probability' in df_churn.columns else df_churn.columns[1]
    fig_hist = px.histogram(df_churn, x=churn_prob_col, nbins=50, title="Lemorzsolódási valószínűség eloszlása")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.dataframe(df_churn.sort_values(by=churn_prob_col, ascending=False).head(100))
    
else:
    st.warning("Nem találhatóak a predikciós `.parquet` fájlok. Futtasd le a 03-as és 04-es notebookokat!")