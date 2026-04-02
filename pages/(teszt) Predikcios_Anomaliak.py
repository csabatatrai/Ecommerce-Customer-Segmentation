import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

st.set_page_config(page_title="Prediction Anomalies", layout="wide")
st.title("📈 Churn és CLV Predikciós Anomáliák")

@st.cache_data
def load_predictions():
    churn_df = None
    clv_df = None
    if config.CHURN_PREDICTIONS_PARQUET.exists():
        churn_df = pd.read_parquet(config.CHURN_PREDICTIONS_PARQUET)
    if config.CLV_PREDICTIONS_PARQUET.exists():
        clv_df = pd.read_parquet(config.CLV_PREDICTIONS_PARQUET)
    return churn_df, clv_df

churn_df, clv_df = load_predictions()

st.header("1. CLV Modell Észszerűség (Sanity Check)")
if clv_df is not None:
    st.write(f"Várt CLV predikciós horizont: **{config.CLV_HORIZON_MONTHS} hónap**")
    st.write("Keresünk olyan ügyfeleket, akiknek **alacsony az eddigi Monetary értéke**, de a modell **irreálisan magas CLV-t** jósol nekik. Ez tipikus BG/NBD túltanulási vagy paraméter-illesztési hiba.")
    
    # Keresztdiagram: Eddigi bevétel vs Jósolt jövőbeli érték
    if 'Monetary' in clv_df.columns and 'Predicted_CLV' in clv_df.columns:
        fig_clv = px.scatter(clv_df, x='Monetary', y='Predicted_CLV', 
                             color='Frequency' if 'Frequency' in clv_df.columns else None,
                             hover_data=clv_df.columns,
                             title="Történeti Vásárlás vs Jósolt CLV (Outlier vadászat)")
        st.plotly_chart(fig_clv, use_container_width=True)
        
        # Matematikai inkonzisztencia keresése
        anomalies = clv_df[(clv_df['Monetary'] < 50) & (clv_df['Predicted_CLV'] > 5000)]
        if len(anomalies) > 0:
            st.error(f"🚨 ANOMÁLIA FIGYELMEZTETÉS: {len(anomalies)} ügyfél eddig alig költött (<50), de a modell óriási CLV-t (>5000) jósol nekik!")
            st.dataframe(anomalies)
        else:
            st.success("✅ A CLV predikciók arányosnak tűnnek a történeti költésekkel.")
else:
    st.warning("Nem található a CLV_PREDICTIONS_PARQUET fájl.")

st.divider()

st.header("2. Churn Valószínűségek Eloszlása")
if churn_df is not None:
    st.write("Ha a modell csak 0-hoz és 1-hez közeli értékeket jósol (U-alak), vagy egyetlen tömbben van az összes predikció, a modell rosszul kalibrált vagy overfit.")
    
    prob_col = [col for col in churn_df.columns if 'prob' in col.lower() or 'pred' in col.lower()]
    if prob_col:
        p_col = prob_col[0]
        fig_churn = px.histogram(churn_df, x=p_col, nbins=50, 
                                 title=f"Lemorzsolódási (Churn) Valószínűségek eloszlása ({p_col})")
        st.plotly_chart(fig_churn, use_container_width=True)
        
        # Ellenőrzés gyanús bizonyosságra
        extreme_certainty = len(churn_df[(churn_df[p_col] < 0.01) | (churn_df[p_col] > 0.99)])
        ratio = extreme_certainty / len(churn_df)
        if ratio > 0.5:
            st.warning(f"⚠️ A predikciók {ratio*100:.1f}%-a extrém bizonyosságú (<1% vagy >99%). Ez potenciális Data Leakage (Target Leakage) jele a XGBoost-nál!")
else:
    st.warning("Nem található a CHURN_PREDICTIONS_PARQUET fájl.")