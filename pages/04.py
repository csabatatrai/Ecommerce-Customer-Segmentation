import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# 1. Kényszerített ellenőrzés: Melyik Python futtatja a Streamlit-et?
# Ez segít neked látni a UI-on, ha még mindig nem a Conda környezetben vagy.
python_path = sys.executable

try:
    import xgboost
    XGB_VERSION = xgboost.__version__
except ImportError:
    st.error(f"🚨 Az 'xgboost' nem található ebben a környezetben: {python_path}")
    st.info("Próbáld meg: `python -m pip install xgboost` a terminálban.")
    st.stop()

# Projekt szintű konfiguráció
try:
    from config import MODELS_DIR, PROCESSED_DIR
    MODEL_PATH = MODELS_DIR / "xgboost_churn.joblib"
    DATA_PATH = PROCESSED_DIR / "churn_predictions.parquet"
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_churn.joblib"
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_predictions.parquet"

st.set_page_config(page_title="Churn Predikció", page_icon="🔴", layout="wide")

# Debug infó (csak hiba esetén hasznos, később kivehető)
# st.sidebar.write(f"Python: {python_path}")
# st.sidebar.write(f"XGBoost verzió: {XGB_VERSION}")

st.title("🔴 Churn (Lemorzsolódás) Predikció és Szimulátor")

# ==========================================
# 1. Adat és Modell betöltése
# ==========================================

@st.cache_data
def load_predictions():
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    return pd.DataFrame()

@st.cache_resource
def load_xgb_model():
    """Betölti a modellt, hibakezeléssel."""
    if not MODEL_PATH.exists():
        st.error(f"Hiányzó modellfájl: {MODEL_PATH}")
        return None
    try:
        # Néha a joblib-nek explicit import kell az unpickling-hez
        import xgboost as xgb
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Modell betöltési hiba: {e}")
        st.info("Tipp: Futtasd újra a 03_churn_prediction.ipynb notebookot a modell frissítéséhez!")
        return None

df = load_predictions()
model = load_xgb_model()

if df.empty or model is None:
    st.warning("Adatok vagy modell nem tölthető be. Ellenőrizd a fájlokat!")
    st.stop()

# ==========================================
# 2. Kockázatos Vásárlók
# ==========================================
st.header("🚨 Kockázatos VIP Vásárlók")

# Churn esély meghatározása (ha nincs fix oszlop)
if 'churn_proba' not in df.columns:
    st.error("A betöltött adatokban nem található 'churn_proba' oszlop!")
    st.stop()

# VIP szűrés (Top 25% költés és > 50% churn)
monetary_threshold = df['monetary_total'].quantile(0.75)
vip_at_risk = df[(df['monetary_total'] >= monetary_threshold) & (df['churn_proba'] > 0.5)].copy()
vip_at_risk = vip_at_risk.sort_values(by='churn_proba', ascending=False)

display_cols = ['monetary_total', 'frequency', 'recency_days', 'churn_proba']
if 'rfm_segment' in vip_at_risk.columns:
    display_cols.insert(0, 'rfm_segment')

st.dataframe(
    vip_at_risk[display_cols].style
    .format({'monetary_total': '{:,.0f} £', 'churn_proba': '{:.2%}', 'recency_days': '{:.0f} nap'})
    .background_gradient(subset=['churn_proba'], cmap='Reds')
    .background_gradient(subset=['monetary_total'], cmap='Greens'),
    use_container_width=True
)

st.divider()

# ==========================================
# 3. Változók Fontossága
# ==========================================
st.header("📊 Befolyásoló tényezők")

try:
    # Megpróbáljuk kinyerni a feature fontosságot (Pipeline vagy alap modell)
    if hasattr(model, 'named_steps'):
        xgb_model = model.named_steps['clf']
        # Ha van ColumnTransformer/Selector, a nevek kinyerése trükkös lehet
        try:
            feature_names = model.feature_names_in_
        except:
            feature_names = [f"F{i}" for i in range(xgb_model.n_features_in_)]
    else:
        xgb_model = model
        feature_names = xgb_model.get_booster().feature_names if hasattr(xgb_model, 'get_booster') else []

    fi_df = pd.DataFrame({'Jellemző': feature_names, 'Fontosság': xgb_model.feature_importances_})
    fi_df = fi_df.sort_values('Fontosság', ascending=True)

    fig_fi = px.bar(fi_df, x='Fontosság', y='Jellemző', orientation='h', color='Fontosság', color_continuous_scale='Blues')
    st.plotly_chart(fig_fi, use_container_width=True)
except Exception as e:
    st.info(f"Feature Importance nem érhető el: {e}")

st.divider()

# ==========================================
# 4. "What-If" Szimulátor
# ==========================================
st.header("🎛️ 'What-If' Szimulátor")

c1, c2 = st.columns([1, 1])

with c1:
    r = st.slider("Recency (nap)", 0, int(df.recency_days.max()), 30)
    f = st.slider("Frequency", 1, int(df.frequency.max()), 5)
    m = st.number_input("Monetary (Összes költés)", value=int(df.monetary_total.median()))
    ret = st.slider("Visszaküldési arány (%)", 0, 100, 5) / 100

with c2:
    # Bemeneti adat előkészítése pontosan úgy, ahogy a modell várja
    input_df = pd.DataFrame({
        'recency_days': [r],
        'frequency': [f],
        'monetary_total': [m],
        'monetary_avg': [m/f if f > 0 else 0],
        'return_ratio': [ret]
    })
    
    try:
        prob = model.predict_proba(input_df)[0][1]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            title={'text': "Churn Valószínűség"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        if prob > 0.5:
            st.error("Kritikus kockázat!")
        else:
            st.success("Biztonságos ügyfél.")
    except Exception as e:
        st.error(f"Predikciós hiba: {e}")
        st.info("Ellenőrizd, hogy a notebookban használt oszlopnevek egyeznek-e a szimulátoréval!")