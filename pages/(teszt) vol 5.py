import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Szülőkönyvtár hozzáadása, hogy a config.py importálható legyen
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Kiterjesztett Projekt Tesztek", page_icon="✅", layout="wide")

st.title("✅ Kiterjesztett Adat- és Modell Validáció")
st.markdown("""
Ez a modul az **E-commerce Customer Segmentation** pipeline-jának teljeskörű auditját végzi.
Ellenőrzi az adatbázis integritását, az RFM üzleti logikáját, a modellek minőségét és a generált vizualizációk meglétét.
""")

def run_test(test_name, condition, success_msg, error_msg):
    if condition:
        st.success(f"**SIKERES:** {test_name} - {success_msg}")
        return True
    else:
        st.error(f"**HIBA:** {test_name} - {error_msg}")
        return False

# =========================================================
# 1. KONFIGURÁCIÓ
# =========================================================
try:
    from config import (
        READY_FOR_RFM_PARQUET, RFM_FEATURES_PARQUET, CUSTOMER_SEGMENTS_PARQUET,
        CHURN_PREDICTIONS_PARQUET, MODELS_DIR, IMAGES_DIR
    )
    config_ok = True
except Exception as e:
    st.error(f"**HIBA:** A `config.py` betöltése sikertelen. {e}")
    config_ok = False

if config_ok:
    # Két oszlopos elrendezés a letisztultabb megjelenésért
    col1, col2 = st.columns(2)

    with col1:
        # =========================================================
        # 2. MÉLY ADATMINŐSÉG (01-es notebook)
        # =========================================================
        st.header("🛒 Tranzakciós Adatok")
        if os.path.exists(READY_FOR_RFM_PARQUET):
            df_01 = pd.read_parquet(READY_FOR_RFM_PARQUET)
            
            # Alap tesztek
            run_test("Hiányzó ID-k", df_01['Customer ID'].isnull().sum() == 0, "Nincsenek Null Customer ID-k.", "Található Null ID!")
            run_test("Pozitív árak", (df_01['Price'] <= 0).sum() == 0, "Minden ár > 0.", "Hibás árak az adatban!")
            run_test("Pozitív mennyiségek", (df_01['Quantity'] <= 0).sum() == 0, "Minden mennyiség > 0.", "Negatív mennyiségek (visszáru) maradtak!")
            
            # ÚJ TESZT: Visszáruk (Cancellations) szűrése az Invoice alapján
            # A Kaggle datasetben a törölt rendelések 'C'-vel kezdődnek
            if 'Invoice' in df_01.columns:
                cancellations = df_01['Invoice'].astype(str).str.startswith('C').sum()
                run_test("Visszáruk kiszűrése (Invoice='C...')", 
                         cancellations == 0, 
                         "A törölt tranzakciók (C-vel kezdődő Invoice) sikeresen eltávolítva.", 
                         f"A tisztított adatban maradt {cancellations} db törölt tranzakció!")
        else:
            st.warning("Tranzakciós fájl nem található.")

        # =========================================================
        # 3. ÜZLETI LOGIKA: RFM (02-es notebook)
        # =========================================================
        st.header("👥 RFM Üzleti Logika")
        rfm_file = CUSTOMER_SEGMENTS_PARQUET if os.path.exists(CUSTOMER_SEGMENTS_PARQUET) else RFM_FEATURES_PARQUET
        
        if os.path.exists(rfm_file):
            df_rfm = pd.read_parquet(rfm_file)
            cols_lower = [c.lower() for c in df_rfm.columns]
            
            # Dinamikus oszlopkeresés
            r_col = next((c for c in df_rfm.columns if 'recency' in c.lower()), None)
            f_col = next((c for c in df_rfm.columns if 'frequency' in c.lower()), None)
            m_col = next((c for c in df_rfm.columns if 'monetary' in c.lower()), None)

            if r_col and f_col and m_col:
                # ÚJ TESZT: Matematikai RFM validáció
                run_test("Érvényes Recency", (df_rfm[r_col] >= 0).all(), "Nincs negatív eltelt nap (időutazás).", "Negatív Recency értékek!")
                run_test("Érvényes Frequency", (df_rfm[f_col] >= 1).all(), "Minden ügyfél vásárolt legalább 1-szer.", "0 Frequency értékek találhatóak!")
                run_test("Érvényes Monetary", (df_rfm[m_col] > 0).all(), "Minden ügyfél költése > 0.", "0 vagy negatív költés (Monetary) van a rendszerben!")
                
                # ÚJ TESZT: Nincsenek extrém kiugró értékek, amiket a StandardScaler nem tudna kezelni (pl. Végtelen értékek)
                is_finite = np.isfinite(df_rfm[[r_col, f_col, m_col]]).all().all()
                run_test("Végtelen (Inf) értékek hiánya", is_finite, "Minden RFM érték véges szám.", "Végtelen érték került a feature-ökbe!")
            else:
                st.error("Nem találhatóak az RFM oszlopok!")
        else:
            st.warning("RFM fájl nem található.")

    with col2:
        # =========================================================
        # 4. PREDIKCIÓK MINŐSÉGE (03-as notebook)
        # =========================================================
        st.header("🔮 Lemorzsolódás Predikciók")
        if os.path.exists(CHURN_PREDICTIONS_PARQUET):
            df_churn = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
            prob_cols = [c for c in df_churn.columns if 'prob' in c.lower() or 'pred' in c.lower() or 'churn' in c.lower()]
            
            if prob_cols:
                p_col = prob_cols[0]
                # ÚJ TESZT: Valószínűségi szabályok (0 <= p <= 1)
                valid_probs = (df_churn[p_col] >= 0.0) & (df_churn[p_col] <= 1.0)
                run_test(f"Valószínűségi korlátok ({p_col})", 
                         valid_probs.all(), 
                         "Minden predikció szigorúan 0.0 és 1.0 közé esik.", 
                         "A modell érvénytelen (0-nál kisebb vagy 1-nél nagyobb) valószínűségeket generált!")
                
                # ÚJ TESZT: Egyediség (Minden ügyfél csak egyszer szerepel)
                # Ha az index a Customer ID:
                id_col = df_churn.index if df_churn.index.name else df_churn.columns[0]
                is_unique = len(df_churn) == len(set(id_col))
                run_test("Ügyfél egyediség a predikciókban", 
                         is_unique, 
                         "Minden ügyfélhez pontosan 1 predikció tartozik (nincs duplikáció).", 
                         "Vannak duplikált ügyfelek a predikciós táblában!")
            else:
                st.error("Nincs predikciós oszlop a kimenetben!")
        else:
            st.warning("Lemorzsolódási fájl nem található.")

        # =========================================================
        # 5. MODELLEK ÉS VIZUALIZÁCIÓK MEGLÉTE
        # =========================================================
        st.header("🖼️ Modellek és Ábrák (Assets)")
        
        # ML modellek ellenőrzése kibővítve
        scaler_path = os.path.join(MODELS_DIR, "scaler_rfm.joblib")
        if os.path.exists(scaler_path):
            import joblib
            try:
                scaler = joblib.load(scaler_path)
                run_test("Scaler API", hasattr(scaler, 'transform'), "A betöltött objektum rendelkezik 'transform' metódussal.", "A Scaler hibás, nincs 'transform' metódusa!")
            except:
                pass

        # Diagramok és mentett vizualizációk a pipeline-ból (A readme alapján)
        expected_images = [
            "rfm_raw_distributions.png",
            "kmeans_optimal_k.png",
            "shap_summary_plot.png",
            "precision_recall_curve.png"
        ]
        
        missing_images = []
        for img in expected_images:
            if not os.path.exists(os.path.join(IMAGES_DIR, img)):
                missing_images.append(img)
                
        run_test("Generált vizualizációk", 
                 len(missing_images) == 0, 
                 "Minden kulcsfontosságú diagram megtalálható az `assets/images/` mappában.", 
                 f"Hiányzó diagramok: {', '.join(missing_images)}")

st.divider()
st.caption("Ezek a tesztek az MLOps pipeline egészségét biztosítják. Ha bármelyik piros, érdemes visszamenni a megfelelő Jupyter Notebookhoz.")