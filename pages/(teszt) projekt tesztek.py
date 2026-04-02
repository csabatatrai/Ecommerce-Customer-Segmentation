import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Szülőkönyvtár hozzáadása, hogy a config.py importálható legyen
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Projekt Tesztek", page_icon="✅", layout="wide")

st.title("✅ Projekt Kimenetek és Adatminőség Tesztelése")
st.markdown("""
Ez a modul az **E-commerce Customer Segmentation** projekt futtatási eredményeit validálja. 
Minden lépésnél ellenőrzi az elvárt fájlok meglétét, az adatminőségi kritériumokat (pl. nincsenek hiányzó azonosítók, negatív értékek), és a modellek szerializációs állapotát.
""")

# --- Segédfüggvény a tesztek futtatásához és megjelenítéséhez ---
def run_test(test_name, condition, success_msg, error_msg):
    if condition:
        st.success(f"**SIKERES:** {test_name} - {success_msg}")
        return True
    else:
        st.error(f"**HIBA:** {test_name} - {error_msg}")
        return False

# =========================================================
# 1. KONFIGURÁCIÓ ÉS KÖRNYEZET
# =========================================================
st.header("1. Konfiguráció és Környezet")
try:
    from config import (
        READY_FOR_RFM_PARQUET, RFM_FEATURES_PARQUET, CUSTOMER_SEGMENTS_PARQUET,
        CHURN_PREDICTIONS_PARQUET, MODELS_DIR, PROCESSED_DIR
    )
    st.success("**SIKERES:** `config.py` importálása sikeres.")
    config_ok = True
except Exception as e:
    st.error(f"**HIBA:** A `config.py` nem található vagy hibás. {e}")
    config_ok = False

if config_ok:
    # =========================================================
    # 2. ADATELŐKÉSZÍTÉS (01_data_preparation)
    # =========================================================
    st.header("2. Adatelőkészítés (01-es notebook)")
    
    file_01_ok = run_test(
        "Tisztított adatok fájl", 
        os.path.exists(READY_FOR_RFM_PARQUET),
        "A `ready_for_rfm.parquet` fájl létezik.",
        "A `ready_for_rfm.parquet` fájl nem található!"
    )
    
    if file_01_ok:
        try:
            df_01 = pd.read_parquet(READY_FOR_RFM_PARQUET)
            st.write(f"*Betöltött sorok száma: {len(df_01):,}*")
            
            # Teszt: Hiányzó ügyfél ID-k
            run_test("Ügyfél ID-k integritása", 
                     df_01['Customer ID'].isnull().sum() == 0,
                     "Nincsenek hiányzó (Null) Customer ID-k az adathalmazban.",
                     f"{df_01['Customer ID'].isnull().sum()} hiányzó Customer ID maradt az adatokban!")
            
            # Teszt: Ár és mennyiség
            run_test("Érvényes árak", 
                     (df_01['Price'] <= 0).sum() == 0,
                     "Minden termék ára pozitív (Price > 0).",
                     "Léteznek 0 vagy negatív árak az adatokban!")
            
            run_test("Érvényes mennyiségek", 
                     (df_01['Quantity'] <= 0).sum() == 0,
                     "Minden vásárolt mennyiség pozitív (Nincsenek visszáruk/technikai sorok).",
                     "Negatív vagy nulla mennyiség található (visszáru nem lett kiszűrve)!")
                     
        except Exception as e:
            st.error(f"**HIBA:** A 01-es adatfájl olvasása közben hiba történt: {e}")

    # =========================================================
    # 3. SZEGMENTÁCIÓ ÉS RFM (02_customer_segmentation)
    # =========================================================
    st.header("3. Vásárlói Szegmentáció (02-es notebook)")
    
    file_02_ok = run_test(
        "Szegmentációs fájl", 
        os.path.exists(CUSTOMER_SEGMENTS_PARQUET) or os.path.exists(RFM_FEATURES_PARQUET),
        "A szegmentációs kimeneti fájlok megvannak.",
        "Hiányzik az rfm_features.parquet vagy a customer_segments.parquet!"
    )
    
    if file_02_ok:
        try:
            # Megpróbáljuk a végleges szegmens fájlt, ha nincs, akkor a nyers RFM-et
            target_file = CUSTOMER_SEGMENTS_PARQUET if os.path.exists(CUSTOMER_SEGMENTS_PARQUET) else RFM_FEATURES_PARQUET
            df_02 = pd.read_parquet(target_file)
            
            # Oszlopok ellenőrzése (Kis- és nagybetűkre is érzéketlenül próbáljuk)
            cols_lower = [c.lower() for c in df_02.columns]
            
            run_test("RFM Oszlopok", 
                     any('recency' in c for c in cols_lower) and any('frequency' in c for c in cols_lower) and any('monetary' in c for c in cols_lower),
                     "A Recency, Frequency és Monetary oszlopok jelen vannak.",
                     "Hiányoznak az alapvető RFM metrikák!")
            
            if 'cluster' in cols_lower:
                st.success("**SIKERES:** Klaszter/Szegmens címkék megtalálhatóak az adatokban.")
            else:
                st.warning("**FIGYELMEZTETÉS:** A betöltött fájlban nem található 'Cluster' oszlop. Biztosan lefutott a K-Means?")
                
        except Exception as e:
            st.error(f"Hiba az RFM adatok olvasásakor: {e}")
            
    # Modellek ellenőrzése
    st.subheader("Modell Artefaktumok (02)")
    scaler_path = os.path.join(MODELS_DIR, "scaler_rfm.joblib")
    kmeans_path = os.path.join(MODELS_DIR, "kmeans_rfm.joblib")
    
    run_test("Scaler modell", os.path.exists(scaler_path), "scaler_rfm.joblib elérhető.", "scaler_rfm.joblib hiányzik!")
    run_test("K-Means modell", os.path.exists(kmeans_path), "kmeans_rfm.joblib elérhető.", "kmeans_rfm.joblib hiányzik!")

    # =========================================================
    # 4. CHURN PREDIKCIÓ (03_churn_prediction)
    # =========================================================
    st.header("4. Churn Predikció (03-as notebook)")
    
    file_03_ok = run_test(
        "Predikciós kimenet", 
        os.path.exists(CHURN_PREDICTIONS_PARQUET),
        "A `churn_predictions.parquet` megtalálható.",
        "A `churn_predictions.parquet` fájl hiányzik!"
    )
    
    if file_03_ok:
        df_03 = pd.read_parquet(CHURN_PREDICTIONS_PARQUET)
        
        # Általában elvárunk egy valószínűségi oszlopot
        prob_cols = [c for c in df_03.columns if 'prob' in c.lower() or 'pred' in c.lower() or 'churn' in c.lower()]
        run_test("Predikciós oszlopok", 
                 len(prob_cols) > 0,
                 f"Predikciós oszlopok felismerve: {', '.join(prob_cols)}",
                 "Nem található valószínűségre vagy predikcióra utaló oszlop a kimenetben!")
                 
    # Modell betölthetőség tesztelése (XGBoost Pipeline)
    st.subheader("Modell Artefaktumok (03)")
    # A config.py a 03-as kimenetet régebben XGB_MODEL_PATH-ként hivatkozta (lehet xgboost_clv.joblib vagy xgboost_churn.joblib)
    # Megkeressük a mappában lévő xgboost modellt:
    xgboost_models = list(Path(MODELS_DIR).glob("xgboost*.joblib"))
    
    if len(xgboost_models) > 0:
        xgb_path = xgboost_models[0]
        st.success(f"**SIKERES:** Megtalált XGBoost modell: {xgb_path.name}")
        
        try:
            import joblib
            model = joblib.load(xgb_path)
            
            # Ellenőrizzük, hogy rendelkezik-e predict_proba vagy predict metódussal
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                st.success("**SIKERES:** A modell sikeresen betöltve a memóriába, és rendelkezik predikciós képességgel (API kompatibilis).")
            else:
                st.error("**HIBA:** A betöltött objektumnak nincs `predict` metódusa!")
        except Exception as e:
            st.error(f"**HIBA:** A modell megtalálható, de nem tölthető be! (Lehet hiányzó csomag pl. xgboost, scikit-learn verzióeltérés). Részletek: {e}")
    else:
        st.error("**HIBA:** Nem található xgboost_*.joblib a models/ mappában!")

st.divider()
st.info("💡 A fenti tesztek minden alkalommal lefutnak, amikor megnyitod ezt az oldalt, így azonnal láthatod, ha egy új kódmódosítás elrontotta az adatok integritását.")