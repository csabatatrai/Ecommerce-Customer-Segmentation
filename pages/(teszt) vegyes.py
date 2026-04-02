import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Projekt gyökérkönyvtárának hozzáadása a path-hoz a config eléréséhez
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Projekt Tesztek", page_icon="✅", layout="wide")

st.title("✅ Projekt Kimenetek Ellenőrzése")
st.markdown("Ez az oldal automatikusan teszteli a **01, 02 és 03-as Jupyter notebookok** által generált adatfájlok és modellek meglétét, valamint az adatok minőségét.")

# ---------------------------------------------------------
# Helper függvények a teszteléshez
# ---------------------------------------------------------
def run_test(test_name, condition, success_msg, error_msg):
    if condition:
        st.success(f"**SIKERES:** {test_name} - {success_msg}")
        return True
    else:
        st.error(f"**HIBA:** {test_name} - {error_msg}")
        return False

def check_file_exists(filepath, name):
    return run_test(
        f"Fájl létezés: {name}", 
        os.path.exists(filepath),
        f"A fájl megtalálható ({filepath})",
        f"A fájl hiányzik! ({filepath})"
    )

# ---------------------------------------------------------
# Config és Path beállítások tesztelése
# ---------------------------------------------------------
st.header("1. Konfiguráció és Fájlrendszer")
try:
    from config import (
        READY_FOR_RFM_PARQUET, RFM_FEATURES_PARQUET,
        MODELS_DIR, CUTOFF_DATE
    )
    st.success("**SIKERES:** `config.py` betöltése.")
    config_ok = True
except Exception as e:
    st.error(f"**HIBA:** Nem sikerült betölteni a `config.py`-t. Részletek: {e}")
    config_ok = False

if config_ok:
    # ---------------------------------------------------------
    # 01_data_preparation.ipynb tesztek
    # ---------------------------------------------------------
    st.header("2. Adatelőkészítés (01-es notebook)")
    
    file_01_ok = check_file_exists(READY_FOR_RFM_PARQUET, "ready_for_rfm.parquet")
    
    if file_01_ok:
        df_01 = pd.read_parquet(READY_FOR_RFM_PARQUET)
        st.write(f"Sorok száma: {len(df_01):,}")
        
        # Teszt 2.1: Hiányzó Customer ID-k
        run_test("Anonim vásárlók szűrése", 
                 df_01['Customer ID'].isnull().sum() == 0,
                 "Nincsenek hiányzó Customer ID-k.",
                 "Találhatóak hiányzó Customer ID-k!")
        
        # Teszt 2.2: Negatív vagy nulla árak szűrése
        run_test("Érvénytelen árak szűrése", 
                 (df_01['Price'] <= 0).sum() == 0,
                 "Minden Price > 0.",
                 "Létezik 0 vagy negatív ár az adatok között!")
        
        # Teszt 2.3: Datetime típus
        run_test("Dátum formátum", 
                 pd.api.types.is_datetime64_any_dtype(df_01['InvoiceDate']),
                 "Az InvoiceDate oszlop megfelelő datetime formátumú.",
                 "Az InvoiceDate nem datetime típusú!")

    # ---------------------------------------------------------
    # 02_customer_segmentation.ipynb tesztek
    # ---------------------------------------------------------
    st.header("3. Vásárlói Szegmentáció (02-es notebook)")
    
    # A notebook kimenetei: rfm_features.parquet és scaler
    file_02_ok = check_file_exists(RFM_FEATURES_PARQUET, "rfm_features.parquet")
    scaler_path = os.path.join(MODELS_DIR, "scaler_rfm.joblib")
    check_file_exists(scaler_path, "scaler_rfm.joblib")
    
    if file_02_ok:
        df_rfm = pd.read_parquet(RFM_FEATURES_PARQUET)
        
        # Teszt 3.1: Szükséges oszlopok megléte (Notebook alapján)
        expected_rfm_cols = ['recency_days', 'frequency', 'monetary_total']
        missing_cols = [col for col in expected_rfm_cols if col not in df_rfm.columns]
        
        run_test("RFM Oszlopok", 
                 len(missing_cols) == 0,
                 "Minden alap RFM oszlop megtalálható.",
                 f"Hiányzó oszlopok: {missing_cols}")
        
        # Teszt 3.2: Cluster oszlop megléte (ha el lett mentve a végeredmény)
        if 'Cluster' in df_rfm.columns or 'cluster' in df_rfm.columns.str.lower():
            st.success("**SIKERES:** Klaszter (szegmens) azonosítók megtalálhatóak az adatban.")
        else:
            st.info("ℹ️ **INFO:** A `Cluster` oszlop nincs a betöltött Parquet fájlban. (Lehet, hogy a 03-as notebook teszi hozzá a Pipeline-ban, vagy külön customer_segments.parquet-be mentetted).")

    # ---------------------------------------------------------
    # 03_churn_prediction.ipynb tesztek
    # ---------------------------------------------------------
    st.header("4. Churn Predikció (03-as notebook)")
    
    xgb_model_path = os.path.join(MODELS_DIR, "xgboost_churn.joblib")
    xgb_ok = check_file_exists(xgb_model_path, "xgboost_churn.joblib")
    
    if xgb_ok:
        try:
            import joblib
            model = joblib.load(xgb_model_path)
            st.success("**SIKERES:** Az XGBoost modell sikeresen betölthető.")
            
            # Ellenőrizzük, hogy Pipeline-e (Notebook 03-ban Pipeline-ként raktuk össze)
            if hasattr(model, 'predict'):
                st.success("**SIKERES:** A betöltött objektum rendelkezik `predict` metódussal.")
            else:
                st.error("**HIBA:** A modell nem rendelkezik predikciós képességgel.")
                
        except Exception as e:
            st.error(f"**HIBA:** A modell betöltése sikertelen! {e}")

st.divider()
st.markdown("*(Frissítsd az oldalt a tesztek újra-futtatásához)*")