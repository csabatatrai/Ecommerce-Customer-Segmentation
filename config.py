# ============================================================
# config.py – Közös projekt-konfiguráció
# ============================================================
# Minden notebook ebből importálja az útvonalakat és paramétereket.
# Path(__file__).parent biztosítja, hogy az útvonalak a config.py
# helyétől (projekt gyökér) függetlenül helyesek legyenek.

from pathlib import Path

# --- Alapvető útvonalak ---
PROJECT_ROOT  = Path(__file__).parent.resolve()
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
IMAGES_DIR    = PROJECT_ROOT / "assets" / "images"

# --- Fájlnevek ---
RAW_FILE              = RAW_DIR       / "online_retail_II.csv"
PARQUET_OUT           = PROCESSED_DIR / "online_retail_raw.parquet"
CLEANED_PARQUET       = PROCESSED_DIR / "online_retail_cleaned.parquet"
READY_FOR_RFM_PARQUET = PROCESSED_DIR / "online_retail_ready_for_rfm.parquet"
RFM_FEATURES_PARQUET  = PROCESSED_DIR / "rfm_features.parquet"

# --- Modell fájlok ---
SCALER_PATH = MODELS_DIR / "scaler_rfm.joblib"

# --- Elemzési paraméterek ---
CUTOFF_DATE   = "2011-09-09"
Q_THRESHOLD   = 10_000
