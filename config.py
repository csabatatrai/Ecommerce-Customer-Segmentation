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
# Mindhárom fájl megjelenik a README mappaszerkezetében (models/)
SCALER_PATH       = MODELS_DIR / "scaler_rfm.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_rfm.joblib"   # 02-es notebook önálló K-Means exportja
XGB_MODEL_PATH    = MODELS_DIR / "xgboost_clv.joblib"  # 03-as notebook: teljes Pipeline (scaler + clf)

# --- Elemzési paraméterek ---
CUTOFF_DATE  = "2011-09-09"
Q_THRESHOLD  = 10_000

# --- 02-es notebook kimenetei ---
CUSTOMER_SEGMENTS_PARQUET = PROCESSED_DIR / "customer_segments.parquet"

# --- 03-as notebook kimenetei ---
CHURN_PREDICTIONS_PARQUET = PROCESSED_DIR / "churn_predictions.parquet"

# ============================================================
# 04-es notebook – BG/NBD + Gamma-Gamma CLV regresszió
# ============================================================
# Szükséges csomag: pip install lifetimes
#
# Bemenet:  READY_FOR_RFM_PARQUET (nyers tranzakciók, cutoff előtt)
# Kimenetek:
#   CLV_PREDICTIONS_PARQUET  – ügyfelenkénti várható CLV (12 hó)
#   BGNBD_MODEL_PATH         – illesztett BG/NBD modell (joblib)
#   GAMMA_GAMMA_MODEL_PATH   – illesztett Gamma-Gamma modell (joblib)

CLV_PREDICTIONS_PARQUET = PROCESSED_DIR / "clv_predictions_bgnbd.parquet"
BGNBD_MODEL_PATH        = MODELS_DIR    / "bgnbd_model.joblib"
GAMMA_GAMMA_MODEL_PATH  = MODELS_DIR    / "gamma_gamma_model.joblib"

# Előrejelzési horizont (hónapokban) a Gamma-Gamma CLV számításhoz
CLV_HORIZON_MONTHS = 12

# --- Kimeneti mappák automatikus létrehozása ---
for _dir in [PROCESSED_DIR, MODELS_DIR, IMAGES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)