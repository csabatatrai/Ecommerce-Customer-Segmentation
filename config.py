# ============================================================
# config.py – Közös projekt-konfiguráció
# ============================================================
# Minden notebook ebből importálja az útvonalakat és paramétereket.
# Path(__file__).parent biztosítja, hogy az útvonalak a config.py
# helyétől (projekt gyökér) függetlenül helyesek legyenek.

from pathlib import Path

# --- Alapvető útvonalak ---
# A projekt gyökerétől kiindulva definiáljuk a fontosabb könyvtárakat.
PROJECT_ROOT  = Path(__file__).parent.resolve()
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
# IMAGES_DIR    = PROJECT_ROOT / "assets" / "images" # szükségtelenné vált, mert végül a docs-ba generáltattam a grafikonokat

# --- Adatfájlok és kimenetek ---
#* data/
RAW_FILE              = RAW_DIR       / "online_retail_II.csv"   # CSV fallback (kézi letöltésnél)
RAW_XLSX              = RAW_DIR       / "online_retail_II.xlsx"  # UCI-ról automatikusan letöltött forrás
PARQUET_OUT           = RAW_DIR       / "online_retail_raw.parquet" #! <-- Most már a raw mappába mentem, hogy elkülönüljön a feldolgozott adatoktól!
CLEANED_PARQUET       = PROCESSED_DIR / "online_retail_cleaned.parquet"
READY_FOR_RFM_PARQUET = PROCESSED_DIR / "online_retail_ready_for_rfm.parquet"
RFM_FEATURES_PARQUET  = PROCESSED_DIR / "rfm_features.parquet"

# --- Modell fájlok ---
#* models/
SCALER_PATH       = MODELS_DIR / "scaler_rfm.joblib"    # 01-es notebook exportja
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_rfm.joblib"   # 02-es notebook exportja
XGB_MODEL_PATH = MODELS_DIR / "xgboost_churn.joblib"  # 03-as notebook exportja

# --- Elemzési paraméterek ---
# CUTOFF_DATE = pd.Timestamp.now() - pd.DateOffset(days=90) #! <-- Dinamikus vágás: mindig a legfrissebb adatoktól számított 90 nap, de most fix dátumra állítva a reprodukálhatóság miatt
CUTOFF_DATE  = "2011-09-09" #! <-- Itt módosítható a vágás helye a bővülő ablakos elemzéshez, ha jönne bővített adatkészlet vagy új cutoff dátum
Q_THRESHOLD  = 10_000

# --- 02-es notebook kimenetei ---
CUSTOMER_SEGMENTS_PARQUET = PROCESSED_DIR / "customer_segments.parquet"

# --- 03-as notebook kimenetei ---
CHURN_PREDICTIONS_PARQUET = PROCESSED_DIR / "churn_predictions.parquet"


"""
! Későbbi 05-ös CLV notebook bővítéshez:
"""
# ============================================================
# 05-ös notebook – BG/NBD + Gamma-Gamma CLV regresszió
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
# [..., IMAGES_DIR] törölve lett, mert a grafikonokat végül a docs/images könyvtárba generáltatom, így nincs rá szükség
# RAW_DIR is belekerült, hogy az első notebook futtatása előtt ne kelljen kézzel létrehozni
for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

