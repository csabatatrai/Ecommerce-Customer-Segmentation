# ecommerce-customer-segmentation

Data Science gépi tanulási projekt Python segítségével, Data Engineering és az MLOps-közeli szemléletmóddal kiegészítve.

## Főbb lépések

| # | Lépés | Notebook |
|---|-------|----------|
| 0 | Adatbetöltés és Parquet-konverzió | `01_data_preparation.ipynb` |
| 1 | Adattisztítás és technikai outlier-szűrés, alap EDA | `01_data_preparation.ipynb` |
| 2 | RFM Feature Engineering | `02_customer_segmentation.ipynb` |
| 3 | Statisztikai outlier-kezelés és skálázás | `02_customer_segmentation.ipynb` |
| 4 | K-means klaszterezés | `02_customer_segmentation.ipynb` |
| 5 | Kiterjesztett EDA: klaszterek vizualizációja | `02_customer_segmentation.ipynb` |
| 6 | XGBoost célváltozó tervezés | `03_clv_prediction.ipynb` |
| 7 | Modellezés és SHAP magyarázatok | `03_clv_prediction.ipynb` |
| 8 | Üzleti kiértékelés | `03_clv_prediction.ipynb` |
| 9 | Export (Streamlit / BI) | `03_clv_prediction.ipynb` |

## Adathalmaz

A projekt alapjául szolgáló adathalmaz leírása és az elemzés ötlete a [Kaggle-ről származik](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data) (eredeti forrás: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)).

Az adathalmaz egy Egyesült Királyságban található, ajándéktárgy-nagykereskedő 2009–2011 közötti tranzakcióit tartalmazza, közel 1 millió sorban. Az ügyfelek viszonteladók (B2B) és magánszemélyek (B2C) vegyesen, ami indokolja az RFM-alapú szegmentációs megközelítést: a visszatérő, nagyértékű vásárlók azonosítása és a lemorzsolódás előrejelzése ebben a szegmensben különösen üzletileg releváns.

Mivel a nyers adathalmaz mérete meghaladja a GitHub által preferált korlátokat (~100 MB), a nyers CSV fájl nem képezi a repository részét, a notebook futtatás közben tölti le.

## Lokális futtatás és környezet beállítása (Setup)

A projekt futtatásához javasolt egy izolált virtuális környezet (pl. Conda) használata:

1. Klónozd a repót és navigálj a mappába:
```bash
git clone https://github.com/csabatatrai/ecommerce-customer-segmentation
```
```bash
cd ecommerce-customer-segmentation
```
2. Hozz létre egy új környezetet:
```bash
conda create --name ecommerce_env python=3.10
```
```bash
conda activate ecommerce_env
```
3. Telepítsd a függőségeket:
```bash
pip install -r requirements.txt
```
4. Töltsd le a nyers adathalmazt a [Kaggle-ről](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data), csomagold ki, és az `online_retail_II.csv` fájlt helyezd el a `data/raw/` mappába.

5. Indítsd el a Jupytert:
```bash
jupyter notebook
```
6. Futtasd a notebookokat **sorrendben**:
   - `01_data_preparation.ipynb` – adatbetöltés és tisztítás
   - `02_customer_segmentation.ipynb` – RFM feature engineering és szegmentáció
   - `03_clv_prediction.ipynb` – prediktív modellezés (XGBoost + SHAP)

> **Fontos:** A pipeline globális paraméterei (útvonalak, `CUTOFF_DATE`, `Q_THRESHOLD`) a `config.py` fájlban vannak központosítva. Ha szükséges, a notebookok futtatása előtt itt állítsd be őket.

> **Fejlesztőknek:** A notebookok kimenetének tisztántartásához (hogy ne kerüljenek fel a kimenetek is pl. Github-ra) futtasd az `nbstripout --install` parancsot a lokális Git hook beállításához.

## Exploratív adatelemzés (EDA) - SQLite alapokon

Az adathalmaz elsődleges feltárása nem közvetlenül Pandas-ban, hanem SQLite adatbázisban történt. A lekérdezések futtatásához használt eszköz: [DB Browser for SQLite](https://sqlitebrowser.org/).

A nyers, „koszos" CSV fájlt egy SQLite táblába (`ecomStore` néven) töltöttem be, és az exploratív elemzést SQL lekérdezéseken keresztül végeztem. Ez lehetővé tette:

- nagyobb adatmennyiség gyors, aggregáció-alapú vizsgálatát
- üzleti kérdések strukturált tesztelését (pl. visszaküldési arányok, vásárlói lojalitás)
- adatminőségi problémák azonosítását (pl. adminisztratív tételek, negatív mennyiségek)

Az EDA során futtatott SQL lekérdezések megtalálhatók a `sql/eda_exploratory_analysis.sql` fájlban.

Az itt szerzett felismerések közvetlenül beépítésre kerültek a Python pipeline-ba:
→ `01_data_preparation.ipynb`: tisztítási szabályok (pl. visszáruk kezelése, szűrések)
→ `02_customer_segmentation.ipynb`: RFM feature engineering és szegmentációs stratégia

Ez a megközelítés szétválasztja:
- a **feltárást (SQL)** és
- a **produkciós pipeline-t (Python + Parquet)**

így a projekt egyszerre reprodukálható és skálázható.

## Műszaki megjegyzések

**Adatformátum (Parquet):** A nyers CSV adatokat az első lépésben Apache Parquet formátumba transzformálom és a `data/processed/` mappában tárolom. Az oszlop-alapú tárolás gyorsabb beolvasást és kisebb memóriahasználatot tesz lehetővé, a Parquet pedig megőrzi a sémát (pl. `InvoiceDate` datetime, `Customer ID` integer típusa), elkerülve a CSV-re jellemző ismételt típuskonverziós hibákat.

**Verziókezelés (nbstripout):** Az `nbstripout` Git-filter biztosítja, hogy a távoli repóba kizárólag a forráskód kerüljön be, a futtatási metaadatok nélkül. A nagy méretű adatfájlok (CSV, Parquet) tudatosan nem részei a repónak (lásd: `.gitignore`).

**Adatbetöltési kihívások:** A projekt tervezésekor az elsődleges cél a teljesen automatizált reprodukálhatóság volt a hivatalos `ucimlrepo` csomag használatával. Bár az adathalmaz létezik az UCI szerverén, a Python API jelenleg nem támogatja a közvetlen DataFrame-be történő beolvasást ennél a specifikus adathalmaznál (ID 502). A `01_data_preparation.ipynb` első cellája ezért egy "graceful fallback" megoldást alkalmaz: ha a nyers CSV hiányzik, pontos letöltési instrukciókat ad, és nem fagy le nyers hibaüzenettel.

## Mappastruktúra
>A notebookok futtatásakor a kód automatikusan létrehozza a teljes szükséges mappastruktúrát.
```
ecommerce-customer-segmentation/
├── assets/
│   └── images/               # mentett ábrák és vizualizációk (auto-generált)
├── data/
│   ├── raw/                  # nyers, eredeti adatfájlok
│   └── processed/            # tisztított, Parquet formátumú adatok
├── pages/                    # .py fájlok Streamlitnek (többoldalas alkalmazás)
├── sql/                      # SQL szkriptek az adatbázis feltérképezéséhez
│   └── eda_exploratory_analysis.sql
├── config.py                 # közös útvonal-konstansok és pipeline paraméterek
├── 01_data_preparation.ipynb
├── 02_customer_segmentation.ipynb
├── 03_clv_prediction.ipynb
├── app.py                    # Streamlit dashboard főfájl
├── .gitignore
├── requirements.txt
└── models/                   # szerializált modell- és transzformátor-objektumok (joblib)
    ├── kmeans_rfm.joblib
    ├── xgboost_clv.joblib
    └── scaler_rfm.joblib
```