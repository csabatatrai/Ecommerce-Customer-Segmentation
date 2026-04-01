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

> **💡 Megjegyzés:** A projekt alapértelmezett bemeneti/kimeneti fájlútvonalait és a főbb paramétereket (pl. `CUTOFF_DATE`) a `config.py` fájl tartalmazza. Az útvonalakat itt lehet módosítani eltérő mappastruktúra használatához.

A projekt futtatásához javasolt egy izolált virtuális környezet (pl. Conda) használata:

1. Klónozd a repót és navigálj a mappába:
```bash
git clone https://github.com/csabatatrai/ecommerce-customer-segmentation
cd ecommerce-customer-segmentation
```

2. Hozz létre egy új környezetet:
```bash
conda create --name ecommerce_env python=3.10
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

## Megjegyzések

> [!NOTE]
>Az elsődleges adatfeltárás **(EDA)** ebben a projektben SQLite-ban történt ([DB Browser for SQLite](https://sqlitebrowser.org/)), nem közvetlenül Pandasban. A futtatott lekérdezések megtalálhatók a `sql/eda_exploratory_analysis.sql` fájlban; az itt szerzett felismerések épültek be a Python pipeline tisztítási és szegmentációs logikájába.

> [!NOTE]
> Kimeneti adatformátum: **Parquet**

> [!NOTE]
> Verziókezelés **nbstripout** Git-filterre, így nem szemeteli tele a repo-t a notebook futtatási metaadatokkal, `nbstripout --install` parancs futtatása szükséges terminálból a lokális Git hook beállításához

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