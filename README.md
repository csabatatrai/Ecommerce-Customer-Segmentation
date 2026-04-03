<a id="teteje"></a>
# ecommerce-customer-segmentation

Data Science gépi tanulási projekt Python segítségével, Data Engineering és az MLOps-közeli szemléletmóddal kiegészítve.

## Főbb lépések

| # | Lépés | Notebook | Lefutott eredmények megtekintése |
|---|-------|----------|----------------------------------|
| 0 | Adatbetöltés és Parquet-konverzió | `01_data_preparation.ipynb` | [📊 Megtekintés](docs/01_data_preparation.md) |
| 1 | Adattisztítás és technikai outlier-szűrés, alap EDA | `01_data_preparation.ipynb` | [📊 Megtekintés](docs/01_data_preparation.md) |
| 2 | RFM Feature Engineering | `02_customer_segmentation.ipynb` | [📊 Megtekintés](docs/02_customer_segmentation.md) |
| 3 | Statisztikai outlier-kezelés és skálázás | `02_customer_segmentation.ipynb` | [📊 Megtekintés](docs/02_customer_segmentation.md) |
| 4 | K-means klaszterezés | `02_customer_segmentation.ipynb` | [📊 Megtekintés](docs/02_customer_segmentation.md) |
| 5 | Kiterjesztett EDA: klaszterek vizualizációja | `02_customer_segmentation.ipynb` | [📊 Megtekintés](docs/02_customer_segmentation.md) |
| 6 | XGBoost célváltozó tervezés | `03_churn_prediction.ipynb` | [📊 Megtekintés](docs/03_churn_prediction.md) |
| 7 | Modellezés és SHAP magyarázatok | `03_churn_prediction.ipynb` | [📊 Megtekintés](docs/03_churn_prediction.md) |
| 8 | Üzleti kiértékelés | `03_churn_prediction.ipynb` | [📊 Megtekintés](docs/03_churn_prediction.md) |
| 9 | Export (Streamlit / BI) | `03_churn_prediction.ipynb` | [📊 Megtekintés](docs/03_churn_prediction.md) |

## Adathalmaz, alapötlet

A projekt alapjául szolgáló adathalmaz leírása és az elemzés ötlete a [Kaggle-ről származik](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data) (eredeti forrás: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)).

Az adathalmaz egy Egyesült Királyságban található, ajándéktárgy-nagykereskedő 2009–2011 közötti tranzakcióit tartalmazza, közel 1 millió sorban. Az ügyfelek viszonteladók (B2B) és magánszemélyek (B2C) vegyesen, ami indokolja az RFM-alapú szegmentációs megközelítést: a visszatérő, nagyértékű vásárlók azonosítása és a lemorzsolódás előrejelzése ebben a szegmensben különösen üzletileg releváns.

## Lokális futtatás és környezet beállítása (Setup)

> **💡 Megjegyzés:** A projekt alapértelmezett bemeneti/kimeneti fájlútvonalait és a főbb paramétereket (pl. `CUTOFF_DATE`) a `config.py` fájl tartalmazza. Az útvonalakat itt lehet módosítani eltérő mappastruktúra használatához.

> [!TIP]
> **Kódolvasás Visual Studio Code-ban:** Ha VSC-t használsz a projekt megtekintéséhez, erősen ajánlott a [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments) bővítmény telepítése. A forráskódban tudatosan használok színkódolt kommenteket a fontos megjegyzések, összefüggések és kiemelések jelölésére, így a bővítmény használatával sokkal átláthatóbbá válik a kód logikája.

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
   - `01_data_preparation.ipynb` – Adatelőkészítés: Data Preparation (Adattisztítás és Parquet Pipeline)
   - `02_customer_segmentation.ipynb` – Ügyfélszegmentáció: Customer Segmentation (RFM Elemzés és K-means)
   - `03_churn_prediction.ipynb` – Prediktív Modellezés: Churn Prediction (XGBoost Klasszifikáció)

## Kiegészítő műszaki megoldások

> [!NOTE]
>Az elsődleges adatfeltárás **(EDA)** ebben a projektben SQLite-ban történt ([DB Browser for SQLite](https://sqlitebrowser.org/)), nem közvetlenül Pandasban. A futtatott lekérdezések megtalálhatók a `sql/eda_exploratory_analysis.sql` fájlban; az itt szerzett felismerések épültek be a Python pipeline tisztítási és szegmentációs logikájába.

> [!NOTE]
> Kimeneti adatformátum: **Parquet**

> [!NOTE]
> Verziókezelés **nbstripout** Git-filterre, így nem szemeteli tele a repo-t a notebook futtatási metaadatokkal, `nbstripout --install` parancs futtatása szükséges terminálból a lokális Git hook beállításához

## Mappastruktúra
>A notebookok futtatásakor a kód automatikusan létrehozza a teljes szükséges mappastruktúrát.
```text
ecommerce-customer-segmentation/
├── assets/
│   └── images/               # mentett ábrák és vizualizációk (auto-generált)
├── data/
│   ├── raw/                  # nyers, eredeti adatfájlok
│   └── processed/            # tisztított, Parquet formátumú adatok
├── docs/                     # 🚀 LEFUTOTT EREDMÉNYEK (Markdown portfólió nézet)
│   ├── images/               # Jupyter által generált inline ábrák
│   ├── 01_data_preparation.md
│   ├── 02_customer_segmentation.md
│   └── 03_churn_prediction.md
├── pages/                    # .py fájlok Streamlitnek (többoldalas alkalmazás)
├── sql/                      # SQL szkriptek az adatbázis feltérképezéséhez
│   └── eda_exploratory_analysis.sql
├── config.py                 # közös útvonal-konstansok és pipeline paraméterek
├── 01_data_preparation.ipynb
├── 02_customer_segmentation.ipynb
├── 03_churn_prediction.ipynb
├── app.py                    # Streamlit dashboard főfájl
├── .gitignore
├── requirements.txt
└── models/                   # szerializált modell- és transzformátor-objektumok (joblib)  
```
<div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 30px 0;">
    <a href="#teteje" style="display: inline-flex; align-items: center; justify-content: center; background-color: #c0253f; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; font-family: sans-serif; font-weight: bold; font-size: 14px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
        ⬆ Visszaugrás az oldal elejére
    </a>
</div>