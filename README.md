<a id="teteje"></a>
# Ecommerce Customer Segmentation & Churn Prediction

![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![ML](https://img.shields.io/badge/focus-MLOps_&_Engineering-green)

**Végponttól végpontig tartó adattermék, amely az RFM alapú szegmentációt ötvözi prediktív churn-modellezéssel, SQL-alapú feltárással és interaktív Streamlit dashboarddal, data engineering és MLOps-közeli szemlélettel.**

[<img src="https://github.com/csabatatrai/csabatatrai/blob/main/dataproduct1.webp?raw=true" alt="Ecommerce projekt kép" title="Kattints a portfólióm megtekintéséhez!">](https://csabatatrai.hu/)

<p align="center">
  <a href="https://csabatatrai.hu/">🌐 Látogasd meg a portfóliómat (külső weboldal)</a>
</p>

## Felhasznált adathalmaz

A projekt alapjául szolgáló adathalmaz leírása és az elemzés ötlete a [Kaggle-ről származik](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data) (eredeti forrás: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)).

Az adathalmaz egy Egyesült Királyságban található, ajándéktárgy-nagykereskedő 2009–2011 közötti tranzakcióit tartalmazza, közel 1 millió sorban. Az ügyfelek viszonteladók (B2B) és magánszemélyek (B2C) vegyesen, ami indokolja az RFM-alapú szegmentációs megközelítést: a visszatérő, nagyértékű vásárlók azonosítása és a lemorzsolódás előrejelzése ebben a szegmensben különösen üzletileg releváns.

## Elemzés főbb lépései

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

### 📊 Interaktív Dashboard & Vizualizáció

> **Ízelítő:** Az alábbi animáció a vásárlási tranzakciók időbeli dinamikáját és a projekt interaktív felületét mutatja be.

<p align="center">
  <a href="https://csabatatrai.hu/">
    <img src="https://github.com/csabatatrai/csabatatrai/blob/main/purchases.gif?raw=true" alt="Purchases dynamics" title="Kattints a teljes dashboardért!">
  </a>
  <br>
  <a href="https://csabatatrai.hu/">
    <img src="https://img.shields.io/badge/Kattints_a_teljes_dashboardért-E4405F?style=for-the-badge&logo=streamlit&logoColor=white" alt="Dashboard gomb">
  </a>
</p>

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
## Architektúra-diagram

```mermaid
flowchart TD
    CSV[("📂 online_retail_II.csv\nKaggle / UCI · ~1M sor")]
    SQL["🔍 SQL EDA\nSQLite / DB Browser"]
    CFG(["⚙️ config.py\nÚtvonalak · paraméterek"])

    SQL -.->|feltárás| CSV
    CFG -.-> PREP
    CFG -.-> SEG
    CFG -.-> CHURN
    CSV --> PREP

    PREP["📋 01 Adatelőkészítés\nParquet konverzió · tisztítás · outlier szűrés"]
    SEG["🎯 02 Ügyfélszegmentáció\nRFM Feature Engineering · K-means K=4"]
    CHURN["🤖 03 Churn Prediction\nXGBoost · SHAP · A/B pipeline tesztelés"]
    DASH["📊 Streamlit Dashboard"]

    PREP --> SEG --> CHURN --> DASH

    PREP -.->|kimenet| P1[("💾 Parquet fájlok\nraw · cleaned · rfm_ready")]
    SEG  -.->|kimenet| M1[("🧩 Modellek × 2\nscaler · kmeans_rfm .joblib")]
    CHURN-.->|kimenet| P2[("📈 Előrejelzések\nxgboost.joblib · churn_pred.parquet")]
```

## Kapcsolat a készítővel

Ha kérdésed van a projekttel kapcsolatban, vagy szívesen beszélgetnél hasonló témákról, keress bátran az alábbi elérhetőségeken:

* **Weboldal:** [csabatatrai.hu](https://csabatatrai.hu/)
* **LinkedIn:** [linkedin.com/in/csabatatrai-datascientist](https://www.linkedin.com/in/csabatatrai-datascientist/)
* **E-mail:** [tatraicsababprof@gmail.com](mailto:tatraicsababprof@gmail.com)

---

<p align="center">
  <br>
  <a href="#teteje">
    <img src="https://img.shields.io/badge/%E2%AC%86%20Vissza%20a%20tetej%C3%A9re-c0253f?style=for-the-badge" alt="Vissza a tetejére">
  </a>
  <br>
</p>

