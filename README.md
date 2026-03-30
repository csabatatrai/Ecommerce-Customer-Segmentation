# ecommerce-customer-segmentation

Data Science gépi tanulási projekt Python segítségével, Data Engineering és az MLOps-közeli szemléletmóddal kiegészítve.

## Főbb lépések

| # | Lépés | Notebook |
|---|-------|----------|
| 0 | Adatbetöltés és Parquet-konverzió | `01_data_preparation.ipynb` |
| 1 | Adattisztítás és technikai outlier-szűrés | `01_data_preparation.ipynb` |
| 2 | RFM Feature Engineering | `02_customer_segmentation.ipynb` |
| 3 | Statisztikai outlier-kezelés és skálázás | `02_customer_segmentation.ipynb` |
| 4 | K-means klaszterezés | `02_customer_segmentation.ipynb` |
| 5 | Kiterjesztett EDA: klaszterek vizualizációja | `02_customer_segmentation.ipynb` |
| 6 | XGBoost célváltozó tervezés | `03_clv_prediction.ipynb` |
| 7 | Modellezés és SHAP magyarázatok | `03_clv_prediction.ipynb` |
| 8 | Üzleti kiértékelés | `03_clv_prediction.ipynb` |
| 9 | Export (Streamlit / BI) | `03_clv_prediction.ipynb` |

## Adathalmaz és reprodukálhatóság

A projekt alapjául szolgáló adathalmaz leírása és az elemzés ötlete a [Kaggle-ről származik](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data) (eredeti forrás: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)). 

Mivel a nyers adathalmaz mérete meghaladja a GitHub által preferált korlátokat (~100 MB), a nyers CSV fájl nem képezi a repository részét. 

**Technikai döntés és adatbetöltési kihívások:** A projekt tervezésekor az elsődleges cél a teljesen automatizált, API-kulcsok nélküli reprodukálhatóság volt a hivatalos `ucimlrepo` csomag használatával. Bár az adathalmaz létezik az UCI szerverén, a Python API-juk jelenleg nem támogatja a közvetlen, szabványos DataFrame-be történő beolvasást ennél a specifikus adathalmaznál (ID 502).

A felesleges memóriaterhelés elkerülése érdekében egy robusztus "graceful fallback" megoldást implementáltam. A `01_data_preparation.ipynb` notebook első cellája ellenőrzi az adatok meglétét: ha friss klónozás történt, a notebook nem fagy le nyers hibaüzenettel, hanem pontos instrukciókat ad a Kaggle-ről történő letöltéshez. Amint a nyers CSV a helyére kerül, a rendszer automatikusan egy memóriahatékony, típusbiztos `Parquet` fájlt generál a `data/processed/` mappába, jelentősen felgyorsítva a későbbi futtatásokat (típusbiztonság és a kisebb fájlméret miatt is előnyös).

A hatékony verzióközelés érdekében a nagy méretű és bináris formátumú adatfájlok (CSV, Parquet) tudatosan nem részei a repónak (lásd: .gitignore), biztosítva ezzel a projekt gyors klónozhatóságát és a forráskód tisztaságát.

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
4. Indítsd el a Jupytert:
```bash
   jupyter notebook
```
5. Futtasd a notebookokat **sorrendben**:
   - `01_data_preparation.ipynb` – adatbetöltés és tisztítás
   - `02_customer_segmentation.ipynb` – RFM feature engineering és szegmentáció
   - `03_clv_prediction.ipynb` – prediktív modellezés (XGBoost + SHAP)

## Fejlesztés és verziókezelés (nbstripout)

A repository tisztán tartása érdekében a projekt az `nbstripout` csomagot használja. Ez a Git filter biztosítja, hogy a Jupyter Notebook (`.ipynb`) fájlok **commitolása során a futtatási kimenetek és ábrák automatikusan eltávolításra kerüljenek**. 

Ezzel a *best practice* megközelítéssel elkerülhető a felesleges bináris adatok felhalmozódása a verziótörténetben, a commit diff-ek pedig tiszták és könnyen áttekinthetők (code review kompatibilisek) maradnak. 

*(Ha te is fejleszteni szeretnéd a kódot, a környezet aktiválása után futtasd az `nbstripout --install` parancsot a lokális Git hook beállításához.)*

## Mappastruktúra
>A notebookok futtatásakor a kód automatikusan létrehozza a teljes szükséges mappastruktúrát.
```bash
ecommerce-customer-segmentation/
├── assets/
│   └── images/           # saved plots and figures (auto-generated)
├── data/
│   ├── raw/              # original dirty dataset
│   └── processed/        # cleaned, Parquet format of data
├── pages/                # .py files for multipage Streamlit app
├── config.py             # shared path constants and pipeline parameters
├── 01_data_preparation.ipynb      # data loading, cleaning, outlier filtering
├── 02_customer_segmentation.ipynb # RFM engineering, scaling, K-means clustering
├── 03_clv_prediction.ipynb        # XGBoost CLV modelling + SHAP explanations
├── app.py                # Streamlit dashboard's main file
├── .gitignore
├── requirements.txt
├── models/               # model & transformator serialization
│   ├── kmeans_rfm.joblib          
│   ├── xgboost_clv.joblib         
│   └── scaler_rfm.joblib          
```
