# ecommerce-customer-segmentation

Gépi tanulási projekt Python segítségével.

## Adathalmaz és reprodukálhatóság

A projekt alapjául szolgáló adathalmaz leírása és az elemzés ötlete a [Kaggle-ről származik](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data). 

Mivel a nyers adathalmaz mérete meghaladja a GitHub által preferált korlátokat (~100 MB), a nyers CSV fájl nem képezi a repository részét. Annak érdekében, hogy a projekt **bárki számára azonnal, manuális fájlletöltés és Kaggle API kulcsok konfigurálása nélkül reprodukálható legyen**, a kód automatizált adatletöltést használ.

Az `analysis.ipynb` notebook első cellái a hivatalos forrásból, az **UCI Machine Learning Repository**-ból húzzák be az adatokat az `ucimlrepo` csomag segítségével. A sikeres letöltés után a rendszer automatikusan egy memóriahatékony, típusbiztos `Parquet` fájlt generál a `data/processed/` mappába, jelentősen felgyorsítva a későbbi futtatásokat.

## Lokális futtatás és környezet beállítása (Setup)

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
4. Indítsd el a Jupytert:
```bash
   jupyter notebook
```

## Fejlesztés és verziókezelés (nbstripout)

A repository tisztán tartása érdekében a projekt az `nbstripout` csomagot használja. Ez a Git filter biztosítja, hogy a Jupyter Notebook (`.ipynb`) fájlok **commitolása során a futtatási kimenetek és ábrák automatikusan eltávolításra kerüljenek**. 

Ezzel a *best practice* megközelítéssel elkerülhető a felesleges bináris adatok felhalmozódása a verziótörténetben, a commit diff-ek pedig tiszták és könnyen áttekinthetők (code review kompatibilisek) maradnak. 

*(Ha te is fejleszteni szeretnéd a kódot, a környezet aktiválása után futtasd az `nbstripout --install` parancsot a lokális Git hook beállításához.)*

## Mappastruktúra
```bash
ecommerce-customer-segmentation/
├── data/
│   ├── raw/              # original dirty dataset
│   └── processed/        # cleaned, Parquet format of data
├── pages/                # .py files for multipage Streamlit app
├── analysis.ipynb        # eda & modelling
├── app.py                # Streamlit dashboard's main file
├── CONTEXT.md            # this is my secret ingredient
├── .gitignore
├── requirements.txt
├── models/               # model & transformator serialization
│   ├── kmeans_rfm.joblib          
│   ├── xgboost_clv.joblib         
│   └── scaler_rfm.joblib          
```