<a id="teteje"></a>
# 03 Prediktív Modellezés: Churn Prediction (XGBoost)
---
**Függőség:** `config.py` (Útvonalak definíciója és a `CUTOFF_DATE` paraméter az időablak felosztásához) és `02_customer_segmentation.ipynb` (előtte kell futtatni!)

---

**Bemenet:**
- `data/processed/online_retail_ready_for_rfm.parquet` (A 01-es notebook kimenete nyers idősoros tranzakciókkal)

**Kimenetek:**
- `models/xgboost_churn.joblib` (A végleges, teljes adathalmazon betanított prediktív modell)
- `data/processed/test_set.parquet` (Holdout teszt szett – a 04-es notebook kalibrációs és kiértékelési lépéseihez)

---

> **📋 Notebook határok:**
> Ez a notebook a modellezési pipeline-t fedi le: feature engineering → holdout split → keresztvalidáció → modellválasztás → overfitting monitoring → végleges modell exportja.
> A részletes kiértékelés (SHAP, kalibráció, feature importance, threshold optimalizálás, akciólista) a `04_model_evaluation.ipynb` notebookban folytatódik.

---

Ebben a fázisban a gépi tanulás felügyelt részére térünk át. A célunk, hogy a `config.py`-ban rögzített `CUTOFF_DATE` mentén az idősort két részre vágva:
1. A **megfigyelési ablakból** (`< CUTOFF_DATE`) futásidőben számítjuk ki az RFM feature-öket (X), ezzel garantálva az adatszivárgás-mentességet.
2. A **célablakból** (`>= CUTOFF_DATE`) keletkező bináris `churn` változót (y) jósoljuk.

**Miért NEM a 02-es notebook `customer_segments.parquet` fájlját olvassuk be?**
Mert az már egy aggregált, előre klaszterezett snapshot. Ha abból próbálnánk célváltozót képezni, az RFM értékekben már benne lenne a jövő (*time-travel leakage*). A megoldás: a tisztított tranzakciós adatokat (`READY_FOR_RFM_PARQUET`) töltjük be, és a cutoff-ot itt, futásidőben alkalmazzuk.

---

## 6. Adatbetöltés, Time-Split és Célváltozó (Churn) kialakítása

### 6.1 Importok és konfiguráció

Az első lépés a szükséges könyvtárak betöltése és a reprodukálhatóság beállítása.

Két dolgot érdemes kiemelni:
- A `warnings.filterwarnings` itt **célzottan** szűr: csak az ismert, ártalmatlan sklearn/xgboost FutureWarning-okat nyomja el, minden mást (pl. konvergencia-figyelmeztetést) látható hagyja.
- A `RANDOM_STATE = 42` rögzítése biztosítja, hogy a notebook minden futtatásnál azonos eredményt adjon.


```python
# ============================================================
# 6.1 – Importok és konfiguráció
# ============================================================
import warnings
# Célzott szűrés az összes elnémítása helyett:
# csak az ismert, ártalmatlan sklearn/xgboost FutureWarning-okat nyomjuk el.
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score,
    precision_recall_curve, make_scorer
)
from xgboost import XGBClassifier

from config import (
    READY_FOR_RFM_PARQUET, CUSTOMER_SEGMENTS_PARQUET,
    MODELS_DIR, PROCESSED_DIR, CUTOFF_DATE
)

# Reprodukálhatóság
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print(f"Cutoff dátum (config.py-ból): {CUTOFF_DATE}")
print(f"Adatforrás: {READY_FOR_RFM_PARQUET}")
```

    Cutoff dátum (config.py-ból): 2011-09-09
    Adatforrás: D:\Workspace\ecommerce-customer-segmentation\data\processed\online_retail_ready_for_rfm.parquet
    

### 6.2 Adatbetöltés és Time-Split

Az adatot két részre osztjuk a `CUTOFF_DATE` mentén:
- **Megfigyelési ablak** (`< CUTOFF_DATE`): ebből számítjuk a feature-öket (X).
- **Célablak** (`>= CUTOFF_DATE`): ebből képezzük a churn célváltozót (y).

Ez a kettéválasztás **adatszivárgás-mentes** (time-split): a modell nem látja a jövőbeli tranzakciókat a feature engineering során.


```python
# ============================================================
# 6.2 – Adatbetöltés és Time-Split
# ============================================================
# KRITIKUS: A READY_FOR_RFM_PARQUET-t töltjük be (nyers tranzakciók),
# NEM a customer_segments.parquet-t! Így tudjuk a cutoff-ot futásidőben
# alkalmazni, és elkerüljük a time-travel data leakage-t.

df = pd.read_parquet(READY_FOR_RFM_PARQUET)
CUTOFF_DATE_TS = pd.to_datetime(CUTOFF_DATE)

# Idővonal kettévágása
df_obs    = df[df['InvoiceDate'] <  CUTOFF_DATE_TS].copy()  # X ablak (feature-ök alapja)
df_target = df[df['InvoiceDate'] >= CUTOFF_DATE_TS].copy()  # y ablak (célváltozó alapja)

print("=" * 60)
print(f"Teljes időszak:        {df['InvoiceDate'].min().date()}  →  {df['InvoiceDate'].max().date()}")
print(f"Cutoff dátum:          {CUTOFF_DATE_TS.date()}")
print("-" * 60)
print(f"Megfigyelési ablak (X): {len(df_obs):,} sor  |  {df_obs['Customer ID'].nunique():,} egyedi ügyfél")
print(f"Célablak (y):           {len(df_target):,} sor  |  {df_target['Customer ID'].nunique():,} egyedi ügyfél")
print("=" * 60)
```

    ============================================================
    Teljes időszak:        2009-12-01  →  2011-12-09
    Cutoff dátum:          2011-09-09
    ------------------------------------------------------------
    Megfigyelési ablak (X): 631,337 sor  |  5,250 egyedi ügyfél
    Célablak (y):           162,563 sor  |  2,920 egyedi ügyfél
    ============================================================
    

### 6.3 RFM és kiterjesztett feature-ök kiszámítása

A klasszikus RFM triple (`recency_days`, `frequency`, `monetary_total`) mellé két further feature kerül:
- **`monetary_avg`**: átlagos kosárérték rendelésenként – kiszűri a nagyrendelők és a sok kis rendelés eltérését.
- **`return_ratio`**: visszáru-arány – az ügyfél elégedetlenségének proxy-ja.

**Kritikus:** minden számítás `df_obs`-on (a megfigyelési ablakon) fut. A `df_target` adatait egyetlen lépés sem látja.


```python
# ============================================================
# 6.3 – RFM + kiterjesztett feature-ök kiszámítása (CSAK az X ablakból)
# ============================================================
# Az összes számítás SZIGORÚAN a df_obs-on történik.
# A df_target adatait NEM látja a feature engineering.

purchases = df_obs[df_obs['Quantity'] > 0]
returns   = df_obs[df_obs['Quantity'] < 0]

# RFM alap
rfm = purchases.groupby('Customer ID').agg(
    recency_days  = ('InvoiceDate', lambda x: (CUTOFF_DATE_TS - x.max()).days),
    frequency     = ('Invoice', 'nunique')
)

# Monetary (nettó, sztornókkal együtt)
monetary = df_obs.groupby('Customer ID')['LineTotal'].sum().rename('monetary_total')
rfm = rfm.join(monetary)

# Return metrikák
return_counts = returns.groupby('Customer ID')['Invoice'].nunique().rename('return_count')
rfm = rfm.join(return_counts).fillna({'return_count': 0})

# Visszaküldési arány
rfm['monetary_avg']  = rfm['monetary_total'] / rfm['frequency']
rfm['return_ratio']  = rfm['return_count'] / (rfm['frequency'] + rfm['return_count'])

# Csak azokat tartjuk, akik az X ablakban pozitív nettó értékkel bírnak
rfm = rfm[rfm['monetary_total'] > 0]

print(f"Kiszámított RFM mátrix: {rfm.shape[0]:,} ügyfél × {rfm.shape[1]} feature")
display(rfm.describe().round(2))
```

    Kiszámított RFM mátrix: 5,243 ügyfél × 6 feature
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency_days</th>
      <th>frequency</th>
      <th>monetary_total</th>
      <th>return_count</th>
      <th>monetary_avg</th>
      <th>return_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5243.00</td>
      <td>5243.00</td>
      <td>5243.00</td>
      <td>5243.00</td>
      <td>5243.00</td>
      <td>5243.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>205.36</td>
      <td>5.71</td>
      <td>2518.84</td>
      <td>1.17</td>
      <td>355.00</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>std</th>
      <td>173.79</td>
      <td>11.21</td>
      <td>11811.34</td>
      <td>3.04</td>
      <td>489.94</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>49.00</td>
      <td>1.00</td>
      <td>310.84</td>
      <td>0.00</td>
      <td>170.23</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>162.00</td>
      <td>3.00</td>
      <td>761.46</td>
      <td>0.00</td>
      <td>268.55</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>323.00</td>
      <td>6.00</td>
      <td>2008.94</td>
      <td>1.00</td>
      <td>399.14</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>max</th>
      <td>646.00</td>
      <td>284.00</td>
      <td>454202.09</td>
      <td>70.00</td>
      <td>13206.50</td>
      <td>0.80</td>
    </tr>
  </tbody>
</table>
</div>


### 6.4 Churn célváltozó (y) kialakítása

A churn definíciója binárisan egyszerű:
- **Churn = 1**: az ügyfél **nem** szerepel a célablakban (eltűnt).
- **Churn = 0**: az ügyfél legalább egyszer tranzaktált a célablakban (aktív maradt).

A churn arány meghatározza, hogy mennyire kiegyensúlyozottak az osztályok – ez befolyásolja a helyes értékelési metrika megválasztását.


```python
# ============================================================
# 6.4 – Churn célváltozó (y) kialakítása
# ============================================================
# Churn = 1: a vásárló NEM tranzaktált a célablakban (eltűnt)
# Churn = 0: a vásárló legalább egyszer tranzaktált a célablakban (aktív maradt)
#
# FONTOS: Csak azokat az ügyfeleket tartjuk meg, akik az X ablakban IS léteznek
# (left join az rfm index-re).

# Az y ablakban aktív ügyfelek halmaza
active_in_target = set(df_target['Customer ID'].unique())

# Churn flag: aki NINCS az aktív halmazban → churned
rfm['churn'] = rfm.index.map(lambda cid: 0 if cid in active_in_target else 1)

churn_counts = rfm['churn'].value_counts()
churn_rate   = rfm['churn'].mean() * 100

print("Churn eloszlás:")
print(f"  Churn = 0 (aktív marad):   {churn_counts[0]:,} ügyfél")
print(f"  Churn = 1 (lemorzsolódik): {churn_counts[1]:,} ügyfél")
print(f"  Churn arány: {churn_rate:.1f}%")

minority_rate = min(churn_rate, 100 - churn_rate)
if minority_rate < 20:
    print(f"\n⚠️  Erősen imbalanced osztályok (kisebbségi osztály: {minority_rate:.1f}%) → PR-AUC, F1 és Recall metrikákra fókuszálunk (nem Accuracy-ra).")
elif minority_rate < 35:
    print(f"\nℹ️  Mérsékelt class imbalance (kisebbségi osztály: {minority_rate:.1f}%) – PR-AUC és F1 informatívabb metrika, mint az Accuracy.")
else:
    print(f"\nℹ️  Az osztályok közel kiegyensúlyozottak (kisebbségi osztály: {minority_rate:.1f}%) – PR-AUC és F1 így is informatívabb metrika, mint az Accuracy-nál.")
```

    Churn eloszlás:
      Churn = 0 (aktív marad):   2,321 ügyfél
      Churn = 1 (lemorzsolódik): 2,922 ügyfél
      Churn arány: 55.7%
    
    ℹ️  Az osztályok közel kiegyensúlyozottak (kisebbségi osztály: 44.3%) – PR-AUC és F1 így is informatívabb metrika, mint az Accuracy-nál.
    

### 6.5 Feature mátrix (X) és célváltozó (y) elkülönítése

A `return_count` nyers értéket tudatosan hagyjuk ki a feature-ök közül: a `return_ratio` már normalizálva tartalmazza ugyanezt az információt (a rendelések számához viszonyítva), így elkerüljük a redundáns, korrelált feature bevonását.


```python
# ============================================================
# 6.5 – Feature mátrix (X) és célváltozó (y) elkülönítése
# ============================================================
FEATURE_COLS = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'return_ratio']

X = rfm[FEATURE_COLS].copy()
y = rfm['churn'].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature-ök: {FEATURE_COLS}")
display(X.head())
```

    X shape: (5243, 5)
    y shape: (5243,)
    
    Feature-ök: ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'return_ratio']
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency_days</th>
      <th>frequency</th>
      <th>monetary_total</th>
      <th>monetary_avg</th>
      <th>return_ratio</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346</th>
      <td>437</td>
      <td>2</td>
      <td>169.36</td>
      <td>84.680</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12347</th>
      <td>37</td>
      <td>6</td>
      <td>3402.39</td>
      <td>567.065</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12348</th>
      <td>156</td>
      <td>4</td>
      <td>1388.40</td>
      <td>347.100</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12349</th>
      <td>315</td>
      <td>2</td>
      <td>2196.99</td>
      <td>1098.495</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>12350</th>
      <td>218</td>
      <td>1</td>
      <td>294.40</td>
      <td>294.400</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 6.6 Holdout teszt szett elkülönítése

A keresztvalidáció (CV) jól becsüli a modell általánosítási képességét, de van egy vakfoltja: mivel a CV-foldok az összes adatból kerülnek ki, a hyperparaméter-választás (pl. melyik pipeline nyert) közvetve az egész adathalmazt felhasználja. Egy teljesen érintetlen **holdout teszt szett** ezért a CV-nél is megbízhatóbb „utolsó szó" a valódi generalizációs képességről.

**A split stratégiája:**
- **80% → tréning (`X_train`):** Ezen fut a CV és a modell betanítása.
- **20% → teszt (`X_test`):** Ehhez a modell a teljes pipeline során nem fér hozzá. Kizárólag a végső overfitting-ellenőrzéshez és a 04-es notebookban a kalibrációs elemzéshez használjuk.
- **`stratify=y`:** Biztosítja, hogy a churn arány mindkét részben azonos legyen – imbalanced osztályoknál ez kritikus.

> ⚠️ **Fontos:** A CV a 8.1-es cellától kizárólag `X_train`-en fut. Az `X_test` az egész modellezési folyamat során „zár alatt marad", és csak a 8.4-es cellában nyitjuk ki egyetlen alkalommal a végső kiértékeléshez.


```python
# ============================================================
# 6.6 – Holdout teszt szett elkülönítése
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Adathalmaz felosztása:")
print(f"  Tréning (X_train): {X_train.shape[0]:,} ügyfél  |  churn arány: {y_train.mean()*100:.1f}%")
print(f"  Teszt   (X_test):  {X_test.shape[0]:,} ügyfél  |  churn arány: {y_test.mean()*100:.1f}%")
print(f"\n✅ X_test mostantól 'zár alatt' – a CV és a modellválasztás nem látja.")
```

    Adathalmaz felosztása:
      Tréning (X_train): 4,194 ügyfél  |  churn arány: 55.7%
      Teszt   (X_test):  1,049 ügyfél  |  churn arány: 55.8%
    
    ✅ X_test mostantól 'zár alatt' – a CV és a modellválasztás nem látja.
    

## 7. A/B Modellezés: Pipeline-ok felépítése

### Az adatszivárgás-mentes K-Means Pipeline logikája

A Modell B-ben a K-Means klaszterező **nem előre futtatott**, hanem a scikit-learn `Pipeline` belső lépéseként kerül be. Ez garantálja, hogy a keresztvalidáció során a K-Means **minden foldban kizárólag a tréningadatokon** tanul – a validációs halmazon csak `predict` (`transform`) fut.

**OHE – one hot encoding**

A pipeline felépítése:
```
X (RFM) ──▶ StandardScaler ──┬──▶ KMeans (4 klaszter) ──▶ Klasztercímke (OHE) ──┐
                              │                                                    ├──▶ XGBClassifier
                              └──▶ Skálázott RFM (passthrough) ────────────────────┘
```

A `FeatureUnion` kombinálja a két ágat egyetlen feature mátrixszá.

### 7.1 KMeansFeaturizer – egyedi, leakage-mentes transformer

A `KMeansFeaturizer` egy scikit-learn kompatibilis egyedi transformer, amely a K-Means klaszterező és a One-Hot Encoding lépéseket egységbe fogja.

Miért szükséges ez az egyedi osztály? Ha a K-Means-t a `Pipeline`-on **kívül** futtatnánk előre, a klaszter-hozzárendelések az egész adathalmazt „látnák" – a validációs foldokat is. A `Pipeline`-ba ágyazva a K-Means **csak a tréningadatokon** tanul minden CV-foldban, a validációs adatokra csak `transform` fut. Ez garantálja az adatszivárgás-mentességet.


```python
# ============================================================
# 7.1 – Egyedi Transformer: KMeans → OHE (Pipeline-kompatibilis)
# ============================================================

class KMeansFeaturizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn kompatibilis transformer, amely:
    1. KMeans-t illeszt a bemeneti adatokra (CSAK fit-kor),
    2. Klasztercímkéket rendel az adatpontokhoz,
    3. A klasztercímkéket One-Hot Encodes formátumba alakítja.

    Pipeline-ban használva garantálja az adatszivárgás-mentességet:
    a KMeans minden CV-foldban csak a tréningadatokon tanul.
    """
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters   = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(X)
        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        labels = self.kmeans_.labels_.reshape(-1, 1)
        self.ohe_.fit(labels)
        return self

    def transform(self, X, y=None):
        labels = self.kmeans_.predict(X).reshape(-1, 1)
        return self.ohe_.transform(labels)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'cluster_{i}' for i in range(self.n_clusters)])


print("✔️ KMeansFeaturizer transformer definiálva.")
print("   Ez a transformer garantálja, hogy a K-Means CSAK a tréningadatokon tanul,")
print("   és a CV során nem szivárog át információ a validációs foldokba.")
```

    ✔️ KMeansFeaturizer transformer definiálva.
       Ez a transformer garantálja, hogy a K-Means CSAK a tréningadatokon tanul,
       és a CV során nem szivárog át információ a validációs foldokba.
    

### 7.2 Modell A Pipeline – Csak RFM

Az alap pipeline: a nyers RFM feature-ök közvetlenül az XGBoost-ba kerülnek, skálázás nélkül (fa-alapú modellnek nincs szüksége standardizálásra).

Az `scale_pos_weight` paramétert az osztályarányokból számítjuk **X_train alapján** – ez biztosítja, hogy a teszt szett semmilyen formában ne befolyásolja a modell inicializálását.


```python
# ============================================================
# 7.2 – Modell A: Csak RFM Pipeline
# ============================================================

xgb_params = dict(
    n_estimators    = 300,
    max_depth       = 4,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 5,
    scale_pos_weight= (y_train == 0).sum() / (y_train == 1).sum(),  # Imbalance kezelés – X_train alapján!
    base_score      = 0.5,          # Fix: XGBoost ≥2.0 + SHAP kompatibilitás
    eval_metric     = 'aucpr',
    random_state    = RANDOM_STATE,
    n_jobs          = -1,
)

pipeline_a = Pipeline([
    ('clf', XGBClassifier(**xgb_params))
])

print("✔️ Modell A Pipeline definiálva:")
print("   XGBClassifier (skálázás nélkül – fa-alapú modellnél felesleges)")
print(f"   scale_pos_weight = {xgb_params['scale_pos_weight']:.2f}  (osztályegyensúly-korrekció, X_train alapján)")
```

    ✔️ Modell A Pipeline definiálva:
       XGBClassifier (skálázás nélkül – fa-alapú modellnél felesleges)
       scale_pos_weight = 0.79  (osztályegyensúly-korrekció, X_train alapján)
    

### 7.3 Modell B Pipeline – RFM + K-Means OHE (FeatureUnion)

A kibővített pipeline két párhuzamos ágból áll (`FeatureUnion`):
1. **`rfm_raw` ág**: az 5 eredeti RFM feature változtatás nélkül.
2. **`cluster_ohe` ág**: csak a 3 alap RFM dimenzió (R, F, M) kerül be → `log1p` transzformáció → `StandardScaler` → `KMeansFeaturizer` → One-Hot kódolt klasztercímke.

A K-Means szándékosan csak az alap R, F, M dimenziókon tanul, hogy a kapott klaszterek megfeleljenek a 02-es notebookban definiált üzleti szegmenseknek (VIP Bajnokok, Lemorzsolódók stb.).


```python
# ============================================================
# 7.3 – Modell B: RFM + K-Means Pipeline (FeatureUnion)
# ============================================================
# A FeatureUnion két párhuzamos ágat futtat egyszerre:
#   - 'rfm_raw': az eredeti (nyers) 5 RFM feature az XGBoost-nak
#                (fa-alapú modellnek nem kell skálázás)
#   - 'cluster_ohe': CSAK a 3 alap RFM feature (recency, frequency, monetary_total)
#                    kerül ide → log1p → StandardScaler → KMeansFeaturizer
#
# KRITIKUS: A K-Means KIZÁRÓLAG a 3 eredeti R, F, M változón tanul,
# ezért az itt képzett klaszterek megfelelnek a 02-es notebookban
# definiált üzleti profiloknak (VIP Bajnokok, Lemorzsolódó, stb.).
# Az XGBoost azonban továbbra is megkapja mind az 5 feature-t a 'rfm_raw' ágon.

from sklearn.compose import ColumnTransformer

# Az oszlopindexek az X DataFrame-ben (FEATURE_COLS sorrendje alapján):
#   0: recency_days | 1: frequency | 2: monetary_total | 3: monetary_avg | 4: return_ratio
RFM_INDICES = [0, 1, 2]   # recency_days, frequency, monetary_total

feature_union = FeatureUnion([
    ('rfm_raw', Pipeline([
        ('passthrough', FunctionTransformer()),   # identity transzformáció (pickle-kompatibilis)
    ])),
    ('cluster_ohe', Pipeline([
        ('rfm_selector', ColumnTransformer(
            [('select_rfm', 'passthrough', RFM_INDICES)],
            remainder='drop'
        )),
        ('log1p',      FunctionTransformer(np.log1p)),
        ('scaler',     StandardScaler()),
        ('featurizer', KMeansFeaturizer(n_clusters=4, random_state=RANDOM_STATE)),
    ])),
])

pipeline_b = Pipeline([
    ('features', feature_union),
    ('clf',      XGBClassifier(**xgb_params))
])

print("✔️ Modell B Pipeline definiálva:")
print("   FeatureUnion[rfm_raw (5 feature) + cluster_ohe (4 OHE feature)] → XGBClassifier")
```

    ✔️ Modell B Pipeline definiálva:
       FeatureUnion[rfm_raw (5 feature) + cluster_ohe (4 OHE feature)] → XGBClassifier
    

## 8. Keresztvalidáció és modellek összehasonlítása

**PR-AUC** (Precision-Recall AUC) a fő metrika – ez az Accuracy-nál és az ROC-AUC-nál is informatívabb, mert nem torzítja a valódi osztályeloszlás, és különösen erős eszköz ritkább pozitív osztálynál. Mellette F1-score és Recall is szerepel a teljes kép kedvéért.

> **Fontos változás a holdout split bevezetésével:**
> A CV mostantól kizárólag `X_train`-en fut (a teljes `X` helyett). Ez biztosítja, hogy az `X_test` valóban érintetlen maradjon – a modellválasztás sem „látja" közvetetten a teszt adatokat.

### 8.1 Keresztvalidáció mindkét modellre


```python
# ============================================================
# 8.1 – Keresztvalidáció mindkét modellre (CSAK X_train-en)
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    'pr_auc':  make_scorer(average_precision_score, response_method='predict_proba'),
    'f1':      make_scorer(f1_score,  zero_division=0),
    'recall':  make_scorer(recall_score, zero_division=0),
}

print("Modell A keresztvalidáció futtatása (5-fold StratifiedKFold, X_train-en)...")
cv_results_a = cross_validate(pipeline_a, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
print("✔️ Modell A kész.")

print("Modell B keresztvalidáció futtatása (5-fold StratifiedKFold, X_train-en)...")
cv_results_b = cross_validate(pipeline_b, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
print("✔️ Modell B kész.")
```

    Modell A keresztvalidáció futtatása (5-fold StratifiedKFold, X_train-en)...
    ✔️ Modell A kész.
    Modell B keresztvalidáció futtatása (5-fold StratifiedKFold, X_train-en)...
    ✔️ Modell B kész.
    

### 8.2 Eredmények összehasonlítása


```python
# ============================================================
# 8.2 – Eredmények összehasonlítása
# ============================================================

def cv_summary(results, name):
    return {
        'Modell': name,
        'PR-AUC (átlag)':  results['test_pr_auc'].mean().round(4),
        'PR-AUC (szórás)': results['test_pr_auc'].std().round(4),
        'F1 (átlag)':      results['test_f1'].mean().round(4),
        'F1 (szórás)':     results['test_f1'].std().round(4),
        'Recall (átlag)':  results['test_recall'].mean().round(4),
        'Recall (szórás)': results['test_recall'].std().round(4),
    }

comparison_df = pd.DataFrame([
    cv_summary(cv_results_a, 'A: Csak RFM'),
    cv_summary(cv_results_b, 'B: RFM + K-Means OHE'),
])

print("Keresztvalidációs eredmények összehasonlítása (X_train 5-fold CV):")
display(comparison_df.set_index('Modell'))

# A nyertes modell automatikus kiválasztása PR-AUC alapján
best_pr_auc_a = cv_results_a['test_pr_auc'].mean()
best_pr_auc_b = cv_results_b['test_pr_auc'].mean()

if best_pr_auc_b > best_pr_auc_a:
    winner_pipeline = pipeline_b
    winner_name     = 'B: RFM + K-Means OHE'
    cv_winner       = cv_results_b
    print(f"\n🏆 Nyertes modell: {winner_name} (PR-AUC: {best_pr_auc_b:.4f} vs {best_pr_auc_a:.4f})")
else:
    winner_pipeline = pipeline_a
    winner_name     = 'A: Csak RFM'
    cv_winner       = cv_results_a
    print(f"\n🏆 Nyertes modell: {winner_name} (PR-AUC: {best_pr_auc_a:.4f} vs {best_pr_auc_b:.4f})")
```

    Keresztvalidációs eredmények összehasonlítása (X_train 5-fold CV):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PR-AUC (átlag)</th>
      <th>PR-AUC (szórás)</th>
      <th>F1 (átlag)</th>
      <th>F1 (szórás)</th>
      <th>Recall (átlag)</th>
      <th>Recall (szórás)</th>
    </tr>
    <tr>
      <th>Modell</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A: Csak RFM</th>
      <td>0.8098</td>
      <td>0.0217</td>
      <td>0.740</td>
      <td>0.0152</td>
      <td>0.7339</td>
      <td>0.0317</td>
    </tr>
    <tr>
      <th>B: RFM + K-Means OHE</th>
      <td>0.8098</td>
      <td>0.0221</td>
      <td>0.743</td>
      <td>0.0151</td>
      <td>0.7377</td>
      <td>0.0299</td>
    </tr>
  </tbody>
</table>
</div>


    
    🏆 Nyertes modell: A: Csak RFM (PR-AUC: 0.8098 vs 0.8098)
    

> **🔍 Modellválasztás értelmezése – Occam borotvája érvényesül**
>
> A két modell CV PR-AUC-ja negyedik tizedesjegyig azonos (0.8098), és az automatikus kiválasztás a **nyers, kerekítetlen** értékek alapján dönt – az A modell marginálisan jobb, ezért nyert.
>
> Ez önmagában is fontos eredmény: a B modell **9 feature-rel** (5 RFM + 4 klaszter OHE) pontosan ugyanannyit teljesít, mint az A **5 feature-rel**. A K-Means szegmentáció tehát nem ad hozzá prediktív erőt a churnhöz – a `recency_days`, `frequency` és `monetary_total` önmagukban megragadják azt az információt, amit a klasztertagság is kódolna.
>
> Összetettebb modell azonos teljesítménnyel = felesleges komplexitás. Az **egyszerűbb, azonos erejű modell** a helyes választás – ezt erősíti meg a 04-es notebook SHAP elemzése is, ahol a klaszter feature-ök (`cluster_0–3`) elhanyagolható SHAP értékekkel a lista alján szerepelnek.

### 8.3 CV eredmények vizualizálása

Minden doboz egy modell 5-fold keresztvalidációs eredményeinek eloszlását mutatja:
- A **doboz közepe (narancssárga vonal)** a medián score.
- A **doboz szélessége (IQR)** a stabilitást jelzi: minél keskenyebb, annál konzisztensebb.
- A **bajusz (whisker)** a min/max értékeket mutatja, a **körök** kiugró foldokat jelölnek.


```python
# ============================================================
# 8.3 CV eredmények vizualizálása
# ============================================================

metrics = ['test_pr_auc', 'test_f1', 'test_recall']
metric_labels = ['PR-AUC', 'F1-Score', 'Recall']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Keresztvalidációs metrikák összehasonlítása (5-fold)', fontsize=14)

for ax, metric, label in zip(axes, metrics, metric_labels):
    data = [cv_results_a[metric], cv_results_b[metric]]
    bp = ax.boxplot(data, tick_labels=['A: Csak RFM', 'B: RFM+KMeans'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#56B4E9')
    bp['boxes'][1].set_facecolor('#009E73')
    ax.set_title(label)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](images/03_churn_prediction/03_8.3_CV_eredmények_vizualizálása.png)
    


### 8.4 A nyertes modell betanítása és kiértékelése (Train vs. Teszt)

A nyertes modellt `X_train`-en tanítjuk be, majd **egyetlen alkalommal** megnyitjuk az `X_test`-et a valódi generalizáció mérésére. Ez az egyetlen pont, ahol a teszt szett „látható" lesz.

A tréning metrikák (X_train-en mért eredmények) természetesen optimistábbak lesznek – ez normális. Az igazi kérdés az, hogy **mekkora a rés** a tréning és a teszt teljesítmény között. Ezt a következő cellában (8.5 Overfitting Monitoring) vizsgáljuk meg részletesen.


```python
# ============================================================
# 8.4 – A nyertes modell betanítása és kiértékelése
# ============================================================

print(f"A nyertes modell ({winner_name}) betanítása X_train-en...")
winner_pipeline.fit(X_train, y_train)
print("✔️ Betanítás kész.\n")

# --- Tréning metrikák (X_train – tájékoztató jellegű) ---
y_train_proba = winner_pipeline.predict_proba(X_train)[:, 1]
y_train_pred  = winner_pipeline.predict(X_train)

train_pr_auc = average_precision_score(y_train, y_train_proba)
train_f1     = f1_score(y_train, y_train_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred, zero_division=0)

# --- Teszt metrikák (X_test – a valódi generalizáció mértéke) ---
y_test_proba = winner_pipeline.predict_proba(X_test)[:, 1]
y_test_pred  = winner_pipeline.predict(X_test)

test_pr_auc = average_precision_score(y_test, y_test_proba)
test_f1     = f1_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)

# --- CV átlagok kinyerése az összehasonlításhoz ---
cv_pr_auc = cv_winner['test_pr_auc'].mean()
cv_f1     = cv_winner['test_f1'].mean()
cv_recall = cv_winner['test_recall'].mean()

print(f"  Tréning  PR-AUC: {train_pr_auc:.4f}  |  F1: {train_f1:.4f}  |  Recall: {train_recall:.4f}")
print(f"  CV átlag PR-AUC: {cv_pr_auc:.4f}  |  F1: {cv_f1:.4f}  |  Recall: {cv_recall:.4f}")
print(f"  Teszt    PR-AUC: {test_pr_auc:.4f}  |  F1: {test_f1:.4f}  |  Recall: {test_recall:.4f}")
```

    A nyertes modell (A: Csak RFM) betanítása X_train-en...
    ✔️ Betanítás kész.
    
      Tréning  PR-AUC: 0.8830  |  F1: 0.7991  |  Recall: 0.7933
      CV átlag PR-AUC: 0.8098  |  F1: 0.7400  |  Recall: 0.7339
      Teszt    PR-AUC: 0.8326  |  F1: 0.7645  |  Recall: 0.7573
    

### 8.5 Overfitting monitoring

Az overfitting monitoring megmutatja, hogy mekkora a rés a tréning, a CV és a teszt metrikák között. Ez a három szám együtt ad megbízható képet a modell valódi állapotáról:

| Szituáció | Tréning | CV | Teszt | Értelmezés |
|---|---|---|---|---|
| ✅ Egészséges | magas | közel azonos | közel azonos | A modell jól általánosít |
| ⚠️ Enyhe overfitting | magas | kissé alacsonyabb | CV-vel egyező | CV még jó becslés, teszt megerősíti |
| 🚨 Súlyos overfitting | magas | lényegesen alacsonyabb | még alacsonyabb | A modell a tréningadatot memorizálta |

**Ökölszabály:** Ha a tréning és a CV/teszt PR-AUC közötti rés meghaladja a **0.05-öt**, érdemes regularizációt erősíteni (pl. `max_depth` csökkentése, `min_child_weight` növelése, `reg_alpha`/`reg_lambda` beállítása).


```python
# ============================================================
# 8.5 – Overfitting monitoring
# ============================================================

monitoring_df = pd.DataFrame({
    'Metrika': ['PR-AUC', 'F1-Score', 'Recall'],
    'Tréning (X_train)': [round(train_pr_auc, 4), round(train_f1, 4), round(train_recall, 4)],
    'CV átlag (X_train)': [round(cv_pr_auc, 4), round(cv_f1, 4), round(cv_recall, 4)],
    'Teszt (X_test)':    [round(test_pr_auc, 4), round(test_f1, 4), round(test_recall, 4)],
    'Rés (Train−Teszt)': [
        round(train_pr_auc - test_pr_auc, 4),
        round(train_f1     - test_f1, 4),
        round(train_recall - test_recall, 4),
    ]
}).set_index('Metrika')

display(monitoring_df)

# --- Automatikus diagnózis ---
pr_auc_gap = train_pr_auc - test_pr_auc
cv_test_gap = abs(cv_pr_auc - test_pr_auc)

print("\n📊 Diagnózis:")
if pr_auc_gap < 0.03:
    print(f"  ✅ Minimális overfitting: Train−Teszt rés = {pr_auc_gap:.4f}")
elif pr_auc_gap < 0.05:
    print(f"  ⚠️  Enyhe overfitting: Train−Teszt rés = {pr_auc_gap:.4f} (elfogadható határon belül)")
else:
    print(f"  🚨 Figyelemfelhívó overfitting: Train−Teszt rés = {pr_auc_gap:.4f}")
    print("     → Javasolt: max_depth csökkentése, reg_alpha / reg_lambda növelése,")
    print("       vagy min_child_weight emelése a RandomizedSearchCV keresési terében.")

if cv_test_gap < 0.02:
    print(f"  ✅ A CV jól becsülte a teszt teljesítményt (CV−Teszt rés = {cv_test_gap:.4f})")
else:
    print(f"  ℹ️  CV−Teszt rés = {cv_test_gap:.4f} – a CV kis mértékben optimistább volt a tesztnél")
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tréning (X_train)</th>
      <th>CV átlag (X_train)</th>
      <th>Teszt (X_test)</th>
      <th>Rés (Train−Teszt)</th>
    </tr>
    <tr>
      <th>Metrika</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PR-AUC</th>
      <td>0.8830</td>
      <td>0.8098</td>
      <td>0.8326</td>
      <td>0.0504</td>
    </tr>
    <tr>
      <th>F1-Score</th>
      <td>0.7991</td>
      <td>0.7400</td>
      <td>0.7645</td>
      <td>0.0347</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.7933</td>
      <td>0.7339</td>
      <td>0.7573</td>
      <td>0.0361</td>
    </tr>
  </tbody>
</table>
</div>


    
    📊 Diagnózis:
      🚨 Figyelemfelhívó overfitting: Train−Teszt rés = 0.0504
         → Javasolt: max_depth csökkentése, reg_alpha / reg_lambda növelése,
           vagy min_child_weight emelése a RandomizedSearchCV keresési terében.
      ℹ️  CV−Teszt rés = 0.0227 – a CV kis mértékben optimistább volt a tesztnél
    

### 8.6 Hiperparaméter-hangolás `RandomizedSearchCV` segítségével *(opcionális)*

Az alábbi cella futtatható, de **nem kötelező** a notebook többi részéhez – az eredmény automatikusan felülírja a `winner_pipeline`-t, ha a keresés jobb paramétert talál.

> **Futtatási idő:** ~5–15 perc CPU-n (100 iteráció × 3-fold CV). Gyorsításhoz csökkentsd `n_iter`-t (pl. 20-ra).

**Fontos:** A keresés itt is `X_train`-en fut. Az `X_test` továbbra is érintetlen marad.

**Miért `RandomizedSearchCV` és nem `GridSearchCV`?**
A keresési tér nagy (~10k kombináció), és a randomizált keresés általában közel azonos eredményt ad töredéknyi idő alatt (*Bergstra & Bengio, 2012*).

> 🚀 **Következő szint:** Ha a pontosság maximalizálása a cél az idő drasztikus növelése nélkül, érdemes **Bayesi optimalizációra** (pl. **Optuna**) váltani, amely tanul a korábbi iterációkból.

> 📌 **Megjegyzés az overfitting monitoringhoz:** A 8.5-ös táblázat az RSCV **előtti** állapotot rögzíti, ez szándékos pillanatkép, amely megmutatja, miért indokolt a hangolás. Ha az RSCV lefut és jobb modellt talál, a végleges Train−Teszt rés a **9.2-es összefoglalóban** olvasható, amely már a hangolt modell értékeit tükrözi.


```python
# ============================================================
# 8.6 Hiperparaméter-hangolás RandomizedSearchCV-vel (opcionális)
# ============================================================
# OPCIONÁLIS CELLA – a notebook többi része a 8.4-ben betanított
# winner_pipeline-t használja, de ez a cella felülírja, ha jobb
# paramétert talál.
#
# A keresési tér az XGBClassifier paramétereire vonatkozik;
# a Pipeline-ban 'clf__' előtaggal hivatkozunk rájuk.
# ============================================================

from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import copy

# --- Keresési tér ---
param_dist = {
    'clf__n_estimators':     stats.randint(100, 600),
    'clf__max_depth':        stats.randint(3, 8),
    'clf__learning_rate':    stats.loguniform(0.01, 0.3),
    'clf__subsample':        stats.uniform(0.6, 0.4),        # 0.6 – 1.0
    'clf__colsample_bytree': stats.uniform(0.5, 0.5),        # 0.5 – 1.0
    'clf__min_child_weight': stats.randint(1, 15),
    'clf__gamma':            stats.loguniform(1e-4, 1.0),
    'clf__reg_alpha':        stats.loguniform(1e-4, 10.0),
    'clf__reg_lambda':       stats.loguniform(0.5, 10.0),
}

search_pipeline = copy.deepcopy(winner_pipeline)
cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rscv = RandomizedSearchCV(
    estimator           = search_pipeline,
    param_distributions = param_dist,
    n_iter              = 100,          # ← csökkentsd 20-ra gyors teszthez
    scoring             = 'average_precision',
    cv                  = cv_search,
    n_jobs              = -1,
    random_state        = RANDOM_STATE,
    verbose             = 1,
    refit               = True,
)

print("RandomizedSearchCV futtatása X_train-en (ez eltarthat néhány percig)...")
rscv.fit(X_train, y_train)

print(f"\n✔️ Keresés kész.")
print(f"   Legjobb CV PR-AUC: {rscv.best_score_:.4f}")
print(f"   Legjobb paraméterek: {rscv.best_params_}")

# Ha a keresés jobb paramétert talált, frissítjük a winner_pipeline-t
if rscv.best_score_ > cv_winner['test_pr_auc'].mean():
    winner_pipeline = rscv.best_estimator_
    winner_name     = winner_name + ' (RSCV hangolt)'
    print(f"\n✅ A keresés javított a modellen → winner_pipeline frissítve.")
    
    # Teszt metrikák frissítése a hangolt modellel
    y_test_proba = winner_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred  = winner_pipeline.predict(X_test)
    test_pr_auc  = average_precision_score(y_test, y_test_proba)
    test_f1      = f1_score(y_test, y_test_pred, zero_division=0)
    test_recall  = recall_score(y_test, y_test_pred, zero_division=0)
    print(f"   Hangolt modell teszt PR-AUC: {test_pr_auc:.4f}")
else:
    print("\nℹ️  A keresés nem javított érdemben – az eredeti winner_pipeline marad.")
```

    RandomizedSearchCV futtatása X_train-en (ez eltarthat néhány percig)...
    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    
    ✔️ Keresés kész.
       Legjobb CV PR-AUC: 0.8121
       Legjobb paraméterek: {'clf__colsample_bytree': np.float64(0.872719737092165), 'clf__gamma': np.float64(0.005328907538812302), 'clf__learning_rate': np.float64(0.015433043380607947), 'clf__max_depth': 3, 'clf__min_child_weight': 5, 'clf__n_estimators': 105, 'clf__reg_alpha': np.float64(3.149466445496976), 'clf__reg_lambda': np.float64(3.08615384985409), 'clf__subsample': np.float64(0.6977412621858619)}
    
    ✅ A keresés javított a modellen → winner_pipeline frissítve.
       Hangolt modell teszt PR-AUC: 0.8253
    

## 9. Végleges modell exportja

### 9.1 Végleges modell betanítása és mentése

A végleges modellt a **teljes adathalmazon** (`X`, `y`) tanítjuk be. Ez az iparági best practice:
1. A holdout szett betöltötte szerepét – overfitting-monitoring ✅
2. Most, hogy tudjuk, a modell jól általánosít, az összes elérhető adat bevonásával tanítunk, hogy a legjobb prediktív teljesítményt kapjuk éles környezetben.

A `clone()` garantálja, hogy a végleges modell friss illesztéssel indul, nem a `X_train`-en betanított állapotból.


```python
# ============================================================
# 9.1 – Végleges modell betanítása a TELJES adathalmazon és mentése
# ============================================================
import pickle
import types

def _patch_lambda_transformers(pipeline):
    """
    A memóriában lévő pipeline-ban kicseréli a lambda-alapú FunctionTransformer-eket
    pickle-kompatibilis FunctionTransformer()-re (identity).
    """
    for name, step in pipeline.steps:
        if hasattr(step, 'transformer_list'):  # FeatureUnion
            for _, sub_pipe in step.transformer_list:
                if hasattr(sub_pipe, 'steps'):
                    for sub_name, sub_step in sub_pipe.steps:
                        if (
                            isinstance(sub_step, FunctionTransformer)
                            and isinstance(getattr(sub_step, 'func', None), types.LambdaType)
                        ):
                            print(f"   ⚠️  Lambda detektálva: '{sub_name}' lépésben → kicserélve identity-re")
                            sub_step.func = None
    return pipeline

MODEL_PATH = MODELS_DIR / "xgboost_churn.joblib"

# Végleges modell klónozása és betanítása az ÖSSZES adaton
final_pipeline = clone(winner_pipeline)
print(f"Végleges modell ({winner_name}) betanítása az összes adaton (X + y)...")
final_pipeline.fit(X, y)
print("✔️ Betanítás kész.")

# Pickle-kompatibilitás ellenőrzés és automatikus javítás
try:
    pickle.dumps(final_pipeline)
except Exception:
    print("⚠️  Lambda FunctionTransformer detektálva – automatikus javítás...")
    final_pipeline = _patch_lambda_transformers(final_pipeline)

joblib.dump(final_pipeline, MODEL_PATH)
print(f"✔️ Végleges Pipeline mentve: {MODEL_PATH}")
print(f"   Fájlméret: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
```

    Végleges modell (A: Csak RFM (RSCV hangolt)) betanítása az összes adaton (X + y)...
    ✔️ Betanítás kész.
    ✔️ Végleges Pipeline mentve: D:\Workspace\ecommerce-customer-segmentation\models\xgboost_churn.joblib
       Fájlméret: 120.2 KB
    

### 9.2 Holdout teszt szett mentése a 04-es notebookhoz

A teszt szett exportjára azért van szükség, hogy a `04_model_evaluation.ipynb` **önállóan** futtatható legyen: nem kell újra végigfuttatni az egész modellezési pipeline-t.

A parquet fájl tartalmazza az `X_test` feature-öket és a `churn` célváltozót – pontosan azt, amire a kalibrációs elemzéshez és a threshold optimalizáláshoz szükség lesz.

A szummáró táblázat az overfitting rést helyesen `average_precision_score`-ral számítja (nem a nyers valószínűségek átlagával).


```python
# ============================================================
# 9.2 – Holdout teszt szett mentése a 04-es notebookhoz
# ============================================================
# A 04_model_evaluation.ipynb a test szettet fogja használni:
#   - kalibrációs elemzéshez (reliability diagram)
#   - threshold optimalizáláshoz
#
# A test szettet itt mentjük el, hogy a 04-es notebook
# függetlenül betölthesse, anélkül hogy újra futtatná az egész pipeline-t.

TEST_SET_PATH = PROCESSED_DIR / "test_set.parquet"

test_export = X_test.copy()
test_export['churn'] = y_test.values
test_export.to_parquet(TEST_SET_PATH, compression='snappy')

print(f"✔️ Holdout teszt szett mentve: {TEST_SET_PATH}")
print(f"   Dimenziók: {test_export.shape[0]:,} ügyfél × {test_export.shape[1]} oszlop")
print(f"   Churn arány a teszt szettben: {y_test.mean()*100:.1f}%")

print("\n" + "="*60)
print("03_churn_prediction.ipynb – KÉSZ")
print("="*60)
print(f"  Nyertes modell:          {winner_name}")
print(f"  CV PR-AUC (átlag):       {cv_winner['test_pr_auc'].mean():.4f}")
print(f"  Teszt PR-AUC:            {test_pr_auc:.4f}")
# FONTOS: a végleges train PR-AUC-t average_precision_score-ral számítjuk,
# NEM a nyers valószínűségek átlagával (az egy teljesen más szám lenne).
_final_train_proba   = winner_pipeline.predict_proba(X_train)[:, 1]
_final_train_pr_auc  = average_precision_score(y_train, _final_train_proba)
print(f"  Overfitting rés:         {(_final_train_pr_auc - test_pr_auc):.4f}  (train−teszt)")
print(f"  Végleges modell mentve:  {MODEL_PATH}")
print(f"  Teszt szett mentve:      {TEST_SET_PATH}")
print("="*60)
print("\n→ Folytatás: 04_model_evaluation.ipynb")
print("  (SHAP, kalibráció, feature importance, threshold optimalizálás, akciólista)")
```

    ✔️ Holdout teszt szett mentve: D:\Workspace\ecommerce-customer-segmentation\data\processed\test_set.parquet
       Dimenziók: 1,049 ügyfél × 6 oszlop
       Churn arány a teszt szettben: 55.8%
    
    ============================================================
    03_churn_prediction.ipynb – KÉSZ
    ============================================================
      Nyertes modell:          A: Csak RFM (RSCV hangolt)
      CV PR-AUC (átlag):       0.8098
      Teszt PR-AUC:            0.8253
      Overfitting rés:         0.0017  (train−teszt)
      Végleges modell mentve:  D:\Workspace\ecommerce-customer-segmentation\models\xgboost_churn.joblib
      Teszt szett mentve:      D:\Workspace\ecommerce-customer-segmentation\data\processed\test_set.parquet
    ============================================================
    
    → Folytatás: 04_model_evaluation.ipynb
      (SHAP, kalibráció, feature importance, threshold optimalizálás, akciólista)
    

<div align="center">
  <br>
  <a href="#teteje">
    <img src="https://img.shields.io/badge/%E2%AC%86%20Vissza%20a%20tetej%C3%A9re-c0253f?style=for-the-badge" alt="Vissza a tetejére" width="250">
  </a>
  <br>
</div>

*Az ugrás gomb nem minden környezetben működik!

# Dokumentáció frissítése README.md-ben és docs mappában
🚨 **Figyelmeztetés:** <kbd>Ctrl</kbd> + <kbd>S</kbd> / <kbd>Cmd ⌘</kbd> + <kbd>S</kbd> szükséges az alábbi cella futtatása előtt!  
(Az nbconvert a fájl utolsó elmentett állapotát olvassa a lemezről, nem az aktuális memóriaképet.)


```python
# 03-as notebook docs generálása/frissítése
!python update_docs.py --notebook 03_churn_prediction.ipynb
```

    Docs frissitese...
    ==================================================
    [03_churn_prediction.ipynb] Konvertalas Markdown-ra...
    [03_churn_prediction.ipynb] [OK] Kesz! (0 kep)
    
    [README] Elemzés főbb lépései táblázat frissítése...
    [README] Táblázat frissítve: 15 sor, 1 csere.
    
    ==================================================
    Kesz!
    
