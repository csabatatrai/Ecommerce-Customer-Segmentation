-- ============================================================
-- eda_exploratory_analysis.sql
-- Projekt: ecommerce-customer-segmentation
-- Eszköz:  SQLite (DB Browser for SQLite)
-- Tábla:   ecomStore (a nyers Online Retail II CSV importálva)
--
-- Cél: Az adathalmaz szerkezetének, minőségének és üzleti
-- mintázatainak feltárása a Python-alapú pipeline megtervezése előtt.
--
-- Az itt azonosított problémák (adminisztratív StockCode-ok,
-- visszaküldési arányok, ügyfélszintű torzítások) közvetlenül
-- megalapozták az 01_data_preparation.ipynb tisztítási logikáját.
-- Az RFM szegmentáció SQL-prototípusa (ld. lent) megmutatta,
-- hogy a kvintilis-alapú megközelítés helyett adatvezérelt
-- K-means klaszterezés szükséges (→ 02_customer_segmentation.ipynb).
-- ============================================================


-- ============================================================
-- 1. ÜZLETI FELTÁRÁS
-- ============================================================

-- 1.1 Havi bevétel trend és szezonalitás
-- Kérdés: Van-e szezonális mintázat, ami befolyásolja az RFM
-- observation window megválasztását?
SELECT 
    strftime('%Y-%m', InvoiceDate) AS SalesMonth,
    SUM(Quantity * Price)          AS TotalRevenue,
    COUNT(DISTINCT Invoice)        AS TotalOrders
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
  AND "Customer ID" IS NOT NULL
GROUP BY SalesMonth
ORDER BY SalesMonth;


-- 1.2 Top 10 legértékesebb vásárló (Customer Lifetime Value – nyers közelítés)
-- Kérdés: Igazolható-e a Pareto-elv (top 20% → 80% bevétel)?
-- Eredmény: Igen – ez megerősítette a CLV-fókuszú megközelítést.
SELECT 
    "Customer ID",
    SUM(Quantity * Price)   AS TotalSpent,
    COUNT(DISTINCT Invoice) AS NumberOfOrders
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
  AND "Customer ID" IS NOT NULL
GROUP BY "Customer ID"
ORDER BY TotalSpent DESC
LIMIT 10;


-- 1.3 Top 10 legtöbb bevételt hozó termék ("Cash Cows")
-- Kérdés: Vannak-e outlier termékek, amik torzíthatják a kosárérték-statisztikákat?
SELECT 
    StockCode,
    Description,
    SUM(Quantity * Price) AS TotalRevenue,
    SUM(Quantity)         AS TotalUnitsSold
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
GROUP BY StockCode, Description
ORDER BY TotalRevenue DESC
LIMIT 10;


-- 1.4 Átlagos kosárérték (AOV – Average Order Value) havonta
-- Kérdés: Stabil-e az AOV, vagy erősen ingadozik? (hatással van a Monetary feature értelmezésére)
SELECT 
    strftime('%Y-%m', InvoiceDate)                    AS SalesMonth,
    SUM(Quantity * Price) / COUNT(DISTINCT Invoice)   AS AverageOrderValue
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
  AND "Customer ID" IS NOT NULL
GROUP BY SalesMonth
ORDER BY SalesMonth;


-- 1.5 Lemorzsolódás kockázata – Recency analízis
-- Kérdés: Mekkora a vásárlók "hallgatási ideje"? Ez indokolta a
-- recency_days feature bevezetését az RFM modellben.
-- Megjegyzés: julianday() az SQLite dátumkülönbség-számítás natív módja.
-- JAVÍTOTT VERZIÓ (1.5-ös query):a julianday('now') a mai dátumot veszi alapul, cserélve lett
SELECT 
    "Customer ID",
    MAX(InvoiceDate) AS LastPurchaseDate,
    CAST(julianday((SELECT MAX(InvoiceDate) FROM ecomStore)) - julianday(MAX(InvoiceDate)) AS INTEGER) AS DaysSinceLastPurchase
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
  AND "Customer ID" IS NOT NULL
GROUP BY "Customer ID"
ORDER BY DaysSinceLastPurchase DESC;


-- 1.6 Termékek visszaküldési aránya
-- Kérdés: Mely termékek visszaküldési aránya kiugróan magas?
-- Eredmény: Ez vezette be a return_count és return_ratio feature-öket
-- az RFM modellbe, mint churn-indikátor.
-- Megjegyzés: IFNULL az SQLite-specifikus COALESCE-megfelelő.
WITH Sales AS (
    SELECT StockCode, Description, SUM(Quantity) AS SoldQty
    FROM ecomStore
    WHERE Invoice NOT LIKE 'C%'
    GROUP BY StockCode, Description
),
Returns AS (
    SELECT StockCode, ABS(SUM(Quantity)) AS ReturnedQty
    FROM ecomStore
    WHERE Invoice LIKE 'C%'
    GROUP BY StockCode
)
SELECT 
    S.StockCode,
    S.Description,
    S.SoldQty,
    IFNULL(R.ReturnedQty, 0)                              AS ReturnedQty,
    ROUND(IFNULL(R.ReturnedQty, 0) * 100.0 / S.SoldQty, 2) AS ReturnRatePercent
FROM Sales S
LEFT JOIN Returns R ON S.StockCode = R.StockCode
WHERE S.SoldQty > 50
ORDER BY ReturnRatePercent DESC
LIMIT 10;


-- 1.7 Országonkénti piaci teljesítmény
-- Kérdés: Mennyire UK-centrikus az adathalmaz? (→ modellezési scope meghatározása)
-- Eredmény: Túlnyomórészt UK – a Country feature nem kerül be az RFM modellbe.
SELECT 
    Country,
    COUNT(DISTINCT "Customer ID") AS UniqueCustomers,
    COUNT(DISTINCT Invoice)       AS TotalOrders,
    SUM(Quantity * Price)         AS TotalRevenue
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
GROUP BY Country
ORDER BY TotalRevenue DESC;


-- 1.8 Új vásárlók száma havonta (akvizíció)
-- Kérdés: Mikor "születtek" az ügyfelek? Fontos a cohort-logika
-- és a time-window megválasztása szempontjából.
WITH FirstPurchases AS (
    SELECT 
        "Customer ID",
        MIN(strftime('%Y-%m', InvoiceDate)) AS CohortMonth
    FROM ecomStore
    WHERE Invoice NOT LIKE 'C%'
      AND "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
)
SELECT 
    CohortMonth,
    COUNT("Customer ID") AS NewCustomers
FROM FirstPurchases
GROUP BY CohortMonth
ORDER BY CohortMonth;


-- 1.9 Leggyakrabban rendelt termékek (Volume Leaders)
-- Kérdés: Melyek a "töltelék" termékek, amik sok kosárban megjelennek,
-- de nem feltétlenül a legnagyobb bevételűek?
SELECT 
    StockCode,
    Description,
    COUNT(DISTINCT Invoice) AS FrequencyInOrders,
    SUM(Quantity)           AS TotalUnitsSold
FROM ecomStore
WHERE Invoice NOT LIKE 'C%'
GROUP BY StockCode, Description
ORDER BY FrequencyInOrders DESC
LIMIT 10;


-- 1.10 Vásárlói lojalitás – rendelési frekvencia eloszlás
-- Kérdés: Mekkora a valóban visszatérő vásárlók aránya?
-- Eredmény: A majority egyszeri vásárló → megerősítette, hogy
-- a frequency feature erős szegmentációs tényező lesz.
WITH CustomerOrders AS (
    SELECT 
        "Customer ID",
        COUNT(DISTINCT Invoice) AS OrderCount
    FROM ecomStore
    WHERE Invoice NOT LIKE 'C%'
      AND "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
)
SELECT 
    CASE 
        WHEN OrderCount = 1            THEN '1 rendelés (Egyszeri)'
        WHEN OrderCount = 2            THEN '2 rendelés'
        WHEN OrderCount BETWEEN 3 AND 5 THEN '3-5 rendelés'
        ELSE                                '5+ rendelés (Lojális)'
    END AS LoyaltySegment,
    COUNT("Customer ID") AS NumberOfCustomers
FROM CustomerOrders
GROUP BY LoyaltySegment
ORDER BY NumberOfCustomers DESC;


-- ============================================================
-- 1.11 Manuális korrekciós bejegyzések azonosítása ("Adjustment by...")
-- Kérdés: Vannak-e személynévhez kötött, manuálisan beírt korrekciós sorok,
-- amelyek nem valódi tranzakciók?
-- Eredmény: Igen – "Adjustment by Peter..." és hasonló leírású sorok kerültek elő,
-- irreális árakkal és mennyiségekkel. Ezek nem üzleti tranzakciók, hanem
-- rendszeroperátori javítások. → Ez vezette be az ADJUSTMENT-szűrőt
-- az 01_data_preparation.ipynb 1.4.2. cellájában.
SELECT
    Invoice,
    StockCode,
    Description,
    Quantity,
    Price,
    "Customer ID",
    Country
FROM ecomStore
WHERE Description LIKE '%adjust%'
   OR Description LIKE '%Adjust%'
   OR Description LIKE '%ADJUST%'
ORDER BY ABS(Quantity * Price) DESC
LIMIT 20;


-- 1.12 Extrém mennyiségű (Quantity) tételek feltérképezése
-- Kérdés: Mekkora a Quantity oszlop valódi terjedelme?
-- Melyek a legnagyobb pozitív és negatív értékek, és mi a mögöttes ok?
-- Eredmény: +/- 80 000 darabos tételek azonosítva – ezek nem valódi rendelések,
-- hanem azonnal sztornózott rendszerhibák vagy tömeges tesztbejegyzések.
-- Egy ajándéktárgy-nagykereskedésben a pár ezer darab még B2B-szempontból
-- reális, de a 10 000+ kiugrások kivétel nélkül ilyen hibás sorokhoz köthetők.
-- → Ez alapozta meg a Q_THRESHOLD konstans értékét a config.py-ban és az
-- 01_data_preparation.ipynb 1.4.4. cellájának szűrési logikáját.
SELECT
    MAX(Quantity)        AS MaxQuantity,
    MIN(Quantity)        AS MinQuantity,
    AVG(Quantity)        AS AvgQuantity,
    COUNT(*)             AS TotalRows,
    SUM(CASE WHEN ABS(Quantity) > 10000 THEN 1 ELSE 0 END) AS RowsOver10k,
    SUM(CASE WHEN ABS(Quantity) > 50000 THEN 1 ELSE 0 END) AS RowsOver50k
FROM ecomStore;

-- A konkrét outlier sorok (a küszöbérték kalibrálásához):
SELECT
    Invoice,
    StockCode,
    Description,
    Quantity,
    Price,
    ROUND(Quantity * Price, 2) AS LineTotal,
    "Customer ID",
    InvoiceDate
FROM ecomStore
WHERE ABS(Quantity) > 10000
ORDER BY ABS(Quantity) DESC
LIMIT 20;


-- 1.13 Az adathalmaz tényleges időtartományának meghatározása
-- Kérdés: Pontosan mikor kezdődik és végződik az adathalmaz?
-- Ez kritikus a "negatív CLV-jű ügyfél" jelenség megértéséhez:
-- ha egy ügyfél visszaküldése szerepel az adatban, de az eredeti vásárlása
-- az adathalmaz kezdete ELŐTT történt, a nettó egyenlege hamisan negatív lesz.
-- Eredmény: Az adathalmaz 2009. december 1-jén kezdődik – ez az a határ,
-- amely alatt az "árva sztornók" keletkeznek.
-- → Ez indokolta az invalid_customers szűrőt az
-- 01_data_preparation.ipynb 1.4.3. cellájában.
SELECT
    MIN(InvoiceDate) AS DatasetStart,
    MAX(InvoiceDate) AS DatasetEnd,
    CAST(julianday(MAX(InvoiceDate)) - julianday(MIN(InvoiceDate)) AS INTEGER) AS SpanDays
FROM ecomStore;

-- Az "árva sztornók" mintájának szemléltetése:
-- Olyan ügyfelek, akiknek van visszaküldésük (C prefix), de nincs egyetlen
-- pozitív vásárlásuk sem az adathalmazban → az eredeti rendelés 2009 előtt volt.
WITH CustomerSales AS (
    SELECT "Customer ID",
           SUM(CASE WHEN Invoice NOT LIKE 'C%' AND Quantity > 0
                    THEN Quantity * Price ELSE 0 END) AS TotalSales,
           SUM(CASE WHEN Invoice     LIKE 'C%'
                    THEN Quantity * Price ELSE 0 END) AS TotalReturns
    FROM ecomStore
    WHERE "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
)
SELECT
    COUNT(*)                                                      AS TotalCustomers,
    SUM(CASE WHEN TotalSales <= 0 THEN 1 ELSE 0 END)             AS CustomersWithNoSales,
    SUM(CASE WHEN TotalSales + TotalReturns <= 0 THEN 1 ELSE 0 END) AS NegativeNetCustomers
FROM CustomerSales;


-- ============================================================
-- 2. RFM SZEGMENTÁCIÓ – SQL PROTOTÍPUS
-- ============================================================
-- Ez a lekérdezés az RFM logika első, exploratív implementációja
-- volt SQLite-ban, NTILE(5) kvintilis-alapú pontozással.
--
-- Miért nem ez lett a végleges megoldás?
-- A kvintilis-határok statisztikailag önkényesek: az NTILE egyenlő
-- méretű csoportokat képez, de nem veszi figyelembe az eloszlás
-- tényleges alakját (erős jobbra-ferdültség, "bálna" ügyfelek).
-- → Ez motiválta a Python-alapú K-means klaszterezésre való
--   áttérést, ahol az algoritmus adatvezérelten találja meg
--   a természetes csoporthatárokat (→ 02_customer_segmentation.ipynb).
-- ============================================================

WITH Utolso_Nap AS (
    -- Az adathalmaz legutolsó dátuma a Recency referenciapont
    SELECT MAX(InvoiceDate) AS MaxDatum FROM ecomStore
),
RFM_Alap_Adatok AS (
    SELECT 
        "Customer ID",
        CAST(
            JULIANDAY((SELECT MaxDatum FROM Utolso_Nap)) - JULIANDAY(MAX(InvoiceDate))
        AS INTEGER)                          AS Recency_Napok,
        COUNT(DISTINCT Invoice)              AS Frequency_Rendelesek,
        ROUND(SUM(Quantity * Price), 2)      AS Monetary_Koltseg
    FROM ecomStore
    WHERE "Customer ID" IS NOT NULL
      AND Quantity > 0
    GROUP BY "Customer ID"
),
RFM_Pontozas AS (
    -- NTILE(5): 1–5 skálán pontozza az ügyfeleket kvintilisenként.
    -- Recency: kisebb érték = jobb (nemrég vásárolt) → DESC sorrend
    -- Frequency, Monetary: nagyobb = jobb → ASC sorrend
    SELECT 
        "Customer ID",
        Recency_Napok,
        Frequency_Rendelesek,
        Monetary_Koltseg,
        NTILE(5) OVER (ORDER BY Recency_Napok    DESC) AS R_Pont,
        NTILE(5) OVER (ORDER BY Frequency_Rendelesek ASC) AS F_Pont,
        NTILE(5) OVER (ORDER BY Monetary_Koltseg ASC) AS M_Pont
    FROM RFM_Alap_Adatok
)
SELECT 
    "Customer ID",
    Recency_Napok,
    Frequency_Rendelesek,
    Monetary_Koltseg,
    (R_Pont || F_Pont || M_Pont) AS RFM_Cella,
    CASE 
        WHEN R_Pont >= 4 AND F_Pont >= 4 AND M_Pont >= 4 THEN '1. Bajnokok (VIP)'
        WHEN R_Pont >= 4 AND F_Pont <= 2                 THEN '2. Új és ígéretes vevők'
        WHEN R_Pont <= 2 AND F_Pont >= 4 AND M_Pont >= 4 THEN '3. Elvesztett VIP-k (Veszély!)'
        WHEN R_Pont >= 3 AND F_Pont >= 3 AND M_Pont >= 3 THEN '4. Hűséges átlagos vevők'
        WHEN R_Pont <= 2 AND F_Pont <= 2 AND M_Pont <= 2 THEN '5. Lemorzsolódott (Inaktív)'
        ELSE                                                   '6. Egyéb / Átlagos'
    END AS Ugyfel_Szegmens
FROM RFM_Pontozas
ORDER BY Monetary_Koltseg DESC
LIMIT 50;
