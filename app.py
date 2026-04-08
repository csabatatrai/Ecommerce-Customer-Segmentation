import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Oldal konfigurációja
# ==========================================
st.set_page_config(
    page_title="Executive Summary | Ügyfélmegtartás",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Adatbetöltés (Két fájlra van szükségünk)
# ==========================================
@st.cache_data
def load_data():
    preds_path = Path("data/processed/churn_predictions.parquet")
    tx_path = Path("data/processed/online_retail_ready_for_rfm.parquet")
    
    if preds_path.exists() and tx_path.exists():
        preds = pd.read_parquet(preds_path)
        tx_data = pd.read_parquet(tx_path)
        
        # Biztosítjuk, hogy a tranzakciós dátum megfelelő formátumú legyen
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
        
        # JAVÍTÁS: Ha a Customer ID az indexben van (RFM aggregáció miatt), 
        # akkor oszloppá alakítjuk, hogy a KeyError-t elkerüljük!
        if 'Customer ID' not in preds.columns:
            preds = preds.reset_index()
            # Biztosítjuk a helyes oszlopnevet
            if 'index' in preds.columns:
                preds = preds.rename(columns={'index': 'Customer ID'})
            elif 'CustomerID' in preds.columns:
                preds = preds.rename(columns={'CustomerID': 'Customer ID'})
                
        return preds, tx_data
    else:
        st.error("Hiba: Hiányzó adatfájlok a data/processed/ mappában!")
        return pd.DataFrame(), pd.DataFrame()

df_preds, df_tx = load_data()

# ==========================================
# 3. Dashboard fejléc
# ==========================================
st.title("Executive Summary")
st.markdown("---")

if not df_preds.empty and not df_tx.empty:
# ==========================================
    # 4. KPI kiszámítása (Szcenárió alapú Tól-Ig sáv)
    # ==========================================
    vip_at_risk_mask = df_preds['action'].str.contains('VIP Veszélyben', na=False)
    vip_at_risk_ids = df_preds.loc[vip_at_risk_mask, 'Customer ID']
    
    vip_tx = df_tx[df_tx['Customer ID'].isin(vip_at_risk_ids)]
    
    cutoff_date = pd.to_datetime("2011-09-09")
    
    # 1. forgatókönyv: Utolsó aktív 365 nap (TTM)
    start_date_ttm = cutoff_date - pd.Timedelta(days=365)
    recent_vip_tx = vip_tx[(vip_tx['InvoiceDate'] >= start_date_ttm) & (vip_tx['InvoiceDate'] < cutoff_date)]
    revenue_ttm = recent_vip_tx['LineTotal'].sum()
    
    # 2. forgatókönyv: Teljes élettartamra vetített átlag (Lifetime Annualized)
    # A teljes megfigyelési ablak hossza (2009-12-01-től a cutoff-ig)
    total_observation_days = (cutoff_date - df_tx['InvoiceDate'].min()).days
    lifetime_revenue = vip_tx[(vip_tx['InvoiceDate'] < cutoff_date)]['LineTotal'].sum()
    revenue_lifetime_avg = (lifetime_revenue / total_observation_days) * 365
    
    # Sáv határainak meghatározása (alsó és felső határ)
    lower_bound = min(revenue_ttm, revenue_lifetime_avg)
    upper_bound = max(revenue_ttm, revenue_lifetime_avg)
    
    # ==========================================
    # 5. KPI Megjelenítése és Transzparencia
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚨 Becsült veszélyben lévő éves bevétel", 
            value=f"£ {lower_bound/1000:,.0f}k - {upper_bound/1000:,.0f}k"
        )
        
        latex_formula_ttm = r"$$ \text{Felső/Alsó határ}_A = \sum_{t = \text{Cutoff} - 365}^{\text{Cutoff}} \text{LineTotal}_t $$"
        latex_formula_lt = r"$$ \text{Felső/Alsó határ}_B = \left( \frac{\sum_{t = \text{Start}}^{\text{Cutoff}} \text{LineTotal}_t}{\text{Összes Nap}} \right) \times 365 $$"
        
        with st.expander("ℹ️ Hogyan számoltuk?"):
            st.markdown(f"""
            **Célcsoport:** A **'🚨 VIP Veszélyben'** szegmens.
            
            A bizonytalanság transzparens kommunikálása érdekében egy konzervatív és egy optimista forgatókönyvből képeztünk sávot:
            
            **1. Forgatókönyv (Utolsó 365 nap futásideje):**
            Az ügyfelek lemorzsolódás előtti legfrissebb (vágási pont előtti 1 év) teljesítménye.
            {latex_formula_ttm}
            Eredmény: **£ {revenue_ttm:,.0f}**
            
            **2. Forgatókönyv (Élettartam-átlag évesítve):**
            Az ügyfelek teljes megfigyelési időszakában ({total_observation_days} nap) mutatott átlagos költési hajlandósága évesítve.
            {latex_formula_lt}
            Eredmény: **£ {revenue_lifetime_avg:,.0f}**
            
            *Megjegyzés: A sáv megjelenítése 'ezer font' (k) formátumban történik a könnyebb olvashatóságért. A legpontosabb előrejelzéshez jövőbeli fejlesztésként Prediktív CLV modell bevezetése javasolt.*
            """)