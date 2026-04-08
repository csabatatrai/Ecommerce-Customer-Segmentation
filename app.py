import streamlit as st
import pandas as pd
import base64
from pathlib import Path

# ==========================================
# 1. Oldal konfigurációja
# ==========================================
st.set_page_config(
    page_title="Vezetői összefoglaló | Ügyfélmegtartás",
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
st.title("Vezetői összefoglaló")
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
    vip_at_risk_count = int(vip_at_risk_mask.sum())
    effective_threshold = float(df_preds.loc[df_preds['churn_pred'] == 1, 'churn_proba'].min())

    # Historikus validáció: TP (valóban elmorzsolódott) vs. FP (aktív maradt)
    vip_df = df_preds[vip_at_risk_mask]
    vip_tp = int((vip_df['actual_churn'] == 1).sum())   # valódi churnerek
    vip_fp = int((vip_df['actual_churn'] == 0).sum())   # téves riasztás
    vip_precision_pct = vip_tp / vip_at_risk_count * 100

    # -------------------------------------------------------
    # Háttérkép betöltése (base64, hogy Streamlit biztosan kiszolgálja)
    # -------------------------------------------------------
    bg_path = Path("assets/streamlit_bg.webp")
    if bg_path.exists():
        bg_b64 = base64.b64encode(bg_path.read_bytes()).decode()
        bg_css = f'background-image: url("data:image/webp;base64,{bg_b64}");'
    else:
        bg_css = ""

    # -------------------------------------------------------
    # Egyéni KPI kártyák CSS stílusa
    # -------------------------------------------------------
    st.markdown(f"""
        <style>
            .stApp {{
                {bg_css}
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.25rem;
                margin-bottom: 1.5rem;
            }}
            .kpi-card {{
                /* #! KÁRTYA HÁTTÉRSZÍN — itt módosítható */
                background: rgba(168, 16, 34, 0.45);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                padding: 1.5rem 1.75rem 1.25rem 1.75rem;
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
            }}
            .kpi-label {{
                font-size: 17px;
                font-weight: 600;
                color: #c8cfe8;
                line-height: 1.4;
                white-space: normal;
            }}
            .kpi-value {{
                font-size: 30px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: -0.5px;
                margin-top: 0.25rem;
            }}
        </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # KPI kártyák HTML renderelése
    # -------------------------------------------------------
    st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">🚨 Becsült veszélyben lévő éves bevétel</div>
                <div class="kpi-value">£ {lower_bound/1000:,.0f}k – {upper_bound/1000:,.0f}k</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">👥 Veszélyben lévő VIP ügyfelek</div>
                <div class="kpi-value">{vip_at_risk_count:,} fő</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # Expander magyarázatok (kártyák alatt, 2 oszlopban)
    # -------------------------------------------------------
    col1, col2 = st.columns(2)

    latex_formula_ttm = r"$$ \text{Felső/Alsó határ}_A = \sum_{t = \text{Cutoff} - 365}^{\text{Cutoff}} \text{LineTotal}_t $$"
    latex_formula_lt  = r"$$ \text{Felső/Alsó határ}_B = \left( \frac{\sum_{t = \text{Start}}^{\text{Cutoff}} \text{LineTotal}_t}{\text{Összes Nap}} \right) \times 365 $$"

    with col1:
        with st.expander("ℹ️ Hogyan számoltuk? – Becsült bevételkiesés"):
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

    with col2:
        with st.expander("ℹ️ Hogyan számoltuk? – VIP ügyfelek száma"):
            st.markdown(f"""
            **Célcsoport:** Magas RFM-értékű ügyfelek, akiknél a modell lemorzsolódást jelez.

            **Kiválasztás feltétele:** Az ügyfél egyidejűleg teljesíti az alábbi két kritériumot:
            1. **VIP szegmens** – a vásárló magas értéket képvisel (Champions / Top szegmens az RFM-besorolás alapján).
            2. **Lemorzsolódás előrejelzett** – a modell által becsült valószínűség átlépi az **optimalizált döntési küszöböt** (`churn_proba ≥ {effective_threshold:.3f}`).

            **A küszöb meghatározása:** Az F1-score-t maximalizáló threshold a Precision–Recall görbéből, kizárólag a holdout teszthalmazon számítva (1 049 ügyfél). Ez a küszöb csökkenti az elszalasztott lemorzsolódókat (FN) a téves riasztások (FP) rovására – üzletileg ez az ésszerű prioritás.

            ---
            **Historikus validáció – mennyire „tiszta" a lista?**

            | | Ügyfelek száma | Arány |
            |---|---|---|
            | ✅ Valódi churner (TP) | {vip_tp:,} fő | {vip_precision_pct:.0f}% |
            | ⚠️ Téves riasztás (FP) | {vip_fp:,} fő | {100 - vip_precision_pct:.0f}% |
            | **Összesen (akció-lista)** | **{vip_at_risk_count:,} fő** | **100%** |

            A lista **{vip_precision_pct:.0f}%-os precizitással** azonosítja a valódi lemorzsolódókat. A fennmaradó {100 - vip_precision_pct:.0f}% (téves riasztás) proaktív megkeresése üzletileg nem kockázatos – egy VIP ügyfelet felhívni soha nem árt.

            *Megjegyzés: Az `actual_churn` kizárólag historikus validációs label. Éles működésben ez az érték nem ismert előre – ezért is van szükség a modellre.*
            """)