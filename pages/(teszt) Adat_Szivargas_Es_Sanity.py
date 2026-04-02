import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

st.set_page_config(page_title="Sanity Checks", layout="wide")
st.title("🕵️ Adatszivárgás és Tisztítási Validáció")

@st.cache_data
def load_data():
    if config.READY_FOR_RFM_PARQUET.exists():
        return pd.read_parquet(config.READY_FOR_RFM_PARQUET)
    return None

df = load_data()

if df is not None:
    st.header("1. Adatszivárgás (Data Leakage) Ellenőrzése")
    st.write(f"A konfigurált `CUTOFF_DATE` értéke: **{config.CUTOFF_DATE}**")
    
    # Ellenőrizzük, van-e tranzakció a cutoff után (RFM adathalmazban ez tilos!)
    if 'InvoiceDate' in df.columns:
        max_date = df['InvoiceDate'].max()
        st.write(f"Az adatbázisban lévő legkésőbbi dátum: **{max_date}**")
        
        cutoff = pd.to_datetime(config.CUTOFF_DATE)
        leakage_df = df[df['InvoiceDate'] > cutoff]
        
        if len(leakage_df) > 0:
            st.error(f"🚨 ADATSZIVÁRGÁS! {len(leakage_df)} tranzakció a CUTOFF_DATE után történt, de benne van a tréning adatban!")
            st.dataframe(leakage_df.head())
        else:
            st.success("✅ Nincs adatszivárgás a Cutoff dátumhoz képest.")

        # Tranzakciók időbeli eloszlása
        fig_time = px.histogram(df, x='InvoiceDate', title="Tranzakciók időbeli eloszlása (Bizonyíték a Cutoffra)")
        fig_time.add_vline(x=cutoff.timestamp() * 1000, line_dash="dash", line_color="red", annotation_text="CUTOFF DATE")
        st.plotly_chart(fig_time, use_container_width=True)

    st.header("2. Alapvető Üzleti Logika (Sanity Check)")
    st.write("Maradtak-e negatív mennyiségek (visszatérítések hibás kezelése) vagy extrém árak?")
    
    col1, col2 = st.columns(2)
    if 'Quantity' in df.columns:
        negative_qty = df[df['Quantity'] <= 0]
        col1.metric("<=0 Mennyiségű sorok", len(negative_qty))
        fig_qty = px.box(df, y='Quantity', title="Mennyiségek eloszlása (Outlier kereső)")
        col1.plotly_chart(fig_qty)
        if len(negative_qty) > 0:
            col1.error("🚨 Hibás tisztítás: negatív/nulla mennyiség maradt az adatban!")

    if 'UnitPrice' in df.columns or 'Price' in df.columns:
        price_col = 'UnitPrice' if 'UnitPrice' in df.columns else 'Price'
        negative_price = df[df[price_col] < 0]
        col2.metric("<0 Egységárú sorok", len(negative_price))
        fig_price = px.box(df, y=price_col, title="Egységárak eloszlása (Outlier kereső)")
        col2.plotly_chart(fig_price)
        if len(negative_price) > 0:
            col2.error("🚨 Hibás tisztítás: negatív árak maradtak!")
else:
    st.warning("Nem található a READY_FOR_RFM_PARQUET fájl.")