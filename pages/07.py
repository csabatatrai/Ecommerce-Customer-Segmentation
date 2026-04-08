import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Oldal beállítása
st.set_page_config(page_title="Egyéni Ügyfél Kockázati Radar", layout="wide")

@st.cache_data
def load_data():
    """Adatok betöltése és előkészítése a notebook specifikációja alapján."""
    file_path = 'data/processed/rfm_features.parquet'
    
    if not os.path.exists(file_path):
        # Mock adatok generálása, ha a fájl még nem létezik (teszteléshez)
        data = {
            'Customer ID': [str(i) for i in range(12345, 12355)],
            'recency_days': [10, 350, 120, 5, 280, 45, 15, 200, 30, 90],
            'frequency': [20, 1, 4, 30, 2, 15, 25, 3, 10, 6],
            'monetary_total': [15000, 500, 4000, 20000, 800, 8000, 12000, 2500, 6000, 3000],
            'return_ratio': [0.01, 0.40, 0.05, 0.02, 0.50, 0.10, 0.01, 0.20, 0.03, 0.08],
            'churn_probability': [0.05, 0.95, 0.45, 0.02, 0.88, 0.15, 0.08, 0.72, 0.25, 0.55]
        }
        return pd.DataFrame(data)
    
    df = pd.read_parquet(file_path)
    
    # JAVÍTÁS: Ha a 'Customer ID' az indexben van, hozzuk ki oszlopba
    if 'Customer ID' not in df.columns:
        df = df.reset_index()
    
    # Típuskonverzió a selectbox miatt
    df['Customer ID'] = df['Customer ID'].astype(str)
    
    # Ha a Churn modell eredménye még nincs benne a fájlban, 
    # a demonstráció kedvéért véletlenszerű pontszámot adunk (valós környezetben ez a modell kimenete)
    if 'churn_probability' not in df.columns:
        np.random.seed(42)
        df['churn_probability'] = np.random.uniform(0, 1, len(df))
        
    return df

def create_radar_chart(customer_id, df):
    """Létrehozza a radardiagramot normalizált értékekkel."""
    
    # Ügyfél adatai
    cust = df[df['Customer ID'] == customer_id].iloc[0]
    
    # KOCKÁZATI LOGIKA: Minden tengelyen a 100% a legmagasabb kockázat.
    # Recency: Minél több nap telt el, annál nagyobb kockázat (Percentilis)
    risk_recency = (df['recency_days'] <= cust['recency_days']).mean()
    
    # Frequency: Minél kevesebbet vásárolt, annál nagyobb kockázat (Fordított percentilis)
    risk_frequency = 1 - (df['frequency'] <= cust['frequency']).mean()
    
    # Monetary: Minél kevesebbet költött, annál nagyobb kockázat (Fordított percentilis)
    risk_monetary = 1 - (df['monetary_total'] <= cust['monetary_total']).mean()
    
    # Churn Valószínűség: Közvetlenül a modell által adott pontszám
    risk_churn = cust['churn_probability']
    
    # Visszaküldési hajlam: Magas arány = magas kockázat
    risk_return = (df['return_ratio'] <= cust['return_ratio']).mean()

    # Adatok a diagramhoz
    categories = [
        'Inaktivitás (Recency)', 
        'Vásárlási Ritkulás (Freq)', 
        'Költés Elmaradása (Mon)',
        'Churn Valószínűség', 
        'Visszaküldési Hajlam'
    ]
    
    values = [risk_recency, risk_frequency, risk_monetary, risk_churn, risk_return]
    # Százalékos formátum (0-100)
    values = [v * 100 for v in values]
    
    # Alakzat bezárása (vissza az első ponthoz)
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Kockázati Profil',
        line=dict(color='#FF4B4B', width=2),
        fillcolor='rgba(255, 75, 75, 0.4)',
        marker=dict(size=8)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                gridcolor="lightgrey"
            ),
            angularaxis=dict(
                gridcolor="lightgrey"
            )
        ),
        showlegend=False,
        title=dict(
            text=f"Risk Radar: Ügyfél #{customer_id}",
            x=0.5,
            font=dict(size=20)
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        height=600
    )
    
    return fig

# UI Felület
st.title("🛡️ Egyéni Ügyfél Kockázati Radar")
st.markdown("Mutatja, hogy egy adott VIP ügyfél miért kapott magas churn-pontszámot a teljes populációhoz képest.")

try:
    df = load_data()
    
    # Kiválasztó sáv
    selected_id = st.selectbox(
        "Válassz ki egy ügyfelet az elemzéshez:",
        options=df['Customer ID'].unique()
    )

    if selected_id:
        # Ügyfél sor kinyerése a metrikákhoz
        cust_row = df[df['Customer ID'] == selected_id].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Ügyfél Mutatók")
            st.metric("Utolsó vásárlás", f"{int(cust_row['recency_days'])} napja")
            st.metric("Vásárlási gyakoriság", f"{int(cust_row['frequency'])} alkalom")
            st.metric("Összes költés", f"£{cust_row['monetary_total']:,.0f}")
            st.metric("Visszaküldési arány", f"{cust_row['return_ratio']:.1%}")
            
            # Churn Score kiemelése
            score = cust_row['churn_probability']
            if score > 0.7:
                st.error(f"Kritikus Churn Kockázat: {score:.1%}")
            elif score > 0.4:
                st.warning(f"Figyelmeztető Churn Kockázat: {score:.1%}")
            else:
                st.success(f"Alacsony Churn Kockázat: {score:.1%}")

        with col2:
            radar_fig = create_radar_chart(selected_id, df)
            st.plotly_chart(radar_fig, use_container_width=True)

except Exception as e:
    st.error(f"Hiba történt az adatok betöltésekor: {e}")
    st.info("Ellenőrizd, hogy a 'data/processed/rfm_features.parquet' fájl létezik-e.")