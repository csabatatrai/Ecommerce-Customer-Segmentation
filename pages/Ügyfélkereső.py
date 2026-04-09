"""
pages/★Ügyfél_Kereső.py
Egyedi ügyfél RFM-profil és churn-kockázat kereső
"""

import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sidebar import render_sidebar
from data_loader import load_churn_predictions, load_transactions

render_sidebar()

# ── Háttérkép (app.py-val azonos megközelítés) ─────────────────────────────────
bg_path = Path("assets/streamlit_bg.webp")
if bg_path.exists():
    bg_b64 = base64.b64encode(bg_path.read_bytes()).decode()
    bg_css = f'background-image: url("data:image/webp;base64,{bg_b64}");'
else:
    bg_css = ""

# ── Globális stílus (app.py-val harmonizált) ───────────────────────────────────
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
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .kpi-card {{
            background: rgba(168, 16, 34, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 12px;
            padding: 1.1rem 1.25rem 1rem 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        }}
        .kpi-card-accent {{
            background: rgba(168, 16, 34, 0.65);
            border: 2px solid rgba(255,255,255,0.35);
        }}
        .kpi-label {{
            font-size: 13px;
            font-weight: 600;
            color: #c8cfe8;
            line-height: 1.3;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .kpi-sub {{
            font-size: 11px;
            color: rgba(200,207,232,0.6);
        }}
        .action-card {{
            border-radius: 12px;
            padding: 16px 22px;
            margin-bottom: 12px;
            backdrop-filter: blur(8px);
        }}
        .tip-card {{
            border-radius: 6px;
            padding: 9px 14px;
            margin-bottom: 8px;
        }}
        .stats-bar {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            color: #c8cfe8;
            font-size: 13px;
        }}
        @media (max-width: 768px) {{
            .stats-item {{
                width: 100%;
            }}
        }}
    </style>
""", unsafe_allow_html=True)

# ── Konstansok (app.py-val azonos paletta) ─────────────────────────────────────
# Pontos szegmens nevek a parquet-ból
SEG_COLORS = {
    "VIP Bajnokok":          "#ff1a3c",
    "Lemorzsolódó / Alvó":   "#ff8c1a",
    "Új / Ígéretes":         "#1ab4ff",
    "Elvesztett / Inaktív":  "#9898c0",
}

SEG_FILL = {
    "VIP Bajnokok":          "rgba(255,26,60,0.22)",
    "Lemorzsolódó / Alvó":   "rgba(255,140,26,0.22)",
    "Új / Ígéretes":         "rgba(26,180,255,0.22)",
    "Elvesztett / Inaktív":  "rgba(152,152,192,0.22)",
}

# Pontos action nevek a parquet-ból
ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció":  "#ff1a3c",
    "💎 VIP Stabil – Lojalitás Program":       "#1ab4ff",
    "⚠️  Lemorzsolódó – Win-Back Kampány":     "#ff8c1a",
    "✅ Stabil – Standard Kommunikáció":        "#9898c0",
}

ACTION_ADVICE = {
    "🚨 VIP Veszélyben – Azonnali Retenció": [
        "Személyes account manager megkeresés (B2B ügyfeleknél kötelező)",
        "Exkluzív visszatérési kupon (15–20% kedvezmény, 30 napos érvényességgel)",
        "Win-back e-mail sorozat (3 üzenet, 2 hetes intervallummal)",
        "NPS/CSAT felmérés küldése – proaktív panaszkezelés",
        "Prioritásos ügyfélszolgálati csoport bevonása",
    ],
    "💎 VIP Stabil – Lojalitás Program": [
        "VIP hűségkártya vagy pontalapú jutalomrendszer",
        "Korai hozzáférés új termékekhez és exkluzív akciókhoz",
        "Személyes születésnapi/évfordulós üzenet és ajándék",
        "Upselling: prémium termékcsaládok és bundlök kiemelése",
        "Referral program: jutalomért cserébe új ügyfelek ajánlása",
    ],
    "⚠️  Lemorzsolódó – Win-Back Kampány": [
        "Automatikus reaktivációs e-mail trigger (6 hónap inaktivitás után)",
        "Win-back kupon (5–10% kedvezmény)",
        "Szezonális hírlevél és akciós termékajánlatok",
        "Kategóriaalapú ajánlás (az ügyfél korábbi vásárlásai alapján)",
    ],
    "✅ Stabil – Standard Kommunikáció": [
        "Általános hírlevél és promóciós kampányok",
        "Kosárelhagyás emlékeztető e-mail aktiválása",
        "Collaborative filtering alapú termékajánló bekapcsolása",
        "Onboarding e-mail sorozat indítása (ha új ügyfél)",
    ],
}

# Közlekedési lámpa logika — zöld→sárga→narancs→piros, dark mode-ra optimalizálva
RISK_LEVELS = [
    (0.0,  0.30, "🟢 Alacsony kockázat",  "#00e676"),   # Material Green A400
    (0.30, 0.55, "🟡 Közepes kockázat",   "#ffd740"),   # Material Amber A400
    (0.55, 0.75, "🟠 Magas kockázat",     "#ff6d00"),   # Material Deep Orange A700
    (0.75, 1.01, "🔴 Kritikus kockázat",  "#ff1744"),   # Material Red A400
]

def churn_risk_label(prob: float):
    for lo, hi, label, color in RISK_LEVELS:
        if lo <= prob < hi:
            return label, color
    return "🔴 Kritikus kockázat", "#ff1744"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Adatbetöltés ───────────────────────────────────────────────────────────────
combined  = load_churn_predictions()
df_tx_all = load_transactions()

if combined.empty:
    st.error("Hiba: Nem található a `data/processed/churn_predictions.parquet` fájl!")
    st.stop()

# Globális statisztikák (adatból számolva, nem hardkódolva)
churn_proba_mean = combined["churn_proba"].mean()

# ── Fő fejléc ─────────────────────────────────────────────────────────────────
st.title("Ügyfélkereső értékesítőknek")
st.markdown("---")

# ── Keresősáv ─────────────────────────────────────────────────────────────────
all_ids = sorted(combined.index.tolist())

search_col, random_col = st.columns([4, 1])
with random_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Véletlenszerű ügyfél"):
        st.session_state["selected_id"] = str(np.random.choice(all_ids))

with search_col:
    options = [""] + all_ids
    default_index = (
        options.index(st.session_state["selected_id"])
        if "selected_id" in st.session_state and st.session_state["selected_id"] in options
        else 0
    )
    selected_id = st.selectbox(
        "Ügyfél ID keresése",
        options=options,
        index=default_index,
        placeholder="Írj be egy Customer ID-t…",
    )
    # Kézi kiválasztás felülírja a session_state-et
    if selected_id:
        st.session_state["selected_id"] = selected_id

# ── Üres állapot ──────────────────────────────────────────────────────────────
if not selected_id:
    st.info("Keress rá egy ügyfél ID-ra, vagy nyomd meg a 🎲 Véletlenszerű ügyfél gombot!")
    st.stop()

# ── Kiválasztott ügyfél adatai ────────────────────────────────────────────────
customer     = combined.loc[selected_id]
churn_prob   = float(customer["churn_proba"])
churn_pred   = int(customer["churn_pred"])
recency      = int(customer["recency_days"])
frequency    = int(customer["frequency"])
monetary     = float(customer["monetary_total"])
monetary_avg = float(customer["monetary_avg"])
return_ratio = float(customer["return_ratio"])
rfm_segment  = str(customer["rfm_segment"])
action       = str(customer["action"])  # közvetlenül a parquet-ból

risk_label, risk_color = churn_risk_label(churn_prob)
action_color = ACTION_COLORS.get(action, "#9898c0")
seg_color    = SEG_COLORS.get(rfm_segment, "#9898c0")

# ── Fejléc kártya ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: {hex_to_rgba(risk_color, 0.30)};
                border: 2px solid {risk_color};
                border-radius: 12px;
                padding: 18px 26px;
                margin-bottom: 18px;
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
                box-shadow: 0 0 22px {hex_to_rgba(risk_color, 0.45)}, 0 4px 28px rgba(0,0,0,0.5);">
        <h2 style="margin:0; color:#ffffff; font-size:1.4em;">
            Ügyfél:&nbsp;
            <code style="background:rgba(255,255,255,0.12); padding:2px 10px;
                border-radius:4px; font-size:1em; color:#ffffff;">{selected_id}</code>
        </h2>
        <p style="margin: 8px 0 0; font-size: 1em; color: #c8cfe8;">
            <b>Ajánlott akció:</b> <span style="color:{action_color}; font-weight:700;">{action}</span>
            &nbsp;|&nbsp;
            <b>Churn kockázat:</b> <span style="color:{risk_color}; font-weight:700;">{risk_label}</span>
            &nbsp;|&nbsp;
            <b>RFM Szegmens:</b> <span style="color:{seg_color}; font-weight:700;">{rfm_segment}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI kártyák ───────────────────────────────────────────────────────────────
seg_avg_churn = combined[combined["rfm_segment"] == rfm_segment]["churn_proba"].mean()
delta_vs_seg  = churn_prob - seg_avg_churn
delta_sign    = "+" if delta_vs_seg >= 0 else ""

st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">🕐 Recency</div>
            <div class="kpi-value">{recency} nap</div>
            <div class="kpi-sub">Utolsó vásárlás óta eltelt</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🛒 Frequency</div>
            <div class="kpi-value">{frequency} db</div>
            <div class="kpi-sub">Összes vásárlás</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">💰 Monetary</div>
            <div class="kpi-value">£{monetary:,.0f}</div>
            <div class="kpi-sub">Átl. kosár: £{monetary_avg:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🔄 Visszaküldési arány</div>
            <div class="kpi-value">{return_ratio:.1%}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Vizualizáció ──────────────────────────────────────────────────────────────
CHART_BG  = "rgba(0,0,0,0)"
CHART_PLT = "rgba(168,16,34,0.08)"
GRID_CLR  = "rgba(255,255,255,0.07)"
TICK_CLR  = "rgba(200,207,232,0.6)"

st.subheader("Churn kockázat mérő")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_prob * 100,
    title={"text": f"Churn valószínűség<br><span style='font-size:0.75em;color:#c8cfe8'>Bázis átlag: {churn_proba_mean:.1%}</span>", "font": {"size": 14, "color": "white"}},
    number={"suffix": "%", "valueformat": ".1f", "font": {"color": risk_color, "size": 42}},
    gauge={
        "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": TICK_CLR, "tickfont": {"color": TICK_CLR}},
        "bar":  {"color": risk_color, "thickness": 0.25},
        "bgcolor": "rgba(0,0,0,0)",
        "borderwidth": 0,
        "steps": [
            {"range": [0,  30],  "color": "rgba(0,230,118,0.45)"},
            {"range": [30, 55],  "color": "rgba(255,215,64,0.45)"},
            {"range": [55, 75],  "color": "rgba(255,109,0,0.50)"},
            {"range": [75, 100], "color": "rgba(255,23,68,0.55)"},
        ],
        "threshold": {
            "line":      {"color": "rgba(255,255,255,0.6)", "width": 2},
            "thickness": 0.8,
            "value":     churn_proba_mean * 100,
        },
    },
))
fig_gauge.update_layout(
    height=380,
    paper_bgcolor=CHART_BG,
    font=dict(color="white"),
    margin=dict(l=80, r=80, t=60, b=20),
)
st.plotly_chart(fig_gauge, use_container_width=True)

# Szegmens átlag összehasonlítás
_, cmp_col, _ = st.columns([1, 2, 1])
with cmp_col:
    delta_sign   = "+" if delta_vs_seg >= 0 else ""
    delta_szoveg = "magasabb a szegmens átlagánál" if delta_vs_seg >= 0 else "alacsonyabb a szegmens átlagánál"
    delta_color  = "#ff6d00" if delta_vs_seg >= 0 else "#00e676"
    pred_text    = "🔴 Lemorzsolódónak előrejelzett" if churn_pred == 1 else "🟢 Megtartottnak előrejelzett"
    pred_color   = "#ff1744" if churn_pred == 1 else "#00e676"
    st.markdown(
        f"""<div style="background:rgba(168,16,34,0.3); border:1px solid rgba(255,255,255,0.15);
                border-radius:8px; padding:12px 16px; color:white;">
            <div style="font-size:13px; color:#c8cfe8; margin-bottom:6px;">
                Szegmens átlag — {rfm_segment}
            </div>
            <div style="font-size:22px; font-weight:700;">
                {seg_avg_churn:.1%}
            </div>
            <div style="margin-top:6px; font-size:13px; color:{delta_color}; font-weight:600;">
                {delta_sign}{delta_vs_seg:.1%} — ez az ügyfél <b>{delta_szoveg}</b>
            </div>
            <div style="margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;
                        text-align:center; font-weight:600; color:{pred_color};">
                {pred_text}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Vásárlási előzmények ───────────────────────────────────────────────────────
if not df_tx_all.empty:
    st.markdown("---")
    cust_tx = (
        df_tx_all[df_tx_all["Customer ID"] == selected_id]
        .sort_values("InvoiceDate", ascending=False)
    )

    if cust_tx.empty:
        st.info("Ehhez az ügyfélhez nem találhatók tranzakciós adatok.")
    else:
        # ── Havi bevétel trend ─────────────────────────────────────────────────
        st.subheader("Havi bevétel trend")

        # Nettó havi bevétel (vásárlások + visszaküldések együtt)
        monthly = (
            cust_tx.groupby(cust_tx["InvoiceDate"].dt.to_period("M"))["LineTotal"]
            .sum()
            .reset_index()
        )
        monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
        monthly = monthly.sort_values("InvoiceDate")

        # Visszaküldéses hónapok (Quantity < 0 sorok)
        returns_tx = cust_tx[cust_tx["Quantity"] < 0].copy()
        monthly_returns = (
            returns_tx.groupby(returns_tx["InvoiceDate"].dt.to_period("M"))["LineTotal"]
            .sum()
            .abs()
            .reset_index()
        )
        monthly_returns["InvoiceDate"] = monthly_returns["InvoiceDate"].astype(str)

        fig_trend = go.Figure()

        # Bevétel terület
        REVENUE_COLOR = "#00e676"

        fig_trend.add_trace(go.Scatter(
            x=monthly["InvoiceDate"],
            y=monthly["LineTotal"],
            mode="lines+markers",
            name="Nettó bevétel",
            line=dict(color=REVENUE_COLOR, width=2.5),
            marker=dict(size=7, color=REVENUE_COLOR,
                        line=dict(width=1.5, color="white")),
            fill="tozeroy",
            fillcolor=hex_to_rgba(REVENUE_COLOR, 0.15),
            hovertemplate="<b>%{x}</b><br>Nettó bevétel: £%{y:,.0f}<extra></extra>",
        ))

        # Visszaküldések – lefelé mutató háromszög a tengelyen
        if not monthly_returns.empty:
            # Marker mérete: visszaküldött összeg alapján skálázva (8–22 px között)
            max_ret = monthly_returns["LineTotal"].max()
            marker_sizes = (
                monthly_returns["LineTotal"] / max_ret * 14 + 8
            ).tolist()

            fig_trend.add_trace(go.Scatter(
                x=monthly_returns["InvoiceDate"],
                y=[0] * len(monthly_returns),
                mode="markers+text",
                name="Visszaküldés",
                marker=dict(
                    symbol="triangle-down",
                    size=marker_sizes,
                    color="#ff6d00",
                    line=dict(width=1.5, color="white"),
                ),
                text=monthly_returns["LineTotal"].map("-£{:,.0f}".format),
                textposition="bottom center",
                textfont=dict(size=10, color="#ff6d00"),
                customdata=monthly_returns["LineTotal"].values,
                hovertemplate="<b>%{x}</b><br>Visszaküldés: -£%{customdata:,.0f}<extra></extra>",
            ))

        # Utolsó aktív pont kiemelése
        last_row = monthly.iloc[-1]
        fig_trend.add_trace(go.Scatter(
            x=[last_row["InvoiceDate"]],
            y=[last_row["LineTotal"]],
            mode="markers",
            name="Utolsó aktív hónap",
            marker=dict(size=13, color=risk_color, symbol="star",
                        line=dict(width=1.5, color="white")),
            hovertemplate=f"<b>Utolsó aktív hónap</b><br>£{last_row['LineTotal']:,.0f}<extra></extra>",
        ))

        fig_trend.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_PLT,
            font=dict(color="white"),
            xaxis=dict(
                color=TICK_CLR, gridcolor=GRID_CLR,
                tickangle=-35, title="Hónap",
            ),
            yaxis=dict(
                color=TICK_CLR, gridcolor=GRID_CLR,
                tickprefix="£", tickformat=",.0f", title="Bevétel (£)",
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
            ),
            margin=dict(l=10, r=10, t=40, b=60),
            height=360,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Összes tranzakció táblázat ─────────────────────────────────────────
        st.subheader("Összes tranzakció")

        only_returns = st.checkbox("Csak visszaküldött tételek", value=False)
        st.caption("Visszaküldött tételek narancssárgával kiemelve.")

        tx_filtered = cust_tx[cust_tx["Quantity"] < 0] if only_returns else cust_tx
        top_n = tx_filtered[
            ["InvoiceDate", "Invoice", "Description", "Quantity", "Price", "LineTotal"]
        ].copy()
        top_n["InvoiceDate"] = top_n["InvoiceDate"].dt.strftime("%Y-%m-%d")
        top_n["Price"]       = top_n["Price"].map("£{:,.2f}".format)
        top_n["LineTotal"]   = top_n["LineTotal"].map("£{:,.2f}".format)
        top_n = top_n.rename(columns={
            "InvoiceDate": "Dátum",
            "Invoice":     "Számla",
            "Description": "Termék",
            "Quantity":    "Mennyiség",
            "Price":       "Egységár",
            "LineTotal":   "Összeg",
        })

        def _highlight_returns(row):
            if row["Mennyiség"] < 0:
                return ["background-color: rgba(255,109,0,0.22); color: #ff6d00;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            top_n.style.apply(_highlight_returns, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Összesítő sor
        total_orders   = cust_tx["Invoice"].nunique()
        total_revenue  = cust_tx["LineTotal"].sum()
        total_items    = cust_tx["Quantity"].sum()
        first_purchase = cust_tx["InvoiceDate"].min().strftime("%Y-%m-%d")
        last_purchase  = cust_tx["InvoiceDate"].max().strftime("%Y-%m-%d")

        st.markdown(f"""
            <div style="background:rgba(168,16,34,0.3); border:1px solid rgba(255,255,255,0.12);
                    border-radius:8px; padding:12px 18px; margin-top:10px;">
                <div class="stats-bar">
                    <span class="stats-item">📦 <b style="color:white">{total_orders:,}</b> rendelés</span>
                    <span class="stats-item">🛍️ <b style="color:white">{total_items:,}</b> tétel összesen</span>
                    <span class="stats-item">💰 <b style="color:white">£{total_revenue:,.0f}</b> teljes bevétel (élettartam)</span>
                    <span class="stats-item">📅 Első vásárlás: <b style="color:white">{first_purchase}</b></span>
                    <span class="stats-item">📅 Utolsó vásárlás: <b style="color:white">{last_purchase}</b></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
