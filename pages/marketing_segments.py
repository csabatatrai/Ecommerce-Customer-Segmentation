"""
pages/Marketing_szegmensek.py
Szegmensszintű marketing elemzés és kampánytervezési eszköz
"""

import base64

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from src.sidebar import render_sidebar
from src.data_loader import load_churn_predictions, load_transactions

render_sidebar()

# ── Háttérkép ─────────────────────────────────────────────────────────────────
bg_path = Path("src/streamlit_bg.webp")
if bg_path.exists():
    bg_b64 = base64.b64encode(bg_path.read_bytes()).decode()
    bg_css = f'background-image: url("data:image/webp;base64,{bg_b64}");'
else:
    bg_css = ""

# ── Stílus ────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <style>
        .stApp {{
            {bg_css}
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .seg-card {{
            background: rgba(168, 16, 34, 0.45);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 12px;
            padding: 1.1rem 1.4rem 1rem 1.4rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.35);
            margin-bottom: 0.5rem;
        }}
        .seg-label {{
            font-size: 13px;
            font-weight: 600;
            color: #c8cfe8;
        }}
        .seg-value {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
            margin: 0.2rem 0;
        }}
        .seg-sub {{
            font-size: 12px;
            color: rgba(200,207,232,0.6);
        }}
        .callout {{
            border-radius: 0 6px 6px 0;
            padding: 8px 14px;
            font-size: 0.85em;
            margin-bottom: 0.4rem;
        }}
    </style>
""", unsafe_allow_html=True)

# ── Konstansok ────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "VIP Bajnokok":         "#ff1a3c",
    "Lemorzsolódó / Alvó":  "#ff8c1a",
    "Új / Ígéretes":        "#1ab4ff",
    "Elvesztett / Inaktív": "#9898c0",
}

GREY_LOWER = 0.3
GREY_UPPER = 0.7

CHART_BG  = "rgba(0,0,0,0)"
CHART_PLT = "rgba(168,16,34,0.08)"
GRID_CLR  = "rgba(255,255,255,0.07)"
TICK_CLR  = "rgba(200,207,232,0.6)"

# ── Adatbetöltés ──────────────────────────────────────────────────────────────
df = load_churn_predictions()
df_tx = load_transactions()

if df.empty:
    st.error("Hiba: Nem található a `data/processed/churn_predictions.parquet` fájl!")
    st.stop()

# ── Fejléc ────────────────────────────────────────────────────────────────────
st.title("Marketing szegmenselemzés")
st.markdown("Szegmensszintű churn-kockázat, bevételi súly és kampánycélpont-azonosítás marketingstratégia tervezéséhez.")
st.markdown("---")

# ── Szegmens szintű aggregáció ────────────────────────────────────────────────
cutoff_date   = pd.to_datetime("2011-09-09")
start_ttm     = cutoff_date - pd.Timedelta(days=365)

seg_lookup    = df["rfm_segment"]
tx_seg        = df_tx.copy()
tx_seg["rfm_segment"] = tx_seg["Customer ID"].map(seg_lookup)
tx_seg        = tx_seg[tx_seg["rfm_segment"].notna()]
tx_ttm        = tx_seg[(tx_seg["InvoiceDate"] >= start_ttm) & (tx_seg["InvoiceDate"] < cutoff_date)]

ttm_by_seg    = tx_ttm.groupby("rfm_segment")["LineTotal"].sum()
total_ttm_rev = ttm_by_seg.sum()

grey_mask     = (df["churn_proba"] >= GREY_LOWER) & (df["churn_proba"] <= GREY_UPPER)

seg_stats = (
    df.groupby("rfm_segment")
    .agg(
        n_customers   =("churn_proba", "count"),
        avg_churn_prob=("churn_proba", "mean"),
        n_predicted_churn=("churn_pred", "sum"),
        n_grey_zone   =("churn_proba", lambda x: ((x >= GREY_LOWER) & (x <= GREY_UPPER)).sum()),
    )
    .reset_index()
)
seg_stats["ttm_revenue"]   = seg_stats["rfm_segment"].map(ttm_by_seg).fillna(0)
seg_stats["revenue_share"] = seg_stats["ttm_revenue"] / total_ttm_rev * 100
seg_stats["churn_rate"]    = seg_stats["n_predicted_churn"] / seg_stats["n_customers"] * 100
seg_stats["grey_pct"]      = seg_stats["n_grey_zone"] / seg_stats["n_customers"] * 100
seg_stats["color"]         = seg_stats["rfm_segment"].map(SEG_COLORS).fillna("#9898c0")
seg_stats = seg_stats.sort_values("ttm_revenue", ascending=False).reset_index(drop=True)

# ── 1. Szegmens KPI kártyák ───────────────────────────────────────────────────
st.subheader("Szegmensek áttekintése")

cols = st.columns(len(seg_stats))
for col, (_, row) in zip(cols, seg_stats.iterrows()):
    color = row["color"]
    with col:
        st.markdown(
            f"<div class='seg-card' style='border-top: 3px solid {color};'>"
            f"<div class='seg-label' style='color:{color};'>{row['rfm_segment']}</div>"
            f"<div class='seg-value'>{int(row['n_customers']):,} fő</div>"
            f"<div class='seg-sub'>TTM bevétel: £{row['ttm_revenue']/1000:,.0f}k "
            f"({row['revenue_share']:.1f}%)</div>"
            f"<div class='seg-sub'>Átl. churn valószínűség: {row['avg_churn_prob']:.1%}</div>"
            f"<div class='seg-sub'>Előrejelzett churner: {int(row['n_predicted_churn']):,} fő "
            f"({row['churn_rate']:.1f}%)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── 2. Bevételi súly vs. churn kockázat – buborékdiagram ─────────────────────
st.subheader("Bevételi súly vs. churn kockázat")
st.caption("Minél nagyobb a buborék, annál több ügyfél van a szegmensben. A vízszintes tengely a bevételi kitettséget, a függőleges a churn-kockázatot mutatja.")

fig_bubble = go.Figure()
for _, row in seg_stats.iterrows():
    fig_bubble.add_trace(go.Scatter(
        x=[row["revenue_share"]],
        y=[row["avg_churn_prob"] * 100],
        mode="markers+text",
        name=row["rfm_segment"],
        text=[row["rfm_segment"]],
        textposition="top center",
        textfont=dict(color="white", size=12),
        marker=dict(
            size=row["n_customers"] / seg_stats["n_customers"].max() * 60 + 20,
            color=row["color"],
            opacity=0.85,
            line=dict(width=2, color="rgba(255,255,255,0.4)"),
        ),
        hovertemplate=(
            f"<b>{row['rfm_segment']}</b><br>"
            f"Bevételi súly: {row['revenue_share']:.1f}%<br>"
            f"Átl. churn valószínűség: {row['avg_churn_prob']:.1%}<br>"
            f"Ügyfelek: {int(row['n_customers']):,} fő<extra></extra>"
        ),
    ))

fig_bubble.update_layout(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_PLT,
    font=dict(color="white"),
    xaxis=dict(
        title="TTM bevételi részarány (%)",
        color=TICK_CLR, gridcolor=GRID_CLR, ticksuffix="%",
    ),
    yaxis=dict(
        title="Átlagos churn valószínűség (%)",
        color=TICK_CLR, gridcolor=GRID_CLR, ticksuffix="%",
    ),
    showlegend=False,
    height=420,
    margin=dict(l=10, r=10, t=20, b=40),
)
st.plotly_chart(fig_bubble, use_container_width=True)

st.markdown("---")

# ── 3. Kampánycélpont-elemzés: Szürke Zóna szegmensenként ────────────────────
st.subheader("Kampánycélpont-elemzés — Szürke Zóna (30–70%)")
st.markdown(
    "<div class='callout' style='background:rgba(200,207,232,0.05); border-left:3px solid rgba(200,207,232,0.3);'>"
    "💡 A <b>30–70% közötti churn-valószínűségű</b> ügyfelek a leginkább befolyásolható kimenetelűek — "
    "ők alkotják az optimális kampánycélcsoportot. Az alábbi diagram szegmensenként mutatja a szürke zónás ügyfelek arányát és bevételi súlyát."
    "</div>",
    unsafe_allow_html=True,
)

grey_tx = tx_ttm[tx_ttm["Customer ID"].isin(df.index[grey_mask])]
grey_rev_by_seg = grey_tx.groupby(
    grey_tx["Customer ID"].map(seg_lookup)
)["LineTotal"].sum().reset_index()
grey_rev_by_seg.columns = ["rfm_segment", "grey_ttm_rev"]

seg_grey = seg_stats.merge(grey_rev_by_seg, on="rfm_segment", how="left").fillna(0)

fig_grey = go.Figure()
fig_grey.add_trace(go.Bar(
    name="Szürke zónás ügyfelek aránya (%)",
    x=seg_grey["rfm_segment"],
    y=seg_grey["grey_pct"],
    marker_color=seg_grey["color"].tolist(),
    opacity=0.85,
    text=[f"{v:.1f}%<br>({int(n):,} fő)" for v, n in zip(seg_grey["grey_pct"], seg_grey["n_grey_zone"])],
    textposition="outside",
    textfont=dict(color="white", size=12),
    hovertemplate="<b>%{x}</b><br>Szürke zóna arány: %{y:.1f}%<extra></extra>",
))

fig_grey.update_layout(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_PLT,
    font=dict(color="white"),
    xaxis=dict(color=TICK_CLR, gridcolor=GRID_CLR),
    yaxis=dict(title="Szürke zónás ügyfelek aránya (%)", color=TICK_CLR, gridcolor=GRID_CLR, ticksuffix="%"),
    showlegend=False,
    height=380,
    margin=dict(l=10, r=10, t=20, b=40),
)
st.plotly_chart(fig_grey, use_container_width=True)

st.markdown("---")

# ── 4. Szegmens-szintű kampányajánlások ──────────────────────────────────────
st.subheader("Kampányajánlások szegmensenként")

SEGMENT_STRATEGY = {
    "VIP Bajnokok": {
        "icon": "💎",
        "priority": "Magas értékvédelmi prioritás",
        "priority_color": "#ff1a3c",
        "goal": "Lojalitás megtartása, lemorzsolódás megelőzése",
        "tactics": [
            "Személyre szabott VIP megtartási program (exkluzív kedvezmények, korai termékbevezetés)",
            "Proaktív account manager kapcsolat B2B ügyfeleknek",
            "NPS/CSAT felmérés — panasz esetén azonnali eszkaláció",
            "Win-back sorozat indítása az első inaktivitási jelnél (30+ nap)",
        ],
    },
    "Lemorzsolódó / Alvó": {
        "icon": "⚠️",
        "priority": "Azonnali reaktivációs kampány",
        "priority_color": "#ff8c1a",
        "goal": "Újra-aktiválás, bevételkiesés csökkentése",
        "tactics": [
            "Automatikus win-back e-mail trigger (45–90 napos inaktivitás után)",
            "Időkorlátozott visszatérési kupon (10–15% kedvezmény, 30 napos érvényességgel)",
            "Korábbi vásárlások alapján személyre szabott termékajánlás",
            "Szezonális reaktivációs kampány (pl. ünnepi időszak előtt 6 héttel)",
        ],
    },
    "Új / Ígéretes": {
        "icon": "🌱",
        "priority": "Növekedési és konverziós fókusz",
        "priority_color": "#1ab4ff",
        "goal": "Második és harmadik vásárlás ösztönzése, szokáskialakítás",
        "tactics": [
            "Onboarding e-mail sorozat (3–5 üzenet, az első vásárlástól számított 60 napon belül)",
            "Második vásárlást ösztönző kupon (5–10% kedvezmény, 21 napos érvényességgel)",
            "Collaborative filtering alapú termékajánló bekapcsolása",
            "Loyalty program belépési meghívó a 2. vásárlás után",
        ],
    },
    "Elvesztett / Inaktív": {
        "icon": "💤",
        "priority": "Szelektív újraaktiválás",
        "priority_color": "#9898c0",
        "goal": "Alacsony költségű reaktivációs kísérlet, nem megtérülő ügyfelek elengedése",
        "tactics": [
            "Egyetlen reaktivációs e-mail küldése (nem sorozat) — ha nincs válasz, lista-eltávolítás",
            "Nagyon alacsony értékű, széles körű promóció (pl. szezonális kiárusítás értesítő)",
            "Szegmens mélyebb analízise: mennyi ideje inaktív és miért — adatvezérelt döntés a befektetésről",
        ],
    },
}

for _, row in seg_stats.iterrows():
    seg_name = row["rfm_segment"]
    strategy = SEGMENT_STRATEGY.get(seg_name)
    if not strategy:
        continue
    color = row["color"]

    with st.expander(f"{strategy['icon']} {seg_name} — {strategy['priority']}"):
        col_kpi, col_tactics = st.columns([1, 2])

        with col_kpi:
            st.markdown(
                f"<div style='background:rgba(168,16,34,0.3); border:1px solid rgba(255,255,255,0.12); "
                f"border-radius:8px; padding:12px 16px;'>"
                f"<div style='font-size:12px; color:#c8cfe8; margin-bottom:4px;'>Szegmens méret</div>"
                f"<div style='font-size:22px; font-weight:700; color:white;'>{int(row['n_customers']):,} fő</div>"
                f"<div style='font-size:12px; color:#c8cfe8; margin-top:10px; margin-bottom:4px;'>TTM bevétel</div>"
                f"<div style='font-size:22px; font-weight:700; color:white;'>£{row['ttm_revenue']/1000:,.0f}k</div>"
                f"<div style='font-size:12px; color:#c8cfe8; margin-top:10px; margin-bottom:4px;'>Előrejelzett churner</div>"
                f"<div style='font-size:22px; font-weight:700; color:{color};'>{int(row['n_predicted_churn']):,} fő</div>"
                f"<div style='font-size:12px; color:#c8cfe8; margin-top:10px; margin-bottom:4px;'>Szürke zóna (befolyásolható)</div>"
                f"<div style='font-size:22px; font-weight:700; color:#ffd740;'>{int(row['n_grey_zone']):,} fő</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_tactics:
            st.markdown(
                f"<div style='background:rgba(200,207,232,0.05); border-left:3px solid {color}; "
                f"border-radius:0 6px 6px 0; padding:10px 14px; margin-bottom:10px; font-size:0.9em;'>"
                f"<b style='color:{color};'>Cél:</b> "
                f"<span style='color:rgba(200,207,232,0.85);'>{strategy['goal']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Javasolt taktikák:**")
            for tactic in strategy["tactics"]:
                st.markdown(f"- {tactic}")

st.markdown("---")
st.caption(f"Adatforrás: churn_predictions.parquet · online_retail_ready_for_rfm.parquet · TTM: {start_ttm.strftime('%Y-%m-%d')} – {cutoff_date.strftime('%Y-%m-%d')}")
