"""
pages/3_🔍_Ügyfél_Kereső.py
Egyedi ügyfél RFM-profil és churn-kockázat kereső
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ügyfél Kereső",
    page_icon="🔍",
    layout="wide",
)

# ── Konstansok ─────────────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#2ecc71",
    "Lemorzsolódó / Alvó":    "#e67e22",
    "Elvesztett / Inaktív":   "#e74c3c",
    "Új / Ígéretes":          "#3498db",
    "Ismeretlen":             "#95a5a6",
    "N/A":                    "#95a5a6",
}

ACTION_COLORS = {
    "🚨 VIP Veszélyben – Azonnali Retenció":             "#e74c3c",
    "💎 VIP Stabil – Lojalitás Program":                  "#2ecc71",
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back":      "#e67e22",
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": "#3498db",
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
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back": [
        "Automatikus reaktivációs e-mail trigger (6 hónap inaktivitás után)",
        "Alacsony értékű win-back kupon (5–10% kedvezmény)",
        "Szezonális hírlevél és akciós termékajánlatok",
        "Kategóriaalapú ajánlás (az ügyfél korábbi vásárlásai alapján)",
    ],
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": [
        "Általános hírlevél és promóciós kampányok",
        "Kosárelhagyás emlékeztető e-mail aktiválása",
        "Collaborative filtering alapú termékajánló bekapcsolása",
        "Onboarding e-mail sorozat indítása (ha új ügyfél)",
    ],
}

RISK_LEVELS = [
    (0.0, 0.3,  "🟢 Alacsony kockázat",  "#2ecc71"),
    (0.3, 0.55, "🟡 Közepes kockázat",   "#f39c12"),
    (0.55, 0.75,"🟠 Magas kockázat",     "#e67e22"),
    (0.75, 1.0, "🔴 Kritikus kockázat",  "#e74c3c"),
]

def churn_risk_label(prob: float):
    for lo, hi, label, color in RISK_LEVELS:
        if lo <= prob < hi or (prob >= hi and hi == 1.0):
            return label, color
    return "❓ Ismeretlen", "#95a5a6"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Konvertál egy #rrggbb hex színt rgba() formátumra Plotly számára."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Adat betöltők ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Adatok betöltése...")
def load_churn() -> pd.DataFrame:
    candidates = [
        Path("data/processed/churn_predictions.parquet"),
        Path("../data/processed/churn_predictions.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df.index = df.index.astype(str)
            return df
    return None


@st.cache_data(show_spinner="Szegmentációs adatok betöltése...")
def load_segments() -> pd.DataFrame:
    candidates = [
        Path("data/processed/customer_segments.parquet"),
        Path("../data/processed/customer_segments.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df.index = df.index.astype(str)
            for col in ["Segment", "segment_label", "Szegmens"]:
                if col in df.columns:
                    df = df.rename(columns={col: "Segment"})
                    return df
            str_cols = df.select_dtypes(include="object").columns
            if len(str_cols):
                df = df.rename(columns={str_cols[0]: "Segment"})
            return df
    return None


def build_demo_data(n=5243) -> pd.DataFrame:
    """Szintetikus demo adatok generálása, ha nincs parquet fájl."""
    rng = np.random.default_rng(42)
    segments = rng.choice(
        ["VIP Bajnokok", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív", "Új / Ígéretes"],
        size=n, p=[0.10, 0.25, 0.45, 0.20],
    )
    recency = np.where(segments == "VIP Bajnokok", rng.integers(5, 60, n),
               np.where(segments == "Új / Ígéretes", rng.integers(5, 70, n),
               np.where(segments == "Lemorzsolódó / Alvó", rng.integers(90, 300, n),
                        rng.integers(270, 650, n))))
    frequency = np.where(segments == "VIP Bajnokok", rng.integers(10, 50, n),
                 np.where(segments == "Lemorzsolódó / Alvó", rng.integers(3, 12, n),
                 np.where(segments == "Új / Ígéretes", rng.integers(1, 5, n),
                          rng.integers(1, 3, n))))
    monetary = np.where(segments == "VIP Bajnokok", rng.uniform(5000, 30000, n),
                np.where(segments == "Lemorzsolódó / Alvó", rng.uniform(500, 5000, n),
                np.where(segments == "Új / Ígéretes", rng.uniform(200, 1500, n),
                         rng.uniform(50, 800, n))))
    return_ratio = rng.uniform(0, 0.25, n)
    monetary_avg = monetary / frequency

    churn_log = (0.012 * recency - 0.05 * np.log1p(frequency)
                 - 0.03 * np.log1p(monetary) + 0.8 * return_ratio
                 + rng.normal(0, 0.3, n))
    churn_proba = 1 / (1 + np.exp(-churn_log))
    monetary_median = np.median(monetary)

    def _action(hv, hc):
        if hv and hc:   return "🚨 VIP Veszélyben – Azonnali Retenció"
        elif hv:        return "💎 VIP Stabil – Lojalitás Program"
        elif hc:        return "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
        else:           return "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

    actions = [_action(m > monetary_median, p >= 0.5) for m, p in zip(monetary, churn_proba)]

    df = pd.DataFrame({
        "recency_days": recency,
        "frequency": frequency,
        "monetary_total": monetary,
        "monetary_avg": monetary_avg,
        "return_ratio": return_ratio,
        "churn_proba": churn_proba,
        "rfm_segment": segments,
        "action": actions,
    }, index=[str(12346 + i) for i in range(n)])
    df.index.name = "Customer ID"
    return df


# ── Adatok összeállítása ───────────────────────────────────────────────────────
churn_df = load_churn()
seg_df = load_segments()

if churn_df is None and seg_df is None:
    st.warning("⚠️ **Demo mód** – Szintetikus adatokkal dolgozunk.", icon="🔔")
    combined = build_demo_data()
elif churn_df is not None:
    combined = churn_df.copy()
    if seg_df is not None and "Segment" in seg_df.columns:
        combined = combined.join(seg_df[["Segment"]], how="left")
        combined["Segment"] = combined.get("Segment", combined.get("rfm_segment", "Ismeretlen"))
    if "rfm_segment" in combined.columns and "Segment" not in combined.columns:
        combined["Segment"] = combined["rfm_segment"]
    combined["Segment"] = combined.get("Segment", "Ismeretlen").fillna("Ismeretlen")
else:
    combined = build_demo_data()

# Monetáris medián az action újraszámításhoz
monetary_median_global = combined["monetary_total"].median()

# ── Fő layout ─────────────────────────────────────────────────────────────────
st.title("🔍 Ügyfél Kereső")
st.caption("Egyedi ügyfél RFM-profil, churn-kockázat és személyre szabott akciójavaslat")
st.markdown("---")

# ── Keresősáv ─────────────────────────────────────────────────────────────────
all_ids = sorted(combined.index.tolist())

search_col, random_col = st.columns([4, 1])
with search_col:
    selected_id = st.selectbox(
        "🔎 Ügyfél ID keresése",
        options=[""] + all_ids,
        index=0,
        placeholder="Írj be egy Customer ID-t…",
    )
with random_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Véletlenszerű ügyfél"):
        selected_id = np.random.choice(all_ids)
        st.session_state["random_id"] = selected_id

# Ha véletlenszerű gomb lett nyomva
if "random_id" in st.session_state and not selected_id:
    selected_id = st.session_state["random_id"]

if not selected_id:
    st.info("👆 Keress rá egy ügyfél ID-ra, vagy nyomd meg a 🎲 Véletlenszerű ügyfél gombot!")

    # ── Kereshető ügyfelek statisztikája ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Adatbázis áttekintése")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Összes ügyfél", f"{len(combined):,}")
    k2.metric("Lemorzsolódók (≥50%)", f"{(combined['churn_proba'] >= 0.5).sum():,}")
    k3.metric("VIP Veszélyben", f"{((combined['monetary_total'] > monetary_median_global) & (combined['churn_proba'] >= 0.5)).sum():,}")
    k4.metric("Átlag Churn Valószínűség", f"{combined['churn_proba'].mean():.1%}")

    seg_col = "Segment" if "Segment" in combined.columns else "rfm_segment"
    if seg_col in combined.columns:
        fig_overview = px.histogram(
            combined,
            x="churn_proba",
            color=seg_col,
            color_discrete_map=SEGMENT_COLORS,
            nbins=30,
            title="Churn valószínűség eloszlása szegmensenként",
            labels={"churn_proba": "Churn valószínűség", seg_col: "Szegmens"},
        )
        st.plotly_chart(fig_overview, use_container_width=True)
    st.stop()

# ── Kiválasztott ügyfél adatai ────────────────────────────────────────────────
customer = combined.loc[selected_id]
churn_prob = float(customer.get("churn_proba", 0.5))
recency   = int(customer.get("recency_days", 0))
frequency = int(customer.get("frequency", 1))
monetary  = float(customer.get("monetary_total", 0))
monetary_avg = float(customer.get("monetary_avg", monetary))
return_ratio = float(customer.get("return_ratio", 0))

seg_col_name = "Segment" if "Segment" in customer.index else "rfm_segment"
rfm_segment  = str(customer.get(seg_col_name, customer.get("rfm_segment", "Ismeretlen")))

# Akció dinamikus meghatározása
hv = monetary > monetary_median_global
hc = churn_prob >= 0.5
if hv and hc:     action = "🚨 VIP Veszélyben – Azonnali Retenció"
elif hv:          action = "💎 VIP Stabil – Lojalitás Program"
elif hc:          action = "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
else:             action = "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

risk_label, risk_color = churn_risk_label(churn_prob)
action_color = ACTION_COLORS.get(action, "#95a5a6")
seg_color = SEGMENT_COLORS.get(rfm_segment, "#95a5a6")

# ── Fejléc kártya ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {hex_to_rgba(action_color, 0.13)}, {hex_to_rgba(action_color, 0.03)});
                border: 2px solid {action_color}; border-radius: 12px;
                padding: 20px 28px; margin-bottom: 20px;">
        <h2 style="margin:0; color:{action_color};">
            Ügyfél: <code style="background:{action_color}22; padding:2px 8px;
            border-radius:4px; font-size:1.1em;">{selected_id}</code>
        </h2>
        <p style="margin: 8px 0 0; font-size: 1.05em;">
            <b>Ajánlott akció:</b> {action}
            &nbsp;|&nbsp;
            <b>Churn kockázat:</b> <span style="color:{risk_color}; font-weight:bold;">{risk_label}</span>
            &nbsp;|&nbsp;
            <b>RFM Szegmens:</b> <span style="color:{seg_color}; font-weight:bold;">{rfm_segment}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI kártyák ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🕐 Recency", f"{recency} nap")
c2.metric("🛒 Frequency", f"{frequency} vásárlás")
c3.metric("💰 Monetary", f"£{monetary:,.0f}")
c4.metric("🔄 Return arány", f"{return_ratio:.1%}")
c5.metric("⚠️ Churn valószínűség", f"{churn_prob:.1%}", delta=f"{churn_prob - 0.5:.1%} (medián felett)" if churn_prob > 0.5 else f"{churn_prob - 0.5:.1%}")

st.markdown("---")

# ── Vizualizációk ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📡 Radar – RFM profil")
    # Percentilis az összes ügyfelek körében
    metrics = {
        "Recency\n(alacsonyabb = jobb)": (1 - np.searchsorted(np.sort(combined["recency_days"]), recency) / len(combined)),
        "Frequency": np.searchsorted(np.sort(combined["frequency"]), frequency) / len(combined),
        "Monetary": np.searchsorted(np.sort(combined["monetary_total"]), monetary) / len(combined),
        "Átl. kosárérték": np.searchsorted(np.sort(combined["monetary_avg"]), monetary_avg) / len(combined),
        "Return arány\n(alacsonyabb = jobb)": 1 - np.searchsorted(np.sort(combined["return_ratio"]), return_ratio) / len(combined),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=hex_to_rgba(action_color, 0.2),
        line=dict(color=action_color, width=2.5),
        name=selected_id,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
        title=f"Ügyfél percentilis rangja (0% = legrosszabb, 100% = legjobb)",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_right:
    st.subheader("📊 Churn kockázat mérő")

    # Churn gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob * 100,
        delta={"reference": 55.7, "valueformat": ".1f", "suffix": "%"},
        title={"text": f"Churn valószínűség (átlag: 55.7%)", "font": {"size": 14}},
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 30],  "color": "#d5f5e3"},
                {"range": [30, 55], "color": "#fef9e7"},
                {"range": [55, 75], "color": "#fdebd0"},
                {"range": [75, 100],"color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": 55.7,
            },
        },
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Összehasonlítás a szegmens átlagával
    seg_col_for_group = "Segment" if "Segment" in combined.columns else "rfm_segment"
    if seg_col_for_group in combined.columns:
        seg_avg = combined[combined[seg_col_for_group] == rfm_segment]["churn_proba"].mean()
        st.metric(
            f"Szegmens átlag churn ({rfm_segment})",
            f"{seg_avg:.1%}",
            delta=f"{(churn_prob - seg_avg):+.1%} vs. szegmens átlag",
            delta_color="inverse",
        )

st.markdown("---")

# ── Elhelyezés a populációban ─────────────────────────────────────────────────
st.subheader("🗺️ Elhelyezés az ügyfélpopulációban")
sample_bg = combined.sample(min(len(combined), 1500), random_state=42)
fig_pos = px.scatter(
    sample_bg,
    x="recency_days",
    y="monetary_total",
    color="churn_proba",
    color_continuous_scale="RdYlGn_r",
    opacity=0.45,
    log_y=True,
    labels={
        "recency_days": "Recency (napok)",
        "monetary_total": "Monetary – log skála (£)",
        "churn_proba": "Churn valószínűség",
    },
    title="Ügyfél elhelyezése a populációban (churn valószínűség szerint színezve)",
)
# Kiemelés
fig_pos.add_trace(go.Scatter(
    x=[recency],
    y=[monetary],
    mode="markers",
    marker=dict(size=18, color=risk_color, symbol="star",
                line=dict(width=2, color="white")),
    name=f"Ügyfél {selected_id}",
    hovertext=f"ID: {selected_id}<br>Recency: {recency} nap<br>£{monetary:,.0f}<br>Churn: {churn_prob:.1%}",
))
fig_pos.update_layout(height=460)
st.plotly_chart(fig_pos, use_container_width=True)

st.markdown("---")

# ── Akció javaslatok ──────────────────────────────────────────────────────────
st.subheader(f"💡 Személyre szabott akciójavaslatok – {action}")
advice_list = ACTION_ADVICE.get(action, [])
advice_col1, advice_col2 = st.columns([2, 1])
with advice_col1:
    for i, tip in enumerate(advice_list, 1):
        st.markdown(
            f"""<div style="border-left: 4px solid {action_color};
                        padding: 8px 14px; margin-bottom: 10px;
                        border-radius: 4px; background: {hex_to_rgba(action_color, 0.07)};">
                <b>{i}.</b> {tip}
            </div>""",
            unsafe_allow_html=True,
        )
with advice_col2:
    st.markdown("**Ügyfél adatlap összefoglaló:**")
    summary = {
        "Customer ID":            selected_id,
        "RFM Szegmens":           rfm_segment,
        "Utolsó vásárlás":        f"{recency} napja",
        "Vásárlások száma":        f"{frequency} db",
        "Nettó bevétel":          f"£{monetary:,.2f}",
        "Átl. kosárérték":        f"£{monetary_avg:,.2f}",
        "Visszaküldési arány":    f"{return_ratio:.1%}",
        "Churn valószínűség":     f"{churn_prob:.1%}",
        "Kockázati szint":        risk_label,
        "Ajánlott akció":         action.split("–")[0].strip(),
    }
    for k, v in summary.items():
        st.markdown(f"**{k}:** {v}")

    # CSV export az ügyfél adataival
    csv_single = pd.DataFrame([summary]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Adatlap letöltése (CSV)",
        data=csv_single,
        file_name=f"ugyfel_{selected_id}_adatlap.csv",
        mime="text/csv",
    )
