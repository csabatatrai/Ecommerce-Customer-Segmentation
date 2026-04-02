"""
pages/1_📊_RFM_Szegmentáció.py
Ügyfélszegmentáció – K-Means (K=4) RFM alapján
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFM Szegmentáció",
    page_icon="📊",
    layout="wide",
)

# ── Konstansok és stílus ───────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "VIP Bajnokok":           "#2ecc71",
    "Lemorzsolódó / Alvó":    "#e67e22",
    "Elvesztett / Inaktív":   "#e74c3c",
    "Új / Ígéretes":          "#3498db",
}

SEGMENT_ICONS = {
    "VIP Bajnokok":           "👑",
    "Lemorzsolódó / Alvó":    "😴",
    "Elvesztett / Inaktív":   "💀",
    "Új / Ígéretes":          "🌱",
}

SEGMENT_DESCRIPTIONS = {
    "VIP Bajnokok": (
        "Legértékesebb ügyfelek. Nemrég vásároltak, sűrűn rendelnek, "
        "és sokat költenek. **Prioritás: Hűségprogram, exkluzív ajánlatok.**"
    ),
    "Lemorzsolódó / Alvó": (
        "Régebben aktívak, de már hónapok óta nem vásároltak. "
        "**Prioritás: Reaktivációs kampány, személyes megkeresés.**"
    ),
    "Elvesztett / Inaktív": (
        "Nagyon régen vásároltak utoljára, alacsony értékkel. "
        "**Prioritás: Win-back kupon vagy kivezetés.**"
    ),
    "Új / Ígéretes": (
        "Friss vásárlók, akikkel még fejleszthető a kapcsolat. "
        "**Prioritás: Onboarding, keresztértékesítés, hűségépítés.**"
    ),
}

# ── Adat betöltő ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Adatok betöltése...")
def load_segments() -> pd.DataFrame:
    """Betölti a customer_segments.parquet fájlt, fallback szintetikus adattal."""
    candidates = [
        Path("data/processed/customer_segments.parquet"),
        Path("../data/processed/customer_segments.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            # Szegmens névoszlop keresése
            for col in ["Segment", "segment_label", "Szegmens"]:
                if col in df.columns:
                    df = df.rename(columns={col: "Segment"})
                    break
            if "Segment" not in df.columns:
                str_cols = df.select_dtypes(include="object").columns
                if len(str_cols):
                    df = df.rename(columns={str_cols[0]: "Segment"})
            return df

    # ── Szintetikus demo adatok ───────────────────────────────────────────────
    st.warning(
        "⚠️ **Demo mód** – A `data/processed/customer_segments.parquet` nem található. "
        "Szintetikus adatokkal dolgozunk.",
        icon="🔔",
    )
    rng = np.random.default_rng(42)
    n = 5243
    segments = rng.choice(
        list(SEGMENT_COLORS.keys()),
        size=n,
        p=[0.10, 0.25, 0.45, 0.20],
    )
    df = pd.DataFrame(
        {
            "Customer ID": [str(12346 + i) for i in range(n)],
            "recency_days": np.where(
                segments == "VIP Bajnokok",
                rng.integers(5, 60, n),
                np.where(
                    segments == "Új / Ígéretes",
                    rng.integers(5, 60, n),
                    np.where(
                        segments == "Lemorzsolódó / Alvó",
                        rng.integers(90, 300, n),
                        rng.integers(270, 650, n),
                    ),
                ),
            ),
            "frequency": np.where(
                segments == "VIP Bajnokok",
                rng.integers(10, 50, n),
                np.where(
                    segments == "Lemorzsolódó / Alvó",
                    rng.integers(3, 12, n),
                    np.where(
                        segments == "Új / Ígéretes",
                        rng.integers(1, 5, n),
                        rng.integers(1, 3, n),
                    ),
                ),
            ),
            "monetary_total": np.where(
                segments == "VIP Bajnokok",
                rng.uniform(5000, 30000, n),
                np.where(
                    segments == "Lemorzsolódó / Alvó",
                    rng.uniform(500, 5000, n),
                    np.where(
                        segments == "Új / Ígéretes",
                        rng.uniform(200, 1500, n),
                        rng.uniform(50, 800, n),
                    ),
                ),
            ),
            "return_ratio": rng.uniform(0, 0.25, n),
            "monetary_avg": rng.uniform(100, 1000, n),
            "Segment": segments,
        }
    )
    df.set_index("Customer ID", inplace=True)
    return df


# ── Fő layout ─────────────────────────────────────────────────────────────────
st.title("📊 RFM Ügyfélszegmentáció")
st.caption("K-Means klaszterezés (K=4) | Megfigyelési ablak: 2009-12-01 – 2011-09-09")
st.markdown("---")

df = load_segments()

# ── Oldalsáv szűrők ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Szűrők")
    selected_segments = st.multiselect(
        "Szegmens",
        options=list(SEGMENT_COLORS.keys()),
        default=list(SEGMENT_COLORS.keys()),
    )
    recency_range = st.slider(
        "Recency (napok)",
        min_value=0,
        max_value=int(df["recency_days"].max()),
        value=(0, int(df["recency_days"].max())),
    )
    monetary_range = st.slider(
        "Nettó bevétel (£)",
        min_value=0,
        max_value=int(df["monetary_total"].max()),
        value=(0, int(df["monetary_total"].max())),
    )

mask = (
    df["Segment"].isin(selected_segments)
    & df["recency_days"].between(*recency_range)
    & df["monetary_total"].between(*monetary_range)
)
filtered = df[mask].copy()

# ── KPI kártyák ───────────────────────────────────────────────────────────────
st.subheader("📌 Összesített mutatók")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Összes ügyfél", f"{len(filtered):,}")
k2.metric("Átlag Recency", f"{filtered['recency_days'].mean():.0f} nap")
k3.metric("Átlag Frequency", f"{filtered['frequency'].mean():.1f} vásárlás")
k4.metric("Átlag Monetary", f"£{filtered['monetary_total'].mean():,.0f}")

st.markdown("---")

# ── Szegmens eloszlás + Átlagok táblázat ─────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Szegmens eloszlás")
    seg_counts = filtered["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Szegmens", "Ügyfelek száma"]
    seg_counts["Arány (%)"] = (seg_counts["Ügyfelek száma"] / len(filtered) * 100).round(1)
    fig_pie = px.pie(
        seg_counts,
        names="Szegmens",
        values="Ügyfelek száma",
        color="Szegmens",
        color_discrete_map=SEGMENT_COLORS,
        hole=0.42,
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
    fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Szegmens profilok (átlagok)")
    profile = (
        filtered.groupby("Segment")[
            ["recency_days", "frequency", "monetary_total", "return_ratio"]
        ]
        .mean()
        .round(2)
        .rename(
            columns={
                "recency_days": "Recency (nap)",
                "frequency": "Frequency",
                "monetary_total": "Monetary (£)",
                "return_ratio": "Return Ratio",
            }
        )
        .sort_values("Monetary (£)", ascending=False)
    )
    # Feltételes formázás
    st.dataframe(
        profile.style
        .background_gradient(subset=["Monetary (£)"], cmap="Greens")
        .background_gradient(subset=["Recency (nap)"], cmap="Reds_r")
        .format({"Return Ratio": "{:.1%}", "Monetary (£)": "£{:,.0f}"}),
        use_container_width=True,
        height=200,
    )
    st.caption(
        "💡 **Insight:** A VIP Bajnokok visszaküldési aránya (~16%) a legmagasabb – "
        "a leglojálisabb vásárlók aktívan élnek a visszaküldési politikával."
    )

st.markdown("---")

# ── Snake Plot ────────────────────────────────────────────────────────────────
st.subheader("🐍 Snake Plot – Szegmensek standardizált profilja")

if "recency_scaled" in filtered.columns:
    snake_cols = {"recency_scaled": "Recency", "frequency_scaled": "Frequency", "monetary_scaled": "Monetary"}
else:
    # Normalizálás menet közben
    for col in ["recency_days", "frequency", "monetary_total"]:
        mu, sd = filtered[col].mean(), filtered[col].std()
        filtered[f"{col}_z"] = (filtered[col] - mu) / (sd + 1e-9)
    snake_cols = {"recency_days_z": "Recency", "frequency_z": "Frequency", "monetary_total_z": "Monetary"}

snake_data = (
    filtered.groupby("Segment")[[*snake_cols.keys()]]
    .mean()
    .rename(columns=snake_cols)
    .reset_index()
)
snake_melted = snake_data.melt(id_vars="Segment", var_name="Metrika", value_name="Standardizált érték")

fig_snake = px.line(
    snake_melted,
    x="Metrika",
    y="Standardizált érték",
    color="Segment",
    color_discrete_map=SEGMENT_COLORS,
    markers=True,
    title="Szegmensek RFM profilja (standardizált értékek)",
    labels={"Standardizált érték": "Z-score"},
)
fig_snake.update_traces(line_width=2.5, marker_size=10)
fig_snake.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5)
fig_snake.update_layout(legend_title="Szegmens", height=420)
st.plotly_chart(fig_snake, use_container_width=True)

st.markdown("---")

# ── Scatter: Recency vs Monetary ──────────────────────────────────────────────
st.subheader("🗺️ Szegmenstérkép – Recency vs Monetary")
sample = filtered.sample(min(len(filtered), 1500), random_state=42)
fig_scatter = px.scatter(
    sample,
    x="recency_days",
    y="monetary_total",
    color="Segment",
    color_discrete_map=SEGMENT_COLORS,
    opacity=0.6,
    size_max=8,
    log_y=True,
    labels={"recency_days": "Recency (napok)", "monetary_total": "Monetary – log skála (£)"},
    title=f"Ügyfélpopuláció szegmensenként (minta: {len(sample):,} ügyfél)",
    hover_data=["frequency"],
)
fig_scatter.update_layout(height=480, legend_title="Szegmens")
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ── Szegmens leírások ─────────────────────────────────────────────────────────
st.subheader("📋 Szegmens leírások és javasolt akciók")
cols = st.columns(2)
for i, (seg, desc) in enumerate(SEGMENT_DESCRIPTIONS.items()):
    icon = SEGMENT_ICONS[seg]
    color = SEGMENT_COLORS[seg]
    count = filtered[filtered["Segment"] == seg].shape[0]
    with cols[i % 2]:
        st.markdown(
            f"""
            <div style="border-left: 5px solid {color}; padding: 12px 16px;
                        margin-bottom: 14px; border-radius: 4px; background: #f9f9f9;">
                <b style="font-size:1.05em;">{icon} {seg}</b>
                &nbsp;<span style="color:grey; font-size:0.9em;">({count:,} ügyfél)</span><br>
                <span style="font-size:0.95em;">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Letölthető táblázat ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📥 Részletes adattábla letöltése"):
    display_cols = [c for c in ["recency_days", "frequency", "monetary_total",
                                "return_ratio", "monetary_avg", "Segment"] if c in filtered.columns]
    csv = filtered[display_cols].reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ CSV letöltése (szűrt adatok)",
        data=csv,
        file_name="rfm_szegmensek.csv",
        mime="text/csv",
    )
    st.dataframe(filtered[display_cols].head(200), use_container_width=True)
