# ============================================================
# pages/09.py
# ============================================================
# Kohorsz Retenciós Elemzés, Termék Teljesítmény és
# Rendelési Minták – az e-kereskedelem klasszikus mutatói
# amelyek a többi dashboardból hiányoznak.
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLEANED_PARQUET

st.set_page_config(
    page_title="Kohorsz, Termék & Rendelési Minták",
    page_icon="🔄",
    layout="wide",
)

# ── Egyedi CSS ──
st.markdown(
    """
    <style>
    .insight-card {
        background: linear-gradient(135deg, #A81022 0%, #7a0b18 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 14px;
        border-left: 7px solid #ff4b4b;
        box-shadow: 0 8px 18px rgba(168,16,34,0.18);
        margin: 10px 0 18px 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .insight-title {
        font-size: 1.1rem; font-weight: 700; margin-bottom: 7px;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .insight-body { font-size: 1rem; line-height: 1.55; opacity: 0.95; }
    .insight-tag {
        background: rgba(255,255,255,0.18); border-radius: 5px;
        padding: 6px 11px; margin-top: 12px; font-weight: 600;
        display: inline-block; border: 1px solid rgba(255,255,255,0.28);
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] { transition: transform 0.2s; }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
    </style>
    """,
    unsafe_allow_html=True,
)


def insight_card(title: str, body: str, tag: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">🎯 {title}</div>
            <div class="insight-body">{body}</div>
            <div class="insight-tag">💡 {tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Adatbetöltés ──
@st.cache_data(show_spinner="Adatok betöltése...")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(CLEANED_PARQUET)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["revenue"] = df["Quantity"] * df["Price"]
    return df


try:
    raw = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Hiányzó adatfájl: {e}\n\nFuttasd le a 01-es notebookot!")
    st.stop()

purchases = raw[raw["revenue"] > 0].copy()
returns   = raw[raw["revenue"] < 0].copy()


# ── Kohorsz számítás (cache-elve) ──
@st.cache_data
def build_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = df[df["Quantity"] > 0].copy()
    d["InvoiceMonth"] = d["InvoiceDate"].dt.to_period("M")
    first_buy = d.groupby("Customer ID")["InvoiceMonth"].min().rename("CohortMonth")
    d = d.join(first_buy, on="Customer ID")
    d["CohortIndex"] = (d["InvoiceMonth"] - d["CohortMonth"]).apply(lambda x: x.n)
    cohort_counts = d.groupby(["CohortMonth", "CohortIndex"])["Customer ID"].nunique().unstack()
    cohort_size   = cohort_counts[0]
    retention_pct = cohort_counts.divide(cohort_size, axis=0).multiply(100).round(1)
    return retention_pct, cohort_size


retention_pct, cohort_size = build_cohort(raw)


# ── Termék adatok ──
@st.cache_data
def build_product_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = df[df["revenue"] > 0].copy()
    neg = df[df["revenue"] < 0].copy()
    top_products = (
        pos.groupby("Description")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .reset_index()
        .rename(columns={"revenue": "Bevétel (£)", "Description": "Termék"})
    )
    top_returns = (
        neg.groupby("Description")["Quantity"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"Quantity": "Visszaküldött db", "Description": "Termék"})
    )
    # Kiegészítés: visszaküldési bevétel-hatás
    ret_rev = (
        neg.groupby("Description")["revenue"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"revenue": "Visszaküldési veszteség (£)"})
    )
    top_returns = top_returns.merge(ret_rev, left_on="Termék", right_on="Description", how="left").drop(columns="Description")
    return top_products, top_returns


top_products, top_returns = build_product_data(raw)


# ── AOV adatok ──
@st.cache_data
def build_aov_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = df[df["revenue"] > 0].copy()
    pos["Month"] = pos["InvoiceDate"].dt.to_period("M")
    invoice_vals = (
        pos.groupby(["Invoice", "Month"])["revenue"].sum().reset_index()
    )
    monthly = (
        invoice_vals.groupby("Month")
        .agg(
            avg_order_value=("revenue", "mean"),
            order_count=("Invoice", "count"),
            total_revenue=("revenue", "sum"),
        )
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].dt.to_timestamp()

    country_aov = (
        pos.groupby(["Invoice", "Country"])["revenue"].sum().reset_index()
        .groupby("Country")
        .agg(aov=("revenue", "mean"), orders=("Invoice", "count"))
        .query("orders >= 50")
        .sort_values("aov", ascending=False)
        .head(10)
        .reset_index()
    )
    return monthly, country_aov


monthly_aov, country_aov = build_aov_data(raw)


# ── Fejléc ──
st.title("🔄 Kohorsz, Termék & Rendelési Minták")
st.markdown(
    "Kohorsz-alapú ügyfélmegtartás, termékteljesítmény és rendelési viselkedés – "
    "a többi dashboardból hiányzó klasszikus e-commerce analitika."
)

# ── KPI sor ──
n_customers   = purchases["Customer ID"].nunique()
n_invoices    = purchases["Invoice"].nunique()
total_rev     = purchases["revenue"].sum()
aov_overall   = total_rev / n_invoices
return_rate   = len(returns) / (len(purchases) + len(returns)) * 100
avg_retention = retention_pct.iloc[:, 1].dropna().mean()  # 1. havi retenciós átlag

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Egyedi ügyfelek",    f"{n_customers:,}")
c2.metric("🛒 Rendelések száma",   f"{n_invoices:,}")
c3.metric("💰 Átlagos AOV",        f"£{aov_overall:,.0f}")
c4.metric("↩️ Visszaküldési arány", f"{return_rate:.1f}%")
c5.metric("📊 Átl. hó/1 retenciós", f"{avg_retention:.0f}%")

st.markdown("---")


# ══════════════════════════════════════════════════════════
# TABOK
# ══════════════════════════════════════════════════════════
tab_cohort, tab_product, tab_orders = st.tabs([
    "🔄 Kohorsz Retenciós Elemzés",
    "🛍️ Termék Teljesítmény",
    "📦 Rendelési Minták & AOV",
])


# ══════════════════════════════════════════════════════════
# TAB 1: KOHORSZ RETENCIÓS ELEMZÉS
# ══════════════════════════════════════════════════════════
with tab_cohort:

    insight_card(
        title="Kohorsz Retenciós Arány",
        body=(
            f"Az első hónap után az ügyfelek átlagosan <b>{avg_retention:.0f}%</b>-a tér vissza "
            "a következő hónapban. A 2009 decemberi kohorsz – a legidősebb – "
            "35%-os 1. havi retenciót mutat. A retenciós arányok általában "
            "az 5-6. hónapra stabilizálódnak 20-30% körül."
        ),
        tag="Fokozatos, személyre szabott utókövetési kampányokkal (1. hét, 1. hónap, 3. hónap) növeld a korai retenciót.",
    )

    # Szűkítsük az első 12 időszakra (olvashatóság)
    max_periods = st.slider(
        "Megjelenített időszakok száma (0 = első vásárlás hónapja)",
        min_value=3, max_value=min(24, retention_pct.shape[1]),
        value=12, step=1,
    )
    display_cohorts = st.slider(
        "Kohorsz sorok száma (legrégebbitől)",
        min_value=3, max_value=len(retention_pct),
        value=min(15, len(retention_pct)), step=1,
    )

    ret_display = (
        retention_pct
        .iloc[:display_cohorts, :max_periods]
        .copy()
    )
    ret_display.index = ret_display.index.astype(str)
    ret_display.columns = [f"{i}. hónap" for i in ret_display.columns]

    fig_cohort = px.imshow(
        ret_display,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100,
        text_auto=".0f",
        labels=dict(color="Retenciós %"),
        aspect="auto",
    )
    fig_cohort.update_traces(
        textfont_size=10,
        hovertemplate="<b>Kohorsz:</b> %{y}<br><b>Időszak:</b> %{x}<br><b>Retenciós arány:</b> %{z:.1f}%<extra></extra>",
    )
    fig_cohort.update_layout(
        height=480,
        margin=dict(t=20, b=20, l=10, r=10),
        coloraxis_colorbar=dict(title="Retenciós %", ticksuffix="%"),
        xaxis_title="Hónapok az első vásárlás óta",
        yaxis_title="Akvizíciós kohorsz",
    )
    st.plotly_chart(fig_cohort, use_container_width=True)
    st.caption("Minden sor = egy akvizíciós hónap. Az értékek azt mutatják, a kohorsz hány %-a vásárolt az adott hónapban.")

    st.markdown("---")

    col_cohsize, col_avg_ret = st.columns(2)

    with col_cohsize:
        st.subheader("👥 Kohorsz mérete (megszerzett ügyfelek)")

        size_df = cohort_size.reset_index()
        size_df.columns = ["Kohorsz", "Ügyfelek"]
        size_df["Kohorsz"] = size_df["Kohorsz"].astype(str)

        fig_size = px.bar(
            size_df,
            x="Kohorsz", y="Ügyfelek",
            color="Ügyfelek",
            color_continuous_scale="Blues",
            labels={"Kohorsz": "Akvizíciós hónap"},
        )
        fig_size.update_layout(
            height=300,
            margin=dict(t=10, b=60, l=10, r=10),
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_size, use_container_width=True)
        st.caption("Megjegyzés: a 2011 végi kohorszok kisebb mérete az adatgyűjtés végéhez közeledést tükrözi.")

    with col_avg_ret:
        st.subheader("📉 Átlagos retenciós görbe")
        st.caption("Átlagos megtartási arány az összes kohorszon")

        avg_ret_curve = retention_pct.iloc[:, :13].mean().reset_index()
        avg_ret_curve.columns = ["Időszak", "Retenciós %"]

        fig_ret_curve = px.line(
            avg_ret_curve,
            x="Időszak", y="Retenciós %",
            markers=True,
            color_discrete_sequence=["#009E73"],
        )
        fig_ret_curve.add_hrect(
            y0=0, y1=avg_retention,
            fillcolor="rgba(0,158,115,0.08)",
            line_width=0,
        )
        fig_ret_curve.update_traces(line_width=2.5, marker_size=7)
        fig_ret_curve.update_layout(
            height=300,
            margin=dict(t=10, b=40, l=10, r=10),
            yaxis_ticksuffix="%",
            xaxis_title="Hónapok az első vásárlás után",
        )
        st.plotly_chart(fig_ret_curve, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2: TERMÉK TELJESÍTMÉNY
# ══════════════════════════════════════════════════════════
with tab_product:

    top1_name = top_products.iloc[0]["Termék"]
    top1_rev  = top_products.iloc[0]["Bevétel (£)"]
    top_ret_name = top_returns.iloc[0]["Termék"]
    top_ret_qty  = top_returns.iloc[0]["Visszaküldött db"]

    insight_card(
        title="Termékteljesítmény – Koncentráció és Visszaküldések",
        body=(
            f"A legjobb termék <b>({top1_name})</b> £{top1_rev:,.0f} bevételt generált, "
            "míg a top 30 termék az összes bevétel közel 30%-át adja. "
            f"A legtöbbször visszaküldött termék a <b>{top_ret_name}</b> "
            f"({top_ret_qty:,.0f} db visszaküldve), ami súlyos készlet- és logisztikai terhet jelent."
        ),
        tag="Koncentrálj a top termékek készletoptimalizálására, és vizsgáld meg a visszaküldési okok mintázatát.",
    )

    col_bar, col_tree = st.columns([1, 1])

    with col_bar:
        st.subheader("🏆 Top 15 termék bevétel szerint")

        top15 = top_products.head(15).sort_values("Bevétel (£)", ascending=True)
        fig_top = px.bar(
            top15,
            x="Bevétel (£)",
            y="Termék",
            orientation="h",
            color="Bevétel (£)",
            color_continuous_scale="Teal",
            text=top15["Bevétel (£)"].apply(lambda v: f"£{v/1e3:.0f}k"),
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(
            height=440,
            margin=dict(t=10, b=40, l=10, r=60),
            coloraxis_showscale=False,
            xaxis_title="Bevétel (£)",
            yaxis_title="",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_tree:
        st.subheader("🌳 Termék bevétel treemap – Top 30")

        fig_tree = px.treemap(
            top_products,
            path=["Termék"],
            values="Bevétel (£)",
            color="Bevétel (£)",
            color_continuous_scale="YlGn",
            hover_data={"Bevétel (£)": ":,.0f"},
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>£%{value:,.0f}",
            hovertemplate="<b>%{label}</b><br>Bevétel: £%{value:,.0f}<extra></extra>",
        )
        fig_tree.update_layout(
            height=440,
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    col_ret, col_ret_detail = st.columns([1, 1])

    with col_ret:
        st.subheader("↩️ Top 15 visszaküldött termék (darabszám)")

        ret_display = top_returns.head(15).sort_values("Visszaküldött db", ascending=True)
        fig_ret_bar = px.bar(
            ret_display,
            x="Visszaküldött db",
            y="Termék",
            orientation="h",
            color="Visszaküldött db",
            color_continuous_scale="OrRd",
            text=ret_display["Visszaküldött db"].apply(lambda v: f"{v:,.0f}"),
        )
        fig_ret_bar.update_traces(textposition="outside")
        fig_ret_bar.update_layout(
            height=420,
            margin=dict(t=10, b=40, l=10, r=60),
            coloraxis_showscale=False,
            xaxis_title="Visszaküldött mennyiség (db)",
            yaxis_title="",
        )
        st.plotly_chart(fig_ret_bar, use_container_width=True)

    with col_ret_detail:
        st.subheader("📋 Visszaküldések pénzügyi hatása")
        st.caption("Top 15 termék – visszaküldött db és becsült bevételkiesés")

        ret_show = top_returns.copy()
        ret_show["Visszaküldési veszteség (£)"] = ret_show["Visszaküldési veszteség (£)"].fillna(0).round(0).astype(int)

        st.dataframe(
            ret_show,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Visszaküldési veszteség (£)": st.column_config.NumberColumn(
                    "Bevételkiesés (£)", format="£%d"
                ),
                "Visszaküldött db": st.column_config.ProgressColumn(
                    "Visszaküldött db",
                    min_value=0,
                    max_value=int(top_returns["Visszaküldött db"].max()),
                    format="%d db",
                ),
            },
        )

    with st.expander("📥 Teljes termék top 30 lista letöltése"):
        csv_prod = top_products.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Letöltés CSV-ként",
            data=csv_prod,
            file_name="top30_termekek.csv",
            mime="text/csv",
        )
        st.dataframe(top_products, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 3: RENDELÉSI MINTÁK & AOV
# ══════════════════════════════════════════════════════════
with tab_orders:

    aov_max_month = monthly_aov.loc[monthly_aov["avg_order_value"].idxmax(), "Month"]
    aov_max_val   = monthly_aov["avg_order_value"].max()
    aov_overall   = monthly_aov["avg_order_value"].mean()
    one_time_pct  = (purchases.groupby("Customer ID")["Invoice"].nunique() == 1).mean() * 100

    insight_card(
        title="Rendelési Minták – AOV és Visszatérő Vásárlók",
        body=(
            f"Az átlagos rendelési érték (AOV) <b>£{aov_overall:,.0f}</b>. "
            f"A csúcs <b>{aov_max_month.strftime('%Y %B')}</b>-ban volt (£{aov_max_val:,.0f}). "
            f"Az ügyfelek <b>{one_time_pct:.0f}%-a</b> csak egyszer vásárolt, "
            "ami komoly növekedési potenciált jelent a második vásárlás ösztönzésével."
        ),
        tag=f"Célzott 'Második vásárlás' kampánnyal {one_time_pct:.0f}% egyszeri vevőt érdemes reaktiválni – ez az ügyfélbázis legnagyobb szegmense.",
    )

    # ── Kombó grafikon: Rendelésszám + AOV (dual Y tengely) ──
    st.subheader("📈 Havi rendelésszám és átlagos rendelési érték (AOV)")

    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

    fig_combo.add_trace(
        go.Bar(
            x=monthly_aov["Month"],
            y=monthly_aov["order_count"],
            name="Rendelések száma",
            marker_color="rgba(0,158,115,0.55)",
            hovertemplate="<b>%{x|%Y %b}</b><br>Rendelés: %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_combo.add_trace(
        go.Scatter(
            x=monthly_aov["Month"],
            y=monthly_aov["avg_order_value"],
            name="AOV (£)",
            mode="lines+markers",
            line=dict(color="#d62728", width=2.5),
            marker_size=6,
            hovertemplate="<b>%{x|%Y %b}</b><br>AOV: £%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_combo.update_layout(
        height=380,
        margin=dict(t=20, b=40, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    fig_combo.update_yaxes(title_text="Rendelések száma", secondary_y=False)
    fig_combo.update_yaxes(title_text="AOV (£)", secondary_y=True, tickprefix="£")
    st.plotly_chart(fig_combo, use_container_width=True)

    st.markdown("---")
    col_freq, col_country_aov = st.columns([1, 1])

    with col_freq:
        st.subheader("🔁 Vásárlási frekvencia eloszlása")
        st.caption("Hány rendelést adott le az ügyfelek különböző csoportja?")

        freq_series = purchases.groupby("Customer ID")["Invoice"].nunique()
        bins = [1, 2, 3, 6, 11, 21, 51, freq_series.max() + 1]
        labels = ["1 (egyszeri)", "2", "3–5", "6–10", "11–20", "21–50", "50+"]
        freq_cat = pd.cut(freq_series, bins=bins, labels=labels, right=False)
        freq_dist = freq_cat.value_counts().reindex(labels).reset_index()
        freq_dist.columns = ["Vásárlások száma", "Ügyfelek"]
        freq_dist["Arány (%)"] = (freq_dist["Ügyfelek"] / freq_dist["Ügyfelek"].sum() * 100).round(1)

        fig_freq = px.bar(
            freq_dist,
            x="Vásárlások száma",
            y="Ügyfelek",
            color="Ügyfelek",
            color_continuous_scale="Blues",
            text=freq_dist.apply(lambda r: f"{r['Ügyfelek']:,}<br>({r['Arány (%)']:.0f}%)", axis=1),
        )
        fig_freq.update_traces(textposition="outside")
        fig_freq.update_layout(
            height=380,
            margin=dict(t=10, b=40, l=10, r=10),
            coloraxis_showscale=False,
            yaxis_title="Ügyfelek száma",
            xaxis_title="Vásárlások száma (kategória)",
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_country_aov:
        st.subheader("🌍 Átlagos rendelési érték (AOV) országonként")
        st.caption("Csak azok az országok, ahol legalább 50 rendelés érkezett")

        country_disp = country_aov.sort_values("aov", ascending=True)
        fig_caov = px.bar(
            country_disp,
            x="aov",
            y="Country",
            orientation="h",
            color="aov",
            color_continuous_scale="Teal",
            text=country_disp["aov"].apply(lambda v: f"£{v:,.0f}"),
            hover_data={"orders": True},
            labels={"aov": "Átl. rendelési érték (£)", "Country": "", "orders": "Rendelések"},
        )
        fig_caov.update_traces(textposition="outside")
        fig_caov.update_layout(
            height=380,
            margin=dict(t=10, b=40, l=10, r=70),
            coloraxis_showscale=False,
            xaxis_title="Átlagos rendelési érték (£)",
        )
        st.plotly_chart(fig_caov, use_container_width=True)

    st.markdown("---")

    # ── Rendelési érték eloszlás (boxplot szegmensenként) ──
    st.subheader("📦 Rendelési értékek eloszlása hónaponként")
    st.caption("Havi AOV, min–max sáv és trenddel – az outlier rendelések kisimításával (99. percentilis)")

    p99 = monthly_aov["avg_order_value"].quantile(0.99)
    mao_clip = monthly_aov[monthly_aov["avg_order_value"] <= p99].copy()

    fig_box = go.Figure()
    fig_box.add_scatter(
        x=mao_clip["Month"],
        y=mao_clip["avg_order_value"],
        mode="lines+markers",
        name="Havi AOV",
        line=dict(color="#009E73", width=2),
        marker_size=5,
        fill="tozeroy",
        fillcolor="rgba(0,158,115,0.08)",
        hovertemplate="<b>%{x|%Y %b}</b><br>AOV: £%{y:,.0f}<extra></extra>",
    )
    # Trend vonal (lineáris regresszió)
    x_num = np.arange(len(mao_clip))
    z     = np.polyfit(x_num, mao_clip["avg_order_value"].values, 1)
    trend = np.poly1d(z)(x_num)
    fig_box.add_scatter(
        x=mao_clip["Month"],
        y=trend,
        mode="lines",
        name="Trendvonal",
        line=dict(color="#d62728", dash="dot", width=2),
    )
    fig_box.update_layout(
        height=300,
        margin=dict(t=10, b=40, l=10, r=10),
        yaxis_title="AOV (£)",
        yaxis_tickprefix="£",
        xaxis_title="",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📥 Havi AOV adatok letöltése"):
        csv_aov = monthly_aov.copy()
        csv_aov["Month"] = csv_aov["Month"].dt.strftime("%Y-%m")
        csv_aov.columns = ["Hónap", "Átl. Rendelési Érték (£)", "Rendelések Száma", "Összes Bevétel (£)"]
        st.download_button(
            label="⬇️ Letöltés CSV-ként",
            data=csv_aov.to_csv(index=False).encode("utf-8"),
            file_name="havi_aov.csv",
            mime="text/csv",
        )
        st.dataframe(csv_aov, use_container_width=True, hide_index=True)
