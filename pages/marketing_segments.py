"""
pages/Marketing_szegmensek.py
Szegmensszintű marketing elemzés és kampánytervezési eszköz
"""

import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        .section-tag {{
            display: inline-block;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 5px 14px;
            border-radius: 6px;
            margin-bottom: 8px;
        }}
        .callout {{
            border-radius: 0 6px 6px 0;
            padding: 8px 14px;
            font-size: 0.85em;
            margin-bottom: 0.4rem;
        }}

        /* ── Reszponzív oszlopok ─────────────────────────────────────── */
        @media (max-width: 768px) {{
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                min-width: calc(50% - 0.5rem) !important;
                flex: 0 0 calc(50% - 0.5rem) !important;
                box-sizing: border-box;
            }}
        }}
        @media (max-width: 480px) {{
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                min-width: 100% !important;
                flex: 0 0 100% !important;
            }}
            .seg-value {{
                font-size: 20px;
            }}
            h1 {{
                font-size: 1.5rem !important;
                line-height: 1.3 !important;
            }}
        }}
        @media (max-width: 768px) and (min-width: 481px) {{
            h1 {{
                font-size: 1.8rem !important;
            }}
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
st.title("Kampánytervező marketingeseknek")
st.markdown(
    "<div style='margin-top:-1rem; color:rgba(250,250,250,0.6); font-size:0.875rem;'>"
    "Szegmensszintű churn-kockázat, bevételi súly és kampánycélpont-azonosítás marketingstratégia tervezéséhez."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Workflow bevezető ─────────────────────────────────────────────────────────
_wf_cols = st.columns([10, 1, 10, 1, 10])
with _wf_cols[0]:
    st.markdown(
        "<div style='background:rgba(255,26,60,0.1); border:1px solid rgba(255,26,60,0.25); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#ff1a3c; margin-bottom:6px;'>① KINEK</div>"
        "<div style='font-size:15px; font-weight:600; color:white; margin-bottom:8px;'>Célcsoport azonosítás</div>"
        "<div style='font-size:12px; color:rgba(200,207,232,0.65); line-height:1.65;'>"
        "Szűrd le az ügyfeleket szegmens, churn-kockázat és összköltés alapján. "
        "Kampánysablonnal vagy egyéni beállítással. A lista prioritás szerint rendezve, azonnal letölthető."
        "</div></div>",
        unsafe_allow_html=True,
    )
with _wf_cols[1]:
    st.markdown(
        "<div style='display:flex; align-items:center; justify-content:center; height:190px; "
        "font-size:22px; color:rgba(200,207,232,0.3);'>→</div>",
        unsafe_allow_html=True,
    )
with _wf_cols[2]:
    st.markdown(
        "<div style='background:rgba(26,180,255,0.08); border:1px solid rgba(26,180,255,0.22); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#1ab4ff; margin-bottom:6px;'>② MIT</div>"
        "<div style='font-size:15px; font-weight:600; color:white; margin-bottom:8px;'>Kampánytartalom</div>"
        "<div style='font-size:12px; color:rgba(200,207,232,0.65); line-height:1.65;'>"
        "A célcsoport szegmensének top termékei TTM alapján: mi szerepeljen az ajánlóban, "
        "és mit kerülj (visszáruk fül)."
        "</div></div>",
        unsafe_allow_html=True,
    )
with _wf_cols[3]:
    st.markdown(
        "<div style='display:flex; align-items:center; justify-content:center; height:190px; "
        "font-size:22px; color:rgba(200,207,232,0.3);'>→</div>",
        unsafe_allow_html=True,
    )
with _wf_cols[4]:
    st.markdown(
        "<div style='background:rgba(255,215,64,0.08); border:1px solid rgba(255,215,64,0.22); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#ffd740; margin-bottom:6px;'>③ MIKOR</div>"
        "<div style='font-size:15px; font-weight:600; color:white; margin-bottom:8px;'>Időzítés</div>"
        "<div style='font-size:12px; color:rgba(200,207,232,0.65); line-height:1.65;'>"
        "Nap és óra szerinti vásárlási csúcsok: mikor a legvalószínűbb, hogy a kampány eléri a célcsoportot."
        "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='margin-top:12px; background:rgba(200,207,232,0.04); border:1px solid rgba(200,207,232,0.12); "
    "border-radius:8px; padding:10px 20px; display:flex; align-items:center; gap:24px;'>"
    "<span style='font-size:11px; font-weight:700; letter-spacing:0.12em; color:rgba(200,207,232,0.4); white-space:nowrap;'>PRIORITÁS KÉPLETE</span>"
    "<span style='font-size:13px; color:rgba(200,207,232,0.75); white-space:nowrap;'>"
    "<b style='color:rgba(200,207,232,0.95);'>churn valószínűség</b>"
    "<span style='color:rgba(200,207,232,0.35); margin:0 8px;'>×</span>"
    "<b style='color:rgba(200,207,232,0.95);'>összköltés (CLV)</b>"
    "</span>"
    "<span style='font-size:11px; color:rgba(200,207,232,0.4);'>—</span>"
    "<span style='font-size:12px; color:rgba(200,207,232,0.55);'>"
    "A letölthető listákban a legnagyobb kockázatú és legértékesebb ügyfelek kerülnek előre"
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)

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

seg_order = list(seg_stats["rfm_segment"])

# ── Főtabs ────────────────────────────────────────────────────────────────────
st.markdown(
    "<span class='section-tag' style='background:rgba(255,26,60,0.15); color:#ff1a3c;'>"
    "① KINEK: célcsoport azonosítás</span>",
    unsafe_allow_html=True,
)
st.subheader("Kampánytervező")
tab_cl, tab_seg, tab_bubble, tab_grey = st.tabs([
    "⚙️ Céllistakészítő",
    "📋 Szegmensek áttekintése",
    "📈 Churn vs. bevétel",
    "🎯 Szürke zóna",
])

# ── Tab 1: Szegmens KPI kártyák + névsorok ───────────────────────────────────
with tab_seg:
    st.subheader("Szegmensek áttekintése")

    _RENAME = {
        "Customer ID": "Ügyfél azonosító",
        "rfm_segment": "Szegmens",
        "churn_proba": "Churn valószínűség",
        "churn_pred":  "Churn előrejelzés (1=igen)",
        "action":      "Javasolt akció",
    }

    def _safe(name: str) -> str:
        import re as _re_safe
        name = _re_safe.sub(
            r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF"
            r"\U0001FA00-\U0001FAFF\U0000FE00-\U0000FE0F]+",
            "", name,
        ).strip()
        return (name.replace(" ", "_").replace("/", "-")
                    .replace("á","a").replace("é","e").replace("í","i")
                    .replace("ó","o").replace("ö","o").replace("ő","o")
                    .replace("ú","u").replace("ü","u").replace("ű","u"))

    def _to_csv(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    cols = st.columns(len(seg_stats))
    for col, (_, row) in zip(cols, seg_stats.iterrows()):
        color    = row["color"]
        seg_name = row["rfm_segment"]
        safe     = _safe(seg_name)

        def _ranked(frame):
            out = (
                frame
                .assign(prioritas=lambda x: (x["churn_proba"] * x["monetary_total"]).round(1))
                .sort_values("prioritas", ascending=False)
                .drop(columns=["prioritas", "monetary_total"])
                .reset_index()
                .rename(columns=_RENAME)
            )
            out.insert(0, "#", range(1, len(out) + 1))
            return out

        seg_df = _ranked(
            df[df["rfm_segment"] == seg_name]
            [["rfm_segment", "churn_proba", "churn_pred", "monetary_total", "action"]]
            .copy()
        )
        gz_df = _ranked(
            df[(df["rfm_segment"] == seg_name) & grey_mask]
            [["rfm_segment", "churn_proba", "churn_pred", "monetary_total", "action"]]
            .copy()
        )
        n_gz = len(gz_df)

        with col:
            st.markdown(
                f"<div class='seg-card' style='border-top: 3px solid {color};'>"
                f"<div class='seg-label' style='color:{color};'>{seg_name}</div>"
                f"<div class='seg-value'>{int(row['n_customers']):,} fő</div>"
                f"<div class='seg-sub'>TTM bevétel: £{row['ttm_revenue']/1000:,.0f}k "
                f"({row['revenue_share']:.1f}%)</div>"
                f"<div class='seg-sub'>Átl. churn valószínűség: {row['avg_churn_prob']:.1%}</div>"
                f"<div class='seg-sub'>Előrejelzett churner: {int(row['n_predicted_churn']):,} fő "
                f"({row['churn_rate']:.1f}%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                label=f"📥 Teljes névsor ({len(seg_df):,} fő)",
                data=_to_csv(seg_df),
                file_name=f"szegmens_{safe}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{safe}",
                help="Prioritás = churn valószínűség × összköltés (CLV) – a legnagyobb kockázatú és értékű ügyfelek kerülnek elöre",
            )
            st.download_button(
                label=f"🎯 Szürke zóna ({n_gz:,} fő)",
                data=_to_csv(gz_df),
                file_name=f"szurkezone_{safe}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_gz_{safe}",
                help="Prioritás = churn valószínűség × összköltés (CLV) – a legnagyobb kockázatú és értékű ügyfelek kerülnek elöre",
                disabled=(n_gz == 0),
            )

# ── Tab 2: Bevételi súly vs. churn kockázat – buborékdiagram ─────────────────
with tab_bubble:
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

# ── Tab 3: Kampánycélpont-elemzés: Szürke Zóna szegmensenként ────────────────
with tab_grey:
    st.subheader("Kampánycélpont-elemzés: szürke zóna (30–70% közti churn-valószínűségűek)")
    st.markdown(
        "<div class='callout' style='background:rgba(200,207,232,0.05); border-left:3px solid rgba(200,207,232,0.3);'>"
        "💡 A <b>30–70% közötti churn-valószínűségű</b> ügyfelek a leginkább befolyásolható kimenetelűek, "
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

# ── Tab 4: Céllistakészítő ────────────────────────────────────────────────────
with tab_cl:
    st.subheader("Céllistakészítő")
    st.caption(
        "Szűrd az ügyfeleket saját kampánylogikád szerint. "
        "a lista prioritás szerint rendezve, azonnal letölthető."
    )

    _MAX_RECENCY  = int(df["recency_days"].max())
    _MAX_MONETARY = int(df["monetary_total"].max())
    _MAX_RETURN   = round(float(df["return_ratio"].max()), 2)
    _MED_MONETARY = int(df["monetary_total"].quantile(0.5))

    _CL_DEFAULTS = {
        "cl_seg":      seg_order,
        "cl_churn":    (0.0, 1.0),
        "cl_recency":  _MAX_RECENCY,
        "cl_monetary": 0,
        "cl_return":   _MAX_RETURN,
    }
    for _k, _v in _CL_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Kampánysablon-definíciók (label, szűrőértékek, magyarázat)
    _PRESET_INFO = {
        "🎯 Szürke zóna": {
            "color":  "rgba(200,207,232,0.85)",
            "bg":     "rgba(200,207,232,0.07)",
            "filters": "Minden szegmens · Churn: 30–70%",
            "desc": (
                "Az ügyfélbázis azon részét célozza, ahol a kimenetel még nem dőlt el, "
                "sem a lojalitás, sem a lemorzsolódás nem biztos. "
                "A modell szerint ez a befolyásolható csoport adja a legjobb reaktivációs ROI-t."
            ),
            "reco": [
                "Proaktív e-mail trigger ~90 napos inaktivitást követően",
                "Mérsékelt ösztönző (5–10% kedvezmény), ne árazd le a teljes bázist",
                "A/B teszteld az üzenet tónusát: lojalitáserősítő vs. visszatérési ösztönző",
            ],
        },
        "💎 VIP veszélyben": {
            "color":  "#ff1a3c",
            "bg":     "rgba(255,26,60,0.08)",
            "filters": "VIP Bajnokok · Churn: 50–100%",
            "desc": (
                "A legmagasabb értékű ügyfelek, akik churn-jelet mutatnak. "
                "Ez a szegmens generálja a bevétel legnagyobb részét, "
                "egyetlen VIP elveszítése aránytalanul nagy kiesést jelent."
            ),
            "reco": [
                "Személyes megkeresés (account manager, nem automatikus e-mail)",
                "Prémium megtartási ajánlat: exkluzív kedvezmény vagy korai termékbevezetés",
                "NPS/CSAT felmérés: panasz esetén azonnali eszkaláció, ne várd meg a következő vásárlást",
            ],
        },
        "💰 Értékes kockázat": {
            "color":  "#ffd740",
            "bg":     "rgba(255,215,64,0.07)",
            "filters": "Minden szegmens · Churn: 50–100% · Összköltés (CLV) ≥ medián",
            "desc": (
                "Bevétel-súlyozott prioritizálás: azok az ügyfelek kerülnek előre, "
                "akiknek elveszítése a legnagyobb kiesést okozná, szegmenstől függetlenül. "
                "Segít eldönteni, hol érdemes személyes beavatkozást alkalmazni tömeges kampány helyett."
            ),
            "reco": [
                "Személyre szabott ajánlat, ne tömeges e-mail, hanem direkt csatorna",
                "Értékalapú kupon: az ügyfél eddigi átlagos kosárméretéhez igazítva",
                "Prioritás-pontszám (churn × összköltés) alapján sorold: felső 20% = személyes megkeresés",
            ],
        },
        "⚡ Win-back": {
            "color":  "#ff8c1a",
            "bg":     "rgba(255,140,26,0.08)",
            "filters": "Lemorzsolódó / Alvó · Churn: 30–70%",
            "desc": (
                "A win-back kampány hatékonysága drasztikusan csökken teljes inaktivitás után. "
                "a 45–90 napos ablak a kritikus. "
                "A szürke zónás Lemorzsolódó ügyfelek még elérhetők, de hamarosan elvesznek."
            ),
            "reco": [
                "Időkorlátozott visszatérési kupon (10–15%, 30 napos érvényességgel)",
                "Korábbi vásárlások alapján személyre szabott termékajánlás (Top visszáruk fül: mit NE ajánlj)",
                "Szezonális reaktivációs kampány: ünnepi időszak előtt 6 héttel indítsd",
            ],
        },
        "🌱 Onboarding": {
            "color":  "#1ab4ff",
            "bg":     "rgba(26,180,255,0.07)",
            "filters": "Új / Ígéretes · Minden churn-szint",
            "desc": (
                "A 2. vásárlás a szokáskialakítás kritikus pontja: aki kétszer vásárolt, "
                "sokkal valószínűbb, hogy visszatér. "
                "Az onboarding sorozat az első 60 napban köt le, vagy nem."
            ),
            "reco": [
                "3–5 üzenetes onboarding e-mail sorozat az első vásárlástól számított 60 napon belül",
                "2. vásárlást ösztönző kupon (5–10%, 21 napos érvényességgel) a 2. üzenetben",
                "Loyalty program meghívó a 2. vásárlás után, ne hamarabb, különben értékét veszíti",
            ],
        },
        "💤 Elvesztett": {
            "color":  "#9898c0",
            "bg":     "rgba(152,152,192,0.07)",
            "filters": "Elvesztett / Inaktív · Minden churn-szint",
            "desc": (
                "A legrégebben inaktív ügyfelek, a reaktiváció valószínűsége alacsony, "
                "ezért az erőforrás-befektetés minimalizálása a cél. "
                "Egyetlen kísérlet, ha nincs visszajelzés: lista-eltávolítás."
            ),
            "reco": [
                "Egyetlen reaktivációs e-mail: ha nincs megnyitás vagy kattintás, lista-eltávolítás (ne küldd sorozatban)",
                "Alacsony értékű, széles körű promóció: szezonális kiárusítás vagy újdonság-értesítő",
                "Ne fektess személyes beavatkozást ebbe a szegmensbe, az így felszabaduló kapacitást VIP megtartásra fordítsd",
            ],
        },
    }

    _CAMPAIGN_PRESETS = [
        (_label, {**_CL_DEFAULTS, **{
            "🎯 Szürke zóna":      {"cl_churn": (0.3, 0.7)},
            "💎 VIP veszélyben":   {"cl_seg": ["VIP Bajnokok"], "cl_churn": (0.5, 1.0)},
            "💰 Értékes kockázat": {"cl_churn": (0.5, 1.0), "cl_monetary": _MED_MONETARY},
            "⚡ Win-back":         {"cl_seg": ["Lemorzsolódó / Alvó"], "cl_churn": (0.3, 0.7)},
            "🌱 Onboarding":       {"cl_seg": ["Új / Ígéretes"]},
            "💤 Elvesztett":       {"cl_seg": ["Elvesztett / Inaktív"]},
        }[_label]})
        for _label in _PRESET_INFO
    ]

    if "cl_active_preset" not in st.session_state:
        st.session_state["cl_active_preset"] = None

    # ── Preset gombok ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:rgba(200,207,232,0.45); "
        "letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px;'>"
        "Kampánysablonok</div>",
        unsafe_allow_html=True,
    )

    for _row_presets in [_CAMPAIGN_PRESETS[:3], _CAMPAIGN_PRESETS[3:]]:
        _row_cols = st.columns(3)
        for _col, (_label, _vals) in zip(_row_cols, _row_presets):
            with _col:
                if st.button(_label, use_container_width=True, key=f"preset_{_label}"):
                    for _k, _v in _vals.items():
                        st.session_state[_k] = _v
                    st.session_state["cl_active_preset"] = _label
                    st.rerun()

    if st.button("🔄 Visszaállítás", use_container_width=True,
                 type="secondary", key="preset_reset"):
        for _k, _v in _CL_DEFAULTS.items():
            st.session_state[_k] = _v
        st.session_state["cl_active_preset"] = None
        st.rerun()

    # ── Aktív sablon magyarázókártya ──────────────────────────────────────────
    _active = st.session_state.get("cl_active_preset")
    if _active and _active in _PRESET_INFO:
        _pi = _PRESET_INFO[_active]
        _reco_html = "".join(
            f"<li style='margin-bottom:4px;'>{r}</li>" for r in _pi["reco"]
        )
        st.markdown(
            f"<div style='background:{_pi['bg']}; border:1px solid rgba(255,255,255,0.1); "
            f"border-left:4px solid {_pi['color']}; border-radius:0 8px 8px 0; "
            f"padding:14px 18px; margin:10px 0 14px 0;'>"
            f"<div style='font-size:13px; font-weight:700; color:{_pi['color']}; margin-bottom:6px;'>"
            f"{_active}</div>"
            f"<div style='font-size:12px; color:rgba(200,207,232,0.55); margin-bottom:8px;'>"
            f"Aktív szűrők: {_pi['filters']}</div>"
            f"<div style='font-size:13px; color:rgba(200,207,232,0.85); line-height:1.6; margin-bottom:10px;'>"
            f"{_pi['desc']}</div>"
            f"<div style='font-size:12px; font-weight:600; color:{_pi['color']}; margin-bottom:4px;'>"
            f"Javasolt lépések:</div>"
            f"<ul style='margin:0; padding-left:18px; color:rgba(200,207,232,0.8); "
            f"font-size:13px; line-height:1.7;'>{_reco_html}</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )

    f_seg = st.multiselect(
        "Szegmensek",
        options=seg_order,
        key="cl_seg",
    )

    _sl_r1c1, _sl_r1c2 = st.columns(2)
    _sl_r2c1, _sl_r2c2 = st.columns(2)

    with _sl_r1c1:
        f_churn = st.slider(
            "Churn valószínűség",
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            key="cl_churn",
        )
    with _sl_r1c2:
        f_recency = st.slider(
            "Max. inaktivitás (nap)",
            min_value=1, max_value=_MAX_RECENCY,
            key="cl_recency",
        )
    with _sl_r2c1:
        f_monetary = st.slider(
            "Min. összköltés (CLV) (£)",
            min_value=0, max_value=_MAX_MONETARY, step=50,
            key="cl_monetary",
        )
    with _sl_r2c2:
        f_return = st.slider(
            "Max. visszáru-arány",
            min_value=0.0, max_value=_MAX_RETURN, step=0.05, format="%.2f",
            key="cl_return",
        )

    target = df[
        df["rfm_segment"].isin(f_seg) &
        df["churn_proba"].between(f_churn[0], f_churn[1]) &
        (df["recency_days"] <= f_recency) &
        (df["monetary_total"] >= f_monetary) &
        (df["return_ratio"] <= f_return)
    ].copy()
    target["prioritas"] = (target["churn_proba"] * target["monetary_total"]).round(1)
    target = target.sort_values("prioritas", ascending=False)

    km1, km2, km3 = st.columns(3)
    km1.metric("Célcsoport mérete", f"{len(target):,} fő")
    km2.metric(
        "Becsült veszélyes bevétel",
        f"£{target['prioritas'].sum():,.0f}" if not target.empty else "£0",
        help="Szűrt ügyfelek churn_proba × monetary_total összege",
    )
    km3.metric(
        "Átl. churn valószínűség",
        f"{target['churn_proba'].mean():.1%}" if not target.empty else "-",
    )

    if target.empty:
        st.info("A szűrési feltételekre nincs találat.")
    else:
        _PREVIEW_COLS = {
            "Customer ID":   "Ügyfél azonosító",
            "rfm_segment":   "Szegmens",
            "prioritas":     "Prioritás (churn × £)",
            "churn_proba":   "Churn valószínűség",
            "recency_days":  "Inaktivitás (nap)",
            "monetary_total":"Összköltés (CLV) (£)",
            "return_ratio":  "Visszáru-arány",
            "action":        "Javasolt akció",
        }
        dl_target = (
            target[list(_PREVIEW_COLS.keys())[1:]]
            .reset_index()
            .rename(columns=_PREVIEW_COLS)
        )
        dl_target.insert(0, "#", range(1, len(dl_target) + 1))
        st.download_button(
            label=f"📥 Célista letöltése: {len(target):,} ügyfél, prioritás szerint rendezve",
            data=_to_csv(dl_target),
            file_name="celcsoport_lista.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_target",
        )

        preview = (
            target[list(_PREVIEW_COLS.keys())[1:]]
            .head(20)
            .reset_index()
            .rename(columns=_PREVIEW_COLS)
        )
        st.dataframe(
            preview,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ügyfél azonosító":    st.column_config.TextColumn("Ügyfél azonosító"),
                "Szegmens":            st.column_config.TextColumn("Szegmens"),
                "Prioritás (churn × £)": st.column_config.NumberColumn(
                    "Prioritás (churn × £)", format="%.1f"),
                "Churn valószínűség":  st.column_config.ProgressColumn(
                    "Churn valószínűség", format="%.2f", min_value=0, max_value=1),
                "Inaktivitás (nap)":   st.column_config.NumberColumn(
                    "Inaktivitás (nap)", format="%d"),
                "Összköltés (CLV) (£)":       st.column_config.NumberColumn(
                    "Összköltés (CLV) (£)", format="£%.0f"),
                "Visszáru-arány":      st.column_config.NumberColumn(
                    "Visszáru-arány", format="%.2f"),
                "Javasolt akció":      st.column_config.TextColumn("Javasolt akció"),
            },
        )
        if len(target) > 20:
            st.caption(
                f"Előnézet: top 20 / {len(target):,} ügyfél, "
                "a teljes, prioritás szerint rendezett lista a letöltött fájlban."
            )

st.markdown("---")

# ── Top termékek szegmensenként ───────────────────────────────────────────────
st.markdown(
    "<span class='section-tag' style='background:rgba(26,180,255,0.12); color:#1ab4ff;'>"
    "② MIT: kampánytartalom és termékajánló</span>",
    unsafe_allow_html=True,
)
st.subheader("Top termékek szegmensenként")
st.caption(
    "TTM-időszaki tényleges vásárlások alapján. kampánytartalom és termékajánló tervezéséhez. "
    "Csak pozitív tételek (visszáru kizárva)."
)

# Aggregáció: szegmens × termék, eladás és visszáru külön
tx_sales   = tx_ttm[(tx_ttm["Quantity"] > 0) & (tx_ttm["LineTotal"] > 0)]
tx_returns = tx_ttm[(tx_ttm["Quantity"] < 0) & (tx_ttm["LineTotal"] < 0)]

seg_product = (
    tx_sales.groupby(["rfm_segment", "Description"])
    .agg(forgalom=("Quantity", "sum"), bevetel=("LineTotal", "sum"))
    .reset_index()
)
seg_returns = (
    tx_returns.groupby(["rfm_segment", "Description"])
    .agg(visszaruzott=("Quantity", "sum"), visszaru_ertek=("LineTotal", "sum"))
    .assign(
        visszaruzott   = lambda d: d["visszaruzott"].abs(),
        visszaru_ertek = lambda d: d["visszaru_ertek"].abs(),
    )
    .reset_index()
)

RETURN_COLOR = "#e07b20"


def _rgba(hex6, alpha):
    return f"rgba({int(hex6[1:3],16)},{int(hex6[3:5],16)},{int(hex6[5:7],16)},{alpha:.2f})"


def _bar_colors(hex6, n):
    return [_rgba(hex6, 0.40 + 0.60 * i / max(n - 1, 1)) for i in range(n - 1, -1, -1)]


def _draw_bar(df_top, x_col, bar_texts, x_title, color):
    fig = go.Figure(go.Bar(
        x=df_top[x_col],
        y=df_top["label"],
        orientation="h",
        marker=dict(color=_bar_colors(color, len(df_top)), line=dict(width=0)),
        text=bar_texts,
        textposition="outside",
        textfont=dict(color="rgba(200,207,232,0.85)", size=11),
        hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLT,
        font=dict(color="white"),
        xaxis=dict(title=x_title, color=TICK_CLR, gridcolor=GRID_CLR, tickprefix="£"),
        yaxis=dict(color=TICK_CLR, gridcolor="rgba(0,0,0,0)",
                   categoryorder="total ascending", tickfont=dict(size=11)),
        height=380, margin=dict(l=10, r=110, t=10, b=30), showlegend=False,
    )
    return fig


_seg_for_tabs = st.session_state.get("cl_seg", seg_order) or seg_order
st.caption(
    f"Megjelenített szegmensek: {', '.join(_seg_for_tabs)}. "
    "A kiválasztás a Céllistakészítőben módosítható."
)
tabs_top = st.tabs(_seg_for_tabs)

for tab, seg_name in zip(tabs_top, _seg_for_tabs):
    seg_color     = SEG_COLORS.get(seg_name, "#9898c0")
    seg_total_rev = ttm_by_seg.get(seg_name, 1)

    top10 = (
        seg_product[seg_product["rfm_segment"] == seg_name]
        .sort_values("bevetel", ascending=False)
        .head(10).reset_index(drop=True)
    )
    top10["arany"] = top10["bevetel"] / seg_total_rev * 100
    top10["label"] = top10["Description"].str.slice(0, 38).str.strip()

    ret10 = (
        seg_returns[seg_returns["rfm_segment"] == seg_name]
        .sort_values("visszaru_ertek", ascending=False)
        .head(10).reset_index(drop=True)
    )
    ret10["arany"] = ret10["visszaru_ertek"] / seg_total_rev * 100
    ret10["label"] = ret10["Description"].str.slice(0, 38).str.strip()

    with tab:
        sub_sales, sub_ret = st.tabs(["📦 Top eladások", "↩️ Top visszáruk"])

        with sub_sales:
            if top10.empty:
                st.info("Nincs TTM-eladás ebben a szegmensben.")
            else:
                st.plotly_chart(
                    _draw_bar(
                        top10, "bevetel",
                        [f"£{v/1000:.1f}k  ({a:.1f}%)" for v, a in zip(top10["bevetel"], top10["arany"])],
                        "TTM bevétel (£)", seg_color,
                    ),
                    use_container_width=True,
                )
                tbl_s = top10[["Description", "forgalom", "bevetel", "arany"]].copy()
                tbl_s.insert(0, "#", range(1, len(tbl_s) + 1))
                st.dataframe(tbl_s, hide_index=True, use_container_width=True,
                    column_config={
                        "#":           st.column_config.NumberColumn("#", width=40),
                        "Description": st.column_config.TextColumn("Termék neve", width="large"),
                        "forgalom":    st.column_config.NumberColumn("Forgalom (db)", format="%d", width="small"),
                        "bevetel":     st.column_config.NumberColumn("Bevétel (£)", format="£%.0f", width="small"),
                        "arany":       st.column_config.ProgressColumn(
                            "Szegmens-arány", format="%.1f%%", min_value=0, max_value=100, width="medium"),
                    },
                )

        with sub_ret:
            if ret10.empty:
                st.info("Nincs rögzített visszáru ebben a szegmensben.")
            else:
                st.plotly_chart(
                    _draw_bar(
                        ret10, "visszaru_ertek",
                        [f"£{v/1000:.1f}k  ({a:.1f}%)" for v, a in zip(ret10["visszaru_ertek"], ret10["arany"])],
                        "Visszáru értéke (£)", RETURN_COLOR,
                    ),
                    use_container_width=True,
                )
                tbl_r = ret10[["Description", "visszaruzott", "visszaru_ertek", "arany"]].copy()
                tbl_r.insert(0, "#", range(1, len(tbl_r) + 1))
                st.dataframe(tbl_r, hide_index=True, use_container_width=True,
                    column_config={
                        "#":              st.column_config.NumberColumn("#", width=40),
                        "Description":    st.column_config.TextColumn("Termék neve", width="large"),
                        "visszaruzott":   st.column_config.NumberColumn("Visszáruzott (db)", format="%d", width="small"),
                        "visszaru_ertek": st.column_config.NumberColumn("Visszáru értéke (£)", format="£%.0f", width="small"),
                        "arany":          st.column_config.ProgressColumn(
                            "Szegmens-arány", format="%.1f%%", min_value=0, max_value=100, width="medium"),
                    },
                )

st.markdown("---")

# ── Vásárlási intenzitás: időbeli eloszlás ────────────────────────────────────
st.markdown(
    "<span class='section-tag' style='background:rgba(255,215,64,0.12); color:#ffd740;'>"
    "③ MIKOR: időzítés és csúcsidőszakok</span>",
    unsafe_allow_html=True,
)
st.subheader("Vásárlási intenzitás – időbeli eloszlás")

_heat_cids = target.index if not target.empty else pd.Index([])
_heat      = df_tx[df_tx["Customer ID"].isin(_heat_cids)].copy()

_f_seg_mikor = st.session_state.get("cl_seg", seg_order)
st.caption(
    f"A céllistán szereplő {len(_heat_cids):,} ügyfél tranzakciói alapján "
    f"({', '.join(_f_seg_mikor) if _f_seg_mikor else 'mind'}, aktuális szűrőkkel), "
    "nap és óra szerinti bontásban, kampányok és villámakciók időzítéséhez."
)

_heat["Hour"]      = _heat["InvoiceDate"].dt.hour
_heat["DayOfWeek"] = _heat["InvoiceDate"].dt.day_name()
_hu_days = ["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"]
_day_map = dict(zip(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    _hu_days,
))
_heat["Nap"] = _heat["DayOfWeek"].map(_day_map)

_hm_data  = _heat.groupby(["Nap", "Hour"])["Invoice"].nunique().reset_index()
_hm_pivot = _hm_data.pivot(index="Nap", columns="Hour", values="Invoice").fillna(0)
_hm_pivot = _hm_pivot.reindex([d for d in _hu_days if d in _hm_pivot.index])

# ── Szegmens-szintű heatmap (teljes szegmens, stabilabb prior) ────────────────
_seg_cids_full    = df[df["rfm_segment"].isin(_f_seg_mikor)].index if _f_seg_mikor else df.index
_heat_seg         = df_tx[df_tx["Customer ID"].isin(_seg_cids_full)].copy()
_heat_seg["Hour"]      = _heat_seg["InvoiceDate"].dt.hour
_heat_seg["DayOfWeek"] = _heat_seg["InvoiceDate"].dt.day_name()
_heat_seg["Nap"]       = _heat_seg["DayOfWeek"].map(_day_map)
_seg_hm_data  = _heat_seg.groupby(["Nap", "Hour"])["Invoice"].nunique().reset_index()
_seg_hm_pivot = _seg_hm_data.pivot(index="Nap", columns="Hour", values="Invoice").fillna(0)
_seg_hm_pivot = _seg_hm_pivot.reindex([d for d in _hu_days if d in _seg_hm_pivot.index])

# ── Csúcsidő-elemzés algoritmus ───────────────────────────────────────────────
if not _hm_pivot.empty and _hm_pivot.values.sum() > 0:
    # ── Súlyozott szintézis: célcsoport + szegmens prior ─────────────────────
    # Kis céllistán a mintazaj magas → a szegmens-szintű eloszlás stabilabb prior.
    # Shrinkage weight: lineáris interpoláció 30 és 300 ügyfél között.
    _n_target   = len(_heat_cids)
    _n_seg_full = len(_seg_cids_full)
    _W_MIN, _W_MAX = 30, 300          # célcsoport-méret küszöbök
    if _n_target >= _W_MAX:
        _w_target = 0.85
    elif _n_target >= _W_MIN:
        _w_target = 0.20 + 0.65 * (_n_target - _W_MIN) / (_W_MAX - _W_MIN)
    else:
        _w_target = 0.20
    _w_seg = 1.0 - _w_target

    # Konfidencia-szint szövegesen
    if _w_target >= 0.70:
        _conf_label = "magas"
    elif _w_target >= 0.45:
        _conf_label = "közepes"
    else:
        _conf_label = "alacsony"

    _synthesis_note = (
        f"Célcsoport: {_n_target} fő (súly: {_w_target:.0%}), "
        f"szegmens prior: {_n_seg_full} fő (súly: {_w_seg:.0%}) – "
        f"jelszintű megbízhatóság: {_conf_label}"
    )

    # Normalizált óra- és napdisztribúciók, majd súlyozott összevonás
    _all_hours = list(range(24))

    _t_hour = _hm_pivot.sum(axis=0).reindex(_all_hours, fill_value=0).astype(float)
    _s_hour = (_seg_hm_pivot.sum(axis=0).reindex(_all_hours, fill_value=0).astype(float)
               if not _seg_hm_pivot.empty else pd.Series(0.0, index=_all_hours))

    _t_hour_n = _t_hour / _t_hour.sum() if _t_hour.sum() > 0 else _t_hour
    _s_hour_n = _s_hour / _s_hour.sum() if _s_hour.sum() > 0 else _s_hour
    _comb_hour = (_w_target * _t_hour_n + _w_seg * _s_hour_n)

    _t_day = _hm_pivot.sum(axis=1).reindex(_hu_days, fill_value=0).astype(float)
    _s_day = (_seg_hm_pivot.sum(axis=1).reindex(_hu_days, fill_value=0).astype(float)
              if not _seg_hm_pivot.empty else pd.Series(0.0, index=_hu_days))

    _t_day_n = _t_day / _t_day.sum() if _t_day.sum() > 0 else _t_day
    _s_day_n = _s_day / _s_day.sum() if _s_day.sum() > 0 else _s_day
    _comb_day = (_w_target * _t_day_n + _w_seg * _s_day_n).sort_values(ascending=False)

    # ── Csúcsok a kombinált eloszlásból ──────────────────────────────────────
    _top_days   = list(_comb_day.head(2).index)
    _total_tx   = float(_hm_pivot.values.sum())
    _top2_pct   = float(_comb_day.head(2).sum())   # normalizált → arány

    _hour_rolling = _comb_hour.rolling(3, center=True, min_periods=1).mean()
    _peak_hour    = int(_hour_rolling.idxmax())
    _peak_start   = max(0, _peak_hour - 1)
    _peak_end     = min(23, _peak_hour + 1)

    # Hétvége arány a kombinált napdisztribúcióból
    _wknd_set    = {"Szombat", "Vasárnap"}
    _weekend_pct = float(sum(_comb_day.get(d, 0) for d in _wknd_set))

    # Bimodális detektálás a kombinált óraeloszláson
    _hr_norm     = (_hour_rolling / _hour_rolling.max()).fillna(0)
    _peaks_found = []
    for _h in range(1, 23):
        if (_hr_norm.get(_h, 0) >= 0.65
                and _hr_norm.get(_h, 0) >= _hr_norm.get(_h - 1, 0)
                and _hr_norm.get(_h, 0) >= _hr_norm.get(_h + 1, 0)):
            if not _peaks_found or (_h - _peaks_found[-1]) >= 4:
                _peaks_found.append(_h)
            elif _hr_norm.get(_h, 0) > _hr_norm.get(_peaks_found[-1], 0):
                _peaks_found[-1] = _h
    _is_bimodal = len(_peaks_found) >= 2

    # E-mail küldési idő: 1,5h a csúcs előtt, min 6:00
    def _fmt_send(_hf):
        _hh = max(6, int(_hf))
        _mm = 30 if (_hf - int(_hf)) >= 0.5 else 0
        return f"{_hh:02d}:{_mm:02d}"

    _send_time   = _fmt_send(_peak_hour - 1.5)
    _send_time_2 = _fmt_send(_peaks_found[1] - 1.5) if _is_bimodal else None

    # Vásárlói mód + javasolt csatorna
    if 6 <= _peak_hour <= 9:
        _peak_mode   = "reggeli döntési mód"
        _channel     = "push értesítés / SMS"
        _peak_tactic = "rövid, lényegre törő üzenet – reggeli rutin közben nyitja meg"
    elif 9 <= _peak_hour <= 12:
        _peak_mode   = "délelőtti kutatási mód"
        _channel     = "e-mail"
        _peak_tactic = "tartalomgazdag ajánlat termékösszehasonlítással – aktív keresési fázis"
    elif 12 <= _peak_hour <= 14:
        _peak_mode   = "ebédidős impulzusvásárlás"
        _channel     = "push értesítés"
        _peak_tactic = "közvetlen CTA, időkorlátozott ajánlat – rövid döntési ablak"
    elif 14 <= _peak_hour <= 17:
        _peak_mode   = "délutáni böngészési mód"
        _channel     = "e-mail"
        _peak_tactic = "vizuális, inspiráló termékbemutatás – hosszabb figyelmi idő"
    elif 17 <= _peak_hour <= 20:
        _peak_mode   = "esti relaxált vásárlás"
        _channel     = "e-mail / push"
        _peak_tactic = "storytelling, bundle ajánlatok – a vásárlónak van ideje dönteni"
    else:
        _peak_mode   = "éjszakai / kora reggeli böngészés"
        _channel     = "push értesítés"
        _peak_tactic = "exkluzív early bird vagy flash sale – ritka, figyelemfelkeltő"

    _days_str   = " és ".join(d.lower() for d in _top_days[:2])
    _insight_md = (
        f"💡 A kombinált elemzés szerint ({_synthesis_note}) a célcsoport "
        f"forgalma **{_days_str}** napon **{_peak_start}:00–{_peak_end+1}:00** "
        f"között a legintenzívebb (*{_peak_mode}*). "
        f"Javasolt küldési idő: **{_send_time}**"
        + (f", második hullám: **{_send_time_2}**" if _is_bimodal and _send_time_2 else "")
        + "."
    )

    _mikor_data = {
        "top_days":        _top_days,
        "peak_start":      _peak_start,
        "peak_end":        _peak_end,
        "peak_mode":       _peak_mode,
        "peak_tactic":     _peak_tactic,
        "send_time":       _send_time,
        "send_time_2":     _send_time_2,
        "channel":         _channel,
        "weekend_pct":     _weekend_pct,
        "top2_day_pct":    _top2_pct,
        "is_bimodal":      _is_bimodal,
        "target_size":     _n_target,
        "seg_size":        _n_seg_full,
        "w_target":        _w_target,
        "conf_label":      _conf_label,
        "synthesis_note":  _synthesis_note,
    }
else:
    _top_days    = ["-"]
    _peak_start  = 0
    _peak_end    = 0
    _peak_mode   = "-"
    _peak_tactic = "-"
    _channel     = "-"
    _days_str    = "-"
    _insight_md  = "💡 Nincs elegendő tranzakciós adat a kiválasztott céllistához."
    _mikor_data  = {
        "top_days": ["-"], "peak_start": 0, "peak_end": 0,
        "peak_mode": "-", "peak_tactic": "-",
        "send_time": "-", "send_time_2": None,
        "channel": "-", "weekend_pct": 0.0, "top2_day_pct": 0.0,
        "is_bimodal": False, "target_size": 0, "seg_size": 0,
        "w_target": 0.0, "conf_label": "-", "synthesis_note": "-",
    }

tab_hm_2d, tab_hm_3d = st.tabs(["📊 2D Hőtérkép", "⛰️ 3D Csúcsmodell"])

with tab_hm_2d:
    fig_2d = px.imshow(
        _hm_pivot,
        color_continuous_scale="Reds",
        aspect="auto",
        labels=dict(x="Óra", y="Nap", color="Tranzakció (db)"),
    )
    fig_2d.update_layout(
        paper_bgcolor=CHART_BG,
        font=dict(color="white"),
        coloraxis_colorbar=dict(tickfont=dict(color="white")),
        xaxis=dict(color=TICK_CLR),
        yaxis=dict(color=TICK_CLR),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_2d, use_container_width=True)

with tab_hm_3d:
    fig_3d = go.Figure(data=[go.Surface(
        z=_hm_pivot.values,
        x=list(_hm_pivot.columns),
        y=list(_hm_pivot.index),
        colorscale="Reds",
        hovertemplate="<b>Nap:</b> %{y}<br><b>Idő:</b> %{x}:00<br><b>Tranzakciók:</b> %{z}<extra></extra>",
    )])
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="Óra (0–24)", ticksuffix=" óra"),
            yaxis=dict(title="Hét napja"),
            zaxis=dict(title="Tranzakciók (db)", ticksuffix=" db"),
            aspectratio=dict(x=1.5, y=1.5, z=0.5),
        ),
        paper_bgcolor=CHART_BG,
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown(_insight_md)

st.markdown("---")

# ── Kampányterv összefoglalója ────────────────────────────────────────────────
st.markdown(
    "<span class='section-tag' style='background:rgba(255,255,255,0.06); color:rgba(200,207,232,0.6);'>"
    "Kampányterv összefoglalója</span>",
    unsafe_allow_html=True,
)
st.subheader("Kampányterv összefoglalója")
st.caption("Az aktív szűrők alapján. a három dimenzió eredménye egy helyen.")

_active_preset = st.session_state.get("cl_active_preset")
_f_seg_now     = st.session_state.get("cl_seg", seg_order)
_f_churn_now   = st.session_state.get("cl_churn", (0.0, 1.0))

# Top 3 termék a kiválasztott szegmensekre
_sum_top3 = (
    seg_product[seg_product["rfm_segment"].isin(_f_seg_now)]
    .groupby("Description")["bevetel"].sum()
    .nlargest(3)
    .reset_index()
)
_top3_html = "".join(
    f"<div style='padding:3px 0; font-size:13px; color:rgba(200,207,232,0.85);'>"
    f"<span style='color:#1ab4ff; font-weight:700;'>#{i+1}</span>&nbsp; "
    f"{row['Description'][:45]} "
    f"<span style='color:rgba(200,207,232,0.45); font-size:11px;'>£{row['bevetel']/1000:.1f}k</span>"
    f"</div>"
    for i, (_, row) in enumerate(_sum_top3.iterrows())
) if not _sum_top3.empty else "<div style='color:rgba(200,207,232,0.4); font-size:12px;'>Nincs adat a szűréshez.</div>"

_target_size  = len(target) if not target.empty else 0
_avg_churn    = f"{target['churn_proba'].mean():.0%}" if not target.empty else "-"
_seg_labels   = ", ".join(_f_seg_now) if _f_seg_now else "-"
_preset_badge = (
    f"<span style='background:rgba(255,255,255,0.08); border-radius:4px; "
    f"padding:2px 8px; font-size:11px;'>{_active_preset}</span> "
    if _active_preset else ""
)

_sum_c1, _sum_c2, _sum_c3 = st.columns(3)
with _sum_c1:
    st.markdown(
        "<div style='background:rgba(255,26,60,0.1); border:1px solid rgba(255,26,60,0.22); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#ff1a3c; margin-bottom:10px;'>① KINEK</div>"
        f"{_preset_badge}"
        f"<div style='font-size:28px; font-weight:700; color:white; margin:8px 0 2px 0;'>{_target_size:,} fő</div>"
        f"<div style='font-size:12px; color:rgba(200,207,232,0.55); margin-bottom:6px;'>Átl. churn: {_avg_churn}</div>"
        f"<div style='font-size:12px; color:rgba(200,207,232,0.65);'>{_seg_labels}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
with _sum_c2:
    st.markdown(
        "<div style='background:rgba(26,180,255,0.08); border:1px solid rgba(26,180,255,0.22); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#1ab4ff; margin-bottom:10px;'>② MIT</div>"
        "<div style='font-size:13px; font-weight:600; color:white; margin-bottom:10px;'>Top 3 ajánlott termék</div>"
        f"{_top3_html}"
        "</div>",
        unsafe_allow_html=True,
    )
with _sum_c3:
    st.markdown(
        "<div style='background:rgba(255,215,64,0.08); border:1px solid rgba(255,215,64,0.22); "
        "border-radius:10px; padding:18px 20px; height:190px; box-sizing:border-box; overflow:hidden;'>"
        "<div style='font-size:11px; font-weight:700; letter-spacing:0.15em; color:#ffd740; margin-bottom:10px;'>③ MIKOR</div>"
        "<div style='font-size:13px; font-weight:600; color:white; margin-bottom:10px;'>Optimális időzítés</div>"
        f"<div style='font-size:13px; color:rgba(200,207,232,0.85); line-height:1.7;'>"
        f"📅 <b>{_days_str}</b><br>"
        f"🕙 <b>{_peak_start}:00 – {_peak_end+1}:00</b><br>"
        f"<span style='font-size:11px; color:rgba(200,207,232,0.5);'>{_peak_mode}</span>"
        "</div>"
        f"<div style='font-size:12px; color:rgba(200,207,232,0.55); margin-top:8px;'>{_peak_tactic}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Kampánybrief PDF export ───────────────────────────────────────────────────
def _generate_brief_pdf(
    active_preset, seg_labels, target_size, avg_churn_str,
    churn_range, top_products_df, mikor_data, preset_info_dict, gen_date,
) -> bytes:
    import io as _io

    from reportlab.lib.units import cm
    from reportlab.lib import colors as rc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        KeepInFrame,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib as mpl

    import re as _re
    def _pt(text: str) -> str:
        """Emoji-k és DejaVuSans által nem támogatott karakterek eltávolítása."""
        return _re.sub(
            r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF"
            r"\U0001FA00-\U0001FAFF\U0000FE00-\U0000FE0F]+",
            "", str(text),
        ).strip()

    try:
        fp  = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        fpb = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
        pdfmetrics.registerFont(TTFont("DV",  str(fp)))
        pdfmetrics.registerFont(TTFont("DVB", str(fpb)))
    except Exception:
        pass  # már regisztrálva

    def _s(name, size=10, leading=14, color=rc.black,
           bold=False, align=TA_LEFT):
        return ParagraphStyle(
            name, fontName="DVB" if bold else "DV",
            fontSize=size, leading=leading,
            textColor=color, alignment=align,
        )

    RED  = rc.HexColor("#A81022")
    BLUE = rc.HexColor("#1565a0")
    GOLD = rc.HexColor("#9a6f00")
    LGRAY = rc.HexColor("#f5f5f7")
    MGRAY = rc.HexColor("#cccccc")

    S = {
        "title":   _s("title",  size=18, bold=True,  color=rc.HexColor("#1a1a2e"), align=TA_CENTER, leading=22),
        "sub":     _s("sub",    size=8,  color=rc.HexColor("#666666"), align=TA_CENTER),
        "sec_hdr": _s("sec_hdr",size=10, bold=True,  color=rc.white,  leading=13),
        "body":    _s("body",   size=9,  leading=12),
        "body_b":  _s("body_b", size=9,  bold=True,  leading=12),
        "small":   _s("small",  size=7.5,color=rc.HexColor("#888888"), leading=10),
        "tbl_hdr": _s("tbl_hdr",size=8.5,bold=True,  color=rc.HexColor("#222222"), align=TA_CENTER),
        "tbl_val": _s("tbl_val",size=8.5,leading=11),
        "reco":    _s("reco",   size=8.5,leading=11, color=rc.HexColor("#333333")),
    }

    def _sec_header(text, color):
        return Table(
            [[Paragraph(text, S["sec_hdr"])]],
            colWidths=["100%"],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), color),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        )

    def _kv_table(rows, col1_w=6):
        return Table(
            rows,
            colWidths=[col1_w * cm, None],
            style=TableStyle([
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [rc.white, LGRAY]),
                ("TOPPADDING",     (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
                ("LEFTPADDING",    (0, 0), (-1, -1), 6),
                ("GRID",           (0, 0), (-1, -1), 0.3, MGRAY),
            ]),
        )

    story = []

    # Fejléc
    story += [
        Paragraph("Marketing kampányterv", S["title"]),
        Spacer(1, 3),
        Paragraph(
            f"Generálva: {gen_date}  |  Sablon: {_pt(active_preset) if active_preset else 'Egyéni szűrés'}",
            S["sub"],
        ),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=2, color=RED),
        Spacer(1, 8),
    ]

    # ① KINEK
    story += [
        _sec_header("(1) KINEK: Célcsoport", RED),
        Spacer(1, 4),
        _kv_table([
            [Paragraph("Szegmensek",              S["body_b"]), Paragraph(seg_labels,    S["body"])],
            [Paragraph("Célcsoport mérete",        S["body_b"]), Paragraph(f"{target_size:,} fő", S["body"])],
            [Paragraph("Átl. churn valószínűség", S["body_b"]), Paragraph(avg_churn_str, S["body"])],
            [Paragraph("Churn szűrő",             S["body_b"]),
             Paragraph(f"{churn_range[0]:.0%} – {churn_range[1]:.0%}", S["body"])],
        ]),
        Spacer(1, 8),
    ]

    # ② MIT
    story += [_sec_header("(2) MIT: Top ajánlott termékek", BLUE), Spacer(1, 4)]
    if top_products_df.empty:
        story.append(Paragraph("Nincs termékadata a szűréshez.", S["body"]))
    else:
        tbl_rows = [[
            Paragraph("#",              S["tbl_hdr"]),
            Paragraph("Termék neve",    S["tbl_hdr"]),
            Paragraph("TTM bevétel",    S["tbl_hdr"]),
        ]]
        for i, (_, row) in enumerate(top_products_df.head(5).iterrows()):
            tbl_rows.append([
                Paragraph(str(i + 1),                  S["tbl_val"]),
                Paragraph(row["Description"][:55],     S["tbl_val"]),
                Paragraph(f"£{row['bevetel']:,.0f}",   S["tbl_val"]),
            ])
        story.append(Table(
            tbl_rows,
            colWidths=[1 * cm, None, 3 * cm],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  rc.HexColor("#e8e8ed")),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  rc.HexColor("#333333")),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rc.white, LGRAY]),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("GRID",          (0, 0), (-1, -1), 0.3, MGRAY),
            ]),
        ))
    story.append(Spacer(1, 8))

    # ③ MIKOR – adatvezérelt, céllistára szabott időzítési elemzés
    _md = mikor_data
    _m_days_str  = " és ".join(d.lower() for d in _md["top_days"][:2])
    _m_peak_str  = f"{_md['peak_start']}:00 – {_md['peak_end'] + 1}:00"
    _m_conc_str  = f"{_md['top2_day_pct']:.0%} a célcsoport forgalmából"
    _m_wknd_str  = f"{_md['weekend_pct']:.0%}"

    _m_conf_str = (
        f"célcsoport {_md['target_size']} fő ({_md['w_target']:.0%})"
        f" + szegmens {_md['seg_size']} fő ({1 - _md['w_target']:.0%})"
        f" – megbízhatóság: {_md['conf_label']}"
    )

    story += [
        _sec_header("(3) MIKOR: Optimális időzítés", GOLD),
        Spacer(1, 6),
        _kv_table([
            [Paragraph("Elemzés alapja",       S["body_b"]),
             Paragraph(_m_conf_str, S["body"])],
            [Paragraph("Csúcsnapok",           S["body_b"]),
             Paragraph(f"{_m_days_str}  ({_m_conc_str})", S["body"])],
            [Paragraph("Csúcsidőszak",         S["body_b"]),
             Paragraph(_m_peak_str, S["body"])],
            [Paragraph("Javasolt küldési idő", S["body_b"]),
             Paragraph(
                 _md["send_time"]
                 + (f"  +  második hullám: {_md['send_time_2']}" if _md.get("send_time_2") else ""),
                 S["body"],
             )],
            [Paragraph("Javasolt csatorna",    S["body_b"]),
             Paragraph(_md["channel"], S["body"])],
            [Paragraph("Vásárlói mód",         S["body_b"]),
             Paragraph(_md["peak_mode"], S["body"])],
            [Paragraph("Hétvégi aktivitás",    S["body_b"]),
             Paragraph(_m_wknd_str, S["body"])],
        ]),
        Spacer(1, 6),
    ]

    # Kampánystruktúra javaslat – 3 sor egyetlen bekezdésként (sortörőkkel)
    _day1 = _md["top_days"][0].lower() if _md["top_days"] else "-"
    _day2 = _md["top_days"][1].lower() if len(_md["top_days"]) >= 2 else _day1

    _b1 = (f"1. Elsődleges küldés: {_day1}, {_md['send_time']} – {_md['channel']}. "
           f"Üzenet: {_md['peak_tactic']}")
    if _md.get("is_bimodal") and _md.get("send_time_2"):
        _b2 = (f"2. Bimodális csúcsminta: {_day1}, második hullám {_md['send_time_2']}-kor "
               f"– eltérő tárgysort, azonos ajánlat")
    else:
        _b2 = (f"2. Emlékeztető: {_day2}, {_md['send_time']} – eltérő tárgysort, "
               f"urgency-elemmel (pl. 'még X órán belül')")
    if _md["weekend_pct"] > 0.35:
        _b3 = (f"3. Hétvégi aktivitás ({_m_wknd_str}) szignifikáns – önálló hétvégi kampány javasolt")
    elif _md["weekend_pct"] < 0.15:
        _b3 = (f"3. Hétvégi aktivitás ({_m_wknd_str}) alacsony – a büdzsét érdemes hétköznapokra koncentrálni")
    else:
        _b3 = f"3. Taktika: {_md['peak_tactic']}"

    story += [
        Paragraph("Kampánystruktúra javaslat:", S["body_b"]),
        Spacer(1, 3),
        Paragraph(
            f"{_pt(_b1)}<br/>{_pt(_b2)}<br/>{_pt(_b3)}",
            S["reco"],
        ),
    ]

    # ── Javasolt lépések ─────────────────────────────────────────────────────
    if active_preset and active_preset in preset_info_dict:
        # Sablon-alapú javaslatok
        _reco_title  = f"Javasolt lépések: {_pt(active_preset)}"
        _reco_lines  = "<br/>".join(
            f"- {_pt(r)}" for r in preset_info_dict[active_preset]["reco"]
        )
    else:
        # ── Adatvezérelt, egyéni szűrőkre szabott javaslatok ─────────────
        #
        # Ha a felhasználó nem választott sablont, hanem kézzel állított be
        # szűrőket, a rendszer 3 egymástól független dimenzió mentén generál
        # egy-egy javaslatot:
        #
        #   Bullet 1 – MIKOR / CSATORNA
        #     Forrás: mikor_data (szintézis-algoritmus eredménye)
        #     Logika: csatorna típusa és viselkedési mód kombinációja alapján
        #             6 ág; prioritás: push > SMS > esti > délelőtti kutatási >
        #             reggeli döntési > ebédidős impulzus > default
        #
        #   Bullet 2 – CHURN-SZINT / SZEGMENS-STRATÉGIA
        #     Forrás: churn_range, avg_churn_str, seg_labels
        #     Logika: churn közép + szegmens-típus kombinációja alapján 7 ág;
        #             prioritás: magas churn+VIP > magas churn+Lemorzsolódó >
        #             közepes churn > alacsony churn+Új > Elvesztett >
        #             vegyes szegmens > default
        #
        #   Bullet 3 – VÉGREHAJTÁS / MÉRET / TERMÉK
        #     Forrás: target_size, top_products_df, mikor_data (hétvége, bimodális)
        #     Logika: célcsoport mérete és speciális heatmap-jellemzők alapján
        #             6 ág; prioritás: <50 fő > >1000 fő > hétvégi dominancia >
        #             top termék elérhető > bimodális > default
        #
        # Minden ág valós adatot (target_size, churn %, termék, küldési idő)
        # ágyaz a szövegbe, és iparági benchmark-értékekkel támaszt alá.
        # ─────────────────────────────────────────────────────────────────

        _md          = mikor_data
        _churn_lo, _churn_hi = churn_range
        _churn_mid   = (_churn_lo + _churn_hi) / 2
        _segs_lower  = seg_labels.lower()
        _is_vip      = "vip" in _segs_lower
        _is_lost     = "elvesztett" in _segs_lower or "inaktív" in _segs_lower
        _is_new      = "ígéretes" in _segs_lower or "új" in _segs_lower
        _is_churn_sg = "lemorzsolódó" in _segs_lower or "alvó" in _segs_lower
        _multi_seg   = "," in seg_labels
        _channel     = _md.get("channel", "e-mail")
        _peak_mode   = _md.get("peak_mode", "")
        _send_t      = _md.get("send_time", "09:00")
        _day1        = (_md["top_days"][0].lower()
                        if _md.get("top_days") and _md["top_days"][0] != "-" else None)
        _wknd_pct    = _md.get("weekend_pct", 0.0)
        _top_prod    = (top_products_df.iloc[0]["Description"][:42]
                        if not top_products_df.empty else None)

        # ── Bullet 1: MIKOR / CSATORNA ────────────────────────────────────
        # Forrás: mikor_data["channel"] + mikor_data["peak_mode"]
        # Az algoritmus a célcsoport heatmap-alapú csúcsidő-elemzéséből
        # (szintézis: target + szegmens prior) határozza meg a csatornát és
        # a viselkedési módot; ezeket kombinálja az alábbi 6 ág.
        if "push" in _channel and "e-mail" not in _channel:
            _r1 = (
                f"Push értesítést {_day1 + ', ' if _day1 else ''}{_send_t}-ra ütemezve küldj "
                f"({_peak_mode}): maximum 2 mondat, egyetlen CTA, közvetlen landing-link – "
                f"a push-csatornán a szöveg terjedelme kritikusan befolyásolja a megnyitási arányt"
            )
        elif "sms" in _channel.lower():
            _r1 = (
                f"SMS-kampányhoz {_day1 + ', ' if _day1 else ''}{_send_t} az optimális küldési idő "
                f"({_peak_mode}): max. 160 karakter, személyes megszólítás + egyetlen URL "
                f"– SMS open rate 98%, ezért a link mögötti oldal konverziójára fókuszálj, ne az üzenet hosszára"
            )
        elif "esti" in _peak_mode or "relaxált" in _peak_mode:
            _r1 = (
                f"E-mailt {_day1 + ', ' if _day1 else ''}{_send_t}-ra időzítve küldj – "
                f"a célcsoport esti, relaxált böngészési módban van: vizuálisan gazdag sablon, "
                f"storytelling-bevezető, bundle-ajánlat; kerüld a sürgősséget hangsúlyozó tárgysorokat ({target_size:,} fős listán a visual-first megközelítés ~18-24%-kal jobb CTR-t hoz)"
            )
        elif "kutatási" in _peak_mode or "délelőtt" in _peak_mode:
            _r1 = (
                f"E-mailt {_day1 + ', ' if _day1 else ''}{_send_t}-ra ütemezve küldj "
                f"({_peak_mode}): tartalomgazdag levél termékösszehasonlítással vagy részletes leírással – "
                f"a célcsoport aktívan keres, tehát a több információ magasabb elköteleződést hoz; "
                f"A/B tesztelj 2 tárgysor-variánst a {target_size:,} fős lista 10%-án (n={max(10, int(target_size * 0.1)):,} fő/variáns)"
            )
        elif "reggeli" in _peak_mode or "döntési" in _peak_mode:
            _r1 = (
                f"Reggeli döntési mód: {_send_t}-ra ütemezett {_channel} – "
                f"tömör tárgysor (max. 6 szó), az ajánlat az első 3 sorban legyen látható, "
                f"egyetlen gomb/CTA; a célcsoport reggeli rutinjába illeszkedő, 30 másodperc alatt feldolgozható üzenet konvertál"
            )
        elif "ebéd" in _peak_mode or "impulzus" in _peak_mode:
            _r1 = (
                f"Ebédidős impulzusvásárlás: {_send_t}-ra ({_channel}) küldve a döntési ablak 45-60 perc – "
                f"időkorlátozott ajánlatot helyezz az üzenet elejére ('csak ma', 'ma éjfélig'), "
                f"az ajánlat azonnali döntést igénylő, egyszerű termék legyen"
            )
        else:
            _r1 = (
                f"Optimális küldési idő: {_day1 + ', ' if _day1 else ''}{_send_t} ({_channel}, {_peak_mode}) – "
                f"a heatmap {_md.get('target_size', 0):,} tranzakció alapján számolódott; "
                f"erre az időablakra optimalizált küldéssel 10-25%-kal magasabb megnyitási arány érhető el "
                f"az iparági benchmarkhoz képest"
            )

        # ── Bullet 2: CHURN-SZINT / SZEGMENS-STRATÉGIA ───────────────────
        # Forrás: churn_range (UI csúszka), avg_churn_str (célcsoport átlag),
        #         seg_labels (kiválasztott szegmensek neve)
        # A churn közepe (_churn_mid) és a szegmens-típus flag-ek kombinációja
        # alapján 7 ág határozza meg a kampány megtartási vagy reaktivációs
        # jellegét, a megfelelő ösztönző erősséggel és csatornával együtt.
        if _churn_mid >= 0.65 and _is_vip:
            _r2 = (
                f"Kritikusan magas churn-rizikójú VIP csoport ({avg_churn_str} átlag, {_churn_lo:.0%}-{_churn_hi:.0%} szűrő): "
                f"ne automatizált tömeges e-mailt küldj – személyes megkeresés (account manager, telefonhívás vagy kézzel írt levél) "
                f"drasztikusan jobb megtartási arányt hoz; NPS-kérdéssel kombinálva az eszkaláció is kezelhető"
            )
        elif _churn_mid >= 0.60 and _is_churn_sg:
            _r2 = (
                f"Magas churn-szintű Lemorzsolódó célcsoport ({avg_churn_str} átl. valószínűség): "
                f"30 napos visszatérési kupon (10-15%), és ha 2 héten belül nincs kattintás, "
                f"automatikus lista-szegmentálás – ne folytasd a sorozatot, a {target_size:,} fős lista "
                f"nem reagáló tagjainál az újabb üzenetek tovább rontják a domain reputation-t"
            )
        elif _churn_mid >= 0.50 and not _is_lost:
            _r2 = (
                f"Az {avg_churn_str}-os átlagos churn valószínűségnél proaktív trigger ajánlott: "
                f"ha egy ügyfél {_churn_lo:.0%} fölé kerül és 45+ napja nem vásárolt, "
                f"automatikusan induljon egy 2-3 üzenetes reaktivációs sorozat – "
                f"ne várd meg a teljes inaktivitást, a beavatkozás hatékonysága exponenciálisan csökken 90 nap után"
            )
        elif _is_new and _churn_mid < 0.40:
            _r2 = (
                f"Alacsony churn-rizikójú Új/Ígéretes szegmens ({avg_churn_str}): a 2. vásárlás ösztönzése a kritikus cél – "
                f"5-10%-os, 21 napos érvényességű kupon a 2. üzenetben; "
                f"aki egyszer vásárolt és 60 napon belül visszatér, ~3× nagyobb valószínűséggel lesz visszatérő vásárló"
            )
        elif _is_lost:
            _r2 = (
                f"Elvesztett/Inaktív szegmens – egyetlen reaktivációs üzenet szabálya: "
                f"ha nincs megnyitás 14 napon belül, automatikus lista-eltávolítás; "
                f"a {target_size:,} fős listán a {avg_churn_str}-os átlagos churn mellett "
                f"a megtartási erőforrást érdemes VIP-megtartásra átirányítani"
            )
        elif _multi_seg:
            _r2 = (
                f"Vegyes szegmens-mix ({seg_labels}): egységes üzenet helyett szegmensenként eltérő tárgysor kötelező – "
                f"a magas churn-rizikójúaknak ({_churn_lo:.0%}–{_churn_hi:.0%}) urgency-elem, "
                f"az alacsonyabb rizikójúaknak értékalapú kommunikáció; "
                f"a szegmentált küldés átlagosan 14-22%-kal javítja az open rate-et az egységes üzenethez képest"
            )
        else:
            _r2 = (
                f"Churn-tartomány: {_churn_lo:.0%}-{_churn_hi:.0%} (átlag: {avg_churn_str}) – "
                f"ennél a szintnél az optimális ösztönző mérete 8-12%; ennél nagyobb margint veszít, "
                f"kisebb nem motivál cselekvésre; a küszöb az iparági RFM-irodalom alapján, "
                f"nem általános webshop-átlag"
            )

        # ── Bullet 3: VÉGREHAJTÁS / MÉRET / TERMÉK ───────────────────────
        # Forrás: target_size (célcsoport mérete a szűrők után),
        #         top_products_df (célcsoport saját top termékei),
        #         mikor_data["weekend_pct"] + mikor_data["is_bimodal"]
        # 6 ág, prioritás sorrendben: extrém kis lista (kézi perszonalizáció)
        # > nagy lista (A/B kötelező) > hétvégi dominancia + heatmap csúcsnap
        # > top termék elérhető > bimodális aktivitás > default szekvencia.
        # A target_size és _n_test értékek közvetlenül a szűrt céllistából
        # számolódnak, nem becsültek.
        if target_size < 50:
            _r3 = (
                f"Kis célcsoport ({target_size:,} fő): személyre szabott, kézzel írt megkeresés – "
                f"{'top termék: ' + _top_prod + '; ' if _top_prod else ''}"
                f"a méret lehetővé teszi a magas personalisation-szintet minimális időköltséggel; "
                f"névvel megszólítva, egyéni vásárlási előzmény megemlítésével az ilyen listákon 3-5× magasabb konverziós arány érhető el"
            )
        elif target_size > 1000:
            _n_test = max(100, int(target_size * 0.1))
            _r3 = (
                f"Nagy célcsoport ({target_size:,} fő): kötelező A/B tesztelni tárgysorokat és CTA szövegét – "
                f"{_n_test:,} fő/variáns elegendő a statisztikai szignifikanciához; "
                f"a nyerő verziót 48 órán belül a maradék {target_size - 2*_n_test:,} főnek küldve "
                f"maximalizálod az eredményt anélkül, hogy az egész lista a gyengébb variánst kapja"
            )
        elif _wknd_pct > 0.35 and _day1 in ("szombat", "vasárnap"):
            _r3 = (
                f"A célcsoport {_wknd_pct:.0%}-a hétvégén aktív ({_day1}): lazább, inspiráló tónus, "
                f"kisebb sürgősség, vizuális hangsúly – "
                f"{'top termék: ' + _top_prod + '; ' if _top_prod else ''}"
                f"hétvégi lifestyle-kontextusban az ajánlat jobban teljesít mint az urgency-alapú hétköznapi üzenet"
            )
        elif _top_prod:
            _r3 = (
                f"A célcsoport saját top terméke: '{_top_prod}' – "
                f"ezt emeld ki az üzenet fókuszába, ne általános katalógust küldj; "
                f"szegmens-specifikus bestseller lista {target_size:,} fős listán mérve 25-35%-kal magasabb "
                f"kattintási arányt hoz a bolt-szintű termékajánlásoknál"
            )
        elif _md.get("is_bimodal") and _md.get("send_time_2"):
            _r3 = (
                f"Bimodális aktivitásminta: a célcsoport napjában kétszer aktív – "
                f"két üzenet ugyanazon a napon ({_send_t} és {_md['send_time_2']}), "
                f"eltérő tárgysorral de azonos ajánlattal; "
                f"a kétszeres elérés nem fárasztja ki a listát, ha az üzenetek vizuálisan eltérők"
            )
        else:
            _r3 = (
                f"Automatizált 3-lépéses szekvencia ajánlott ({target_size:,} fős lista): "
                f"1. küldés {_send_t or '09:00'} (értékalapú), "
                f"2. küldés +7 nap (urgency), "
                f"3. küldés +14 nap (last chance, 50%-os lefedettség a nem megnyitókra) – "
                f"minden alkalommal {_peak_mode or 'csúcsidőre'} optimalizált küldési időponttal"
            )

        _reco_title = "Javasolt lépések: egyéni szűrők alapján"
        _reco_lines = f"- {_pt(_r1)}<br/>- {_pt(_r2)}<br/>- {_pt(_r3)}"

    story += [
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=0.4, color=MGRAY),
        Spacer(1, 5),
        Paragraph(_reco_title, S["body_b"]),
        Spacer(1, 3),
        Paragraph(_reco_lines, S["reco"]),
    ]

    # ── SVG ikonok betöltése (fekete) ────────────────────────────────────────
    import tempfile, os as _os
    from svglib.svglib import svg2rlg

    def _svg_icon_black(path, size=14):
        """SVG betöltése fekete színnel, megadott méretben."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                svg_text = f.read()
            # Minden szín → fekete
            svg_text = _re.sub(r'fill\s*=\s*"(?!none)[^"]*"', 'fill="black"', svg_text)
            svg_text = _re.sub(r'stroke\s*=\s*"(?!none)[^"]*"', 'stroke="black"', svg_text)
            svg_text = _re.sub(r'fill\s*:\s*(?!none)[^;}"]+', 'fill:black', svg_text)
            svg_text = _re.sub(r'stroke\s*:\s*(?!none)[^;}"]+', 'stroke:black', svg_text)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".svg", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(svg_text)
                tmp_path = tmp.name
            d = svg2rlg(tmp_path)
            _os.unlink(tmp_path)
            if d is None or d.width == 0:
                return None
            sx, sy = size / d.width, size / d.height
            d.width, d.height = size, size
            d.transform = (sx, 0, 0, sy, 0, 0)
            return d
        except Exception:
            return None

    _ico_gh  = _svg_icon_black("src/github.svg",   14)
    _ico_li  = _svg_icon_black("src/linkedin.svg", 14)
    _ico_web = _svg_icon_black("src/website.svg",  14)

    _link_s = _s("lnk", size=10, color=rc.HexColor("#1a1a2e"), leading=15)
    _ico_fb = _s("ifb",  size=10, color=rc.black, leading=15)

    def _contact_row(ico, url, display):
        return [
            ico if ico else Paragraph("-", _ico_fb),
            Paragraph(f'<link href="{url}">{display}</link>', _link_s),
        ]

    _contact_tbl = Table(
        [
            _contact_row(_ico_gh,  "https://github.com/csabatatrai",
                         "github.com/csabatatrai"),
            _contact_row(_ico_li,  "https://www.linkedin.com/in/csabatatrai-datascientist/",
                         "linkedin.com/in/csabatatrai-datascientist"),
            _contact_row(_ico_web, "https://csabatatrai.hu/", "csabatatrai.hu"),
        ],
        colWidths=[0.6 * cm, None],
        style=TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (0, -1),  2),
            ("LEFTPADDING",   (1, 0), (1, -1),  8),
        ]),
    )

    # ── Fix canvas-alapú footer (garantált 1 oldal) ──────────────────────────
    from reportlab.graphics import renderPDF as _rpdf
    from reportlab.lib.pagesizes import A4 as _A4

    _LM         = 2 * cm
    _RM         = 2 * cm
    _FONT_SMALL = 7.5
    _FONT_LINK  = 8.5
    _ICO_SZ     = 14          # meg kell egyeznie a _svg_icon_black size paraméterével
    _GAP        = 6           # ikon és felirat közötti rés pontban
    _LINE_H     = 0.65 * cm   # sortávolság a linkek között
    _N_LINKS    = 3
    # Foglalt terület alul: 3 link + elválasztó + forrásszöveg + paddinng
    _FOOTER_H   = _N_LINKS * _LINE_H + 1.4 * cm

    _footer_icons = (_ico_gh, _ico_li, _ico_web)

    def _draw_footer(canvas, doc):
        from reportlab.graphics import renderPDF as rpdf
        import reportlab.lib.colors as rlc

        canvas.saveState()
        pw = _A4[0]

        # Linkek: top-to-bottom sorrendben tárolva, de alulról rajzolunk
        # Sorrend (felülről): LinkedIn, GitHub, csabatatrai.hu
        links = [
            (_footer_icons[2], "csabatatrai.hu",
             "https://csabatatrai.hu/"),                         # legalsó
            (_footer_icons[0], "github.com/csabatatrai",
             "https://github.com/csabatatrai"),                  # középső
            (_footer_icons[1], "Tátrai Csaba Attila",
             "https://www.linkedin.com/in/csabatatrai-datascientist/"),  # legfelső
        ]

        canvas.setFont("DV", _FONT_LINK)
        canvas.setFillColor(rlc.HexColor("#222222"))

        y_bottom = 0.35 * cm   # legalsó link baseline-ja

        for i, (ico, label, url) in enumerate(links):
            ty = y_bottom + i * _LINE_H

            if ico:
                rpdf.draw(ico, canvas, _LM, ty - _ICO_SZ * 0.2)

            tx = _LM + (_ICO_SZ + _GAP if ico else 0)
            canvas.drawString(tx, ty, label)

            tw = canvas.stringWidth(label, "DV", _FONT_LINK)
            canvas.linkURL(url, (tx, ty - 2, tx + tw, ty + _ICO_SZ - 2),
                           relative=0)

        # Elválasztó vonal + forrásszöveg a linkek fölé
        sep_y = y_bottom + _N_LINKS * _LINE_H + 0.18 * cm
        canvas.setStrokeColor(rlc.HexColor("#cccccc"))
        canvas.setLineWidth(0.4)
        canvas.line(_LM, sep_y, pw - _RM, sep_y)

        canvas.setFont("DV", _FONT_SMALL)
        canvas.setFillColor(rlc.HexColor("#aaaaaa"))
        canvas.drawString(
            _LM, sep_y + 0.12 * cm,
            "Kampánytervező marketingeseknek  ·  churn_predictions.parquet",
        )

        canvas.restoreState()

    # Dokumentum összeállítása – KeepInFrame garantálja az egyoldalasságot
    _top_m    = 2 * cm
    _bot_m    = _FOOTER_H + 0.6 * cm
    _avail_w  = _A4[0] - _LM - _RM
    _avail_h  = _A4[1] - _top_m - _bot_m

    # Ha a tartalom szorosan illeszkedik vagy nem fér el, arányosan zsugorodik
    story_frame = KeepInFrame(_avail_w, _avail_h, story, mode="shrink")

    buf2  = _io.BytesIO()
    doc2  = SimpleDocTemplate(
        buf2, pagesize=_A4,
        leftMargin=_LM, rightMargin=_RM,
        topMargin=_top_m, bottomMargin=_bot_m,
    )
    doc2.build([story_frame], onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buf2.getvalue()


# PDF export gomb
import datetime as _dt
_pdf_top5 = (
    seg_product[seg_product["rfm_segment"].isin(_f_seg_now)]
    .groupby("Description")["bevetel"].sum()
    .nlargest(5)
    .reset_index()
)
try:
    _pdf_bytes = _generate_brief_pdf(
        active_preset    = _active_preset,
        seg_labels       = _seg_labels,
        target_size      = _target_size,
        avg_churn_str    = _avg_churn,
        churn_range      = _f_churn_now,
        top_products_df  = _pdf_top5,
        mikor_data       = _mikor_data,
        preset_info_dict = _PRESET_INFO,
        gen_date         = _dt.date.today().strftime("%Y. %m. %d."),
    )
    _pdf_filename = (
        f"kampanyterv_{_safe(_active_preset) if _active_preset else 'egyeni'}"
        f"_{_dt.date.today().strftime('%Y%m%d')}.pdf"
    )
    st.download_button(
        label="📄 Kampánybrief letöltése (PDF)",
        data=_pdf_bytes,
        file_name=_pdf_filename,
        mime="application/pdf",
        use_container_width=True,
        key="dl_brief_pdf",
    )
except Exception as _pdf_err:
    st.warning(f"PDF generálás nem sikerült: {_pdf_err}")

st.markdown("---")
st.caption(f"Adatforrás: churn_predictions.parquet · online_retail_ready_for_rfm.parquet · TTM: {start_ttm.strftime('%Y-%m-%d')} – {cutoff_date.strftime('%Y-%m-%d')}")
