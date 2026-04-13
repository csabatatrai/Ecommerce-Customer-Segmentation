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
st.caption("Szegmensszintű churn-kockázat, bevételi súly és kampánycélpont-azonosítás marketingstratégia tervezéséhez.")
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

        seg_df = (
            df[df["rfm_segment"] == seg_name]
            [["rfm_segment", "churn_proba", "churn_pred", "action"]]
            .copy().reset_index().rename(columns=_RENAME)
        )
        gz_df = (
            df[(df["rfm_segment"] == seg_name) & grey_mask]
            [["rfm_segment", "churn_proba", "churn_pred", "action"]]
            .copy().reset_index().rename(columns=_RENAME)
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
            )
            st.download_button(
                label=f"🎯 Szürke zóna ({n_gz:,} fő)",
                data=_to_csv(gz_df),
                file_name=f"szurkezone_{safe}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_gz_{safe}",
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

# ── Tab 4: Céllistakészítő ────────────────────────────────────────────────────
with tab_cl:
    st.subheader("Céllistakészítő")
    st.caption(
        "Szűrd az ügyfeleket saját kampánylogikád szerint — "
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
                "Az ügyfélbázis azon részét célozza, ahol a kimenetel még nem dőlt el — "
                "sem a lojalitás, sem a lemorzsolódás nem biztos. "
                "A modell szerint ez a befolyásolható csoport adja a legjobb reaktivációs ROI-t."
            ),
            "reco": [
                "Proaktív e-mail trigger ~90 napos inaktivitást követően",
                "Mérsékelt ösztönző (5–10% kedvezmény) — ne árazd le a teljes bázist",
                "A/B teszteld az üzenet tónusát: lojalitáserősítő vs. visszatérési ösztönző",
            ],
        },
        "💎 VIP veszélyben": {
            "color":  "#ff1a3c",
            "bg":     "rgba(255,26,60,0.08)",
            "filters": "VIP Bajnokok · Churn: 50–100%",
            "desc": (
                "A legmagasabb értékű ügyfelek, akik churn-jelet mutatnak. "
                "Ez a szegmens generálja a bevétel legnagyobb részét — "
                "egyetlen VIP elveszítése aránytalanul nagy kiesést jelent."
            ),
            "reco": [
                "Személyes megkeresés (account manager, nem automatikus e-mail)",
                "Prémium megtartási ajánlat: exkluzív kedvezmény vagy korai termékbevezetés",
                "NPS/CSAT felmérés — panasz esetén azonnali eszkaláció, ne várd meg a következő vásárlást",
            ],
        },
        "💰 Értékes kockázat": {
            "color":  "#ffd740",
            "bg":     "rgba(255,215,64,0.07)",
            "filters": "Minden szegmens · Churn: 50–100% · Összköltés (CLV) ≥ medián",
            "desc": (
                "Bevétel-súlyozott prioritizálás: azok az ügyfelek kerülnek előre, "
                "akiknek elveszítése a legnagyobb kiesést okozná — szegmenstől függetlenül. "
                "Segít eldönteni, hol érdemes személyes beavatkozást alkalmazni tömeges kampány helyett."
            ),
            "reco": [
                "Személyre szabott ajánlat — ne tömeges e-mail, hanem direkt csatorna",
                "Értékalapú kupon: az ügyfél eddigi átlagos kosárméretéhez igazítva",
                "Prioritás-pontszám (churn × összköltés) alapján sorold: felső 20% = személyes megkeresés",
            ],
        },
        "⚡ Win-back": {
            "color":  "#ff8c1a",
            "bg":     "rgba(255,140,26,0.08)",
            "filters": "Lemorzsolódó / Alvó · Churn: 30–70%",
            "desc": (
                "A win-back kampány hatékonysága drasztikusan csökken teljes inaktivitás után — "
                "a 45–90 napos ablak a kritikus. "
                "A szürke zónás Lemorzsolódó ügyfelek még elérhetők, de hamarosan elvesznek."
            ),
            "reco": [
                "Időkorlátozott visszatérési kupon (10–15%, 30 napos érvényességgel)",
                "Korábbi vásárlások alapján személyre szabott termékajánlás (Top visszáruk fül: mit NE ajánlj)",
                "Szezonális reaktivációs kampány — ünnepi időszak előtt 6 héttel indítsd",
            ],
        },
        "🌱 Onboarding": {
            "color":  "#1ab4ff",
            "bg":     "rgba(26,180,255,0.07)",
            "filters": "Új / Ígéretes · Minden churn-szint",
            "desc": (
                "A 2. vásárlás a szokáskialakítás kritikus pontja — aki kétszer vásárolt, "
                "sokkal valószínűbb, hogy visszatér. "
                "Az onboarding sorozat az első 60 napban köt le, vagy nem."
            ),
            "reco": [
                "3–5 üzenetes onboarding e-mail sorozat az első vásárlástól számított 60 napon belül",
                "2. vásárlást ösztönző kupon (5–10%, 21 napos érvényességgel) a 2. üzenetben",
                "Loyalty program meghívó a 2. vásárlás után — ne hamarabb, különben értékét veszíti",
            ],
        },
        "💤 Elvesztett": {
            "color":  "#9898c0",
            "bg":     "rgba(152,152,192,0.07)",
            "filters": "Elvesztett / Inaktív · Minden churn-szint",
            "desc": (
                "A legrégebben inaktív ügyfelek — a reaktiváció valószínűsége alacsony, "
                "ezért az erőforrás-befektetés minimalizálása a cél. "
                "Egyetlen kísérlet, ha nincs visszajelzés: lista-eltávolítás."
            ),
            "reco": [
                "Egyetlen reaktivációs e-mail — ha nincs megnyitás vagy kattintás, lista-eltávolítás (ne küldd sorozatban)",
                "Alacsony értékű, széles körű promóció: szezonális kiárusítás vagy újdonság-értesítő",
                "Ne fektess személyes beavatkozást ebbe a szegmensbe — az így felszabaduló kapacitást VIP megtartásra fordítsd",
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
        f"{target['churn_proba'].mean():.1%}" if not target.empty else "—",
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
        st.download_button(
            label=f"📥 Célista letöltése — {len(target):,} ügyfél, prioritás szerint rendezve",
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
                f"Előnézet: top 20 / {len(target):,} ügyfél — "
                "a teljes, prioritás szerint rendezett lista a letöltött fájlban."
            )

st.markdown("---")

# ── Top termékek szegmensenként ───────────────────────────────────────────────
st.subheader("Top termékek szegmensenként")
st.caption(
    "TTM-időszaki tényleges vásárlások alapján — kampánytartalom és termékajánló tervezéséhez. "
    "Csak pozitív tételek (visszáru kizárva)."
)

# Aggregáció: szegmens × termék — eladás és visszáru külön
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


tabs_top = st.tabs(seg_order)

for tab, seg_name in zip(tabs_top, seg_order):
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
st.subheader("Vásárlási intenzitás – időbeli eloszlás")
st.caption("Az összes tranzakció alapján, nap és óra szerinti bontásban — kampányok és villámakciók időzítéséhez.")

_heat = df_tx.copy()
_heat["Hour"] = _heat["InvoiceDate"].dt.hour
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

st.caption("💡 A forgalom kedden és csütörtökön 10–13 óra között a legintenzívebb — villámakciókat ezekre a csúcsokra érdemes időzíteni.")

st.markdown("---")
st.caption(f"Adatforrás: churn_predictions.parquet · online_retail_ready_for_rfm.parquet · TTM: {start_ttm.strftime('%Y-%m-%d')} – {cutoff_date.strftime('%Y-%m-%d')}")
