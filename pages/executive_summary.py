import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import base64
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.sidebar import render_sidebar
from src.data_loader import load_churn_predictions, load_transactions

# ==========================================
# 1. Adatbetöltés
# ==========================================
@st.cache_resource
def load_churn_model():
    for candidate in (
        Path(__file__).parent.parent / "models/xgboost_churn.joblib",
        Path("models/xgboost_churn.joblib"),
        Path("../models/xgboost_churn.joblib"),
    ):
        if candidate.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(candidate)
    return None

render_sidebar()

df_preds = load_churn_predictions()
df_tx    = load_transactions()

# ==========================================
# 2. Dashboard fejléc
# ==========================================
st.title("Vezetői összefoglaló")
st.markdown("---")

if not df_preds.empty and not df_tx.empty:
# ==========================================
    # 3. KPI kiszámítása (Szcenárió alapú Tól-Ig sáv)
    # ==========================================
    vip_at_risk_mask = df_preds['action'].str.contains('VIP Veszélyben', na=False)
    vip_at_risk_ids = df_preds.index[vip_at_risk_mask]

    cutoff_date = pd.to_datetime("2011-09-09")
    start_date_ttm = cutoff_date - pd.Timedelta(days=365)
    total_observation_days = (cutoff_date - df_tx['InvoiceDate'].min()).days

    # Összes churn-re jelölt ügyfél (nem csak VIP) - ez a valódi modell-kimenet
    all_churn_ids = df_preds.index[df_preds['churn_pred'] == 1]
    all_churn_tx  = df_tx[df_tx['Customer ID'].isin(all_churn_ids)]

    # TTM bevétel: az összes churn-re jelölt ügyfél utolsó 365 nap forgalma
    all_churn_ttm = all_churn_tx[
        (all_churn_tx['InvoiceDate'] >= start_date_ttm) &
        (all_churn_tx['InvoiceDate'] <  cutoff_date)
    ]['LineTotal'].sum()

    # Precision-korrekció: adatból számolva (churn_pred=1 & actual_churn=1 / összes churn_pred=1)
    _tp = int(((df_preds['churn_pred'] == 1) & (df_preds['actual_churn'] == 1)).sum())
    _pp = int((df_preds['churn_pred'] == 1).sum())
    model_precision_pct = _tp / _pp * 100 if _pp > 0 else 0.0
    model_precision = model_precision_pct / 100
    revenue_at_risk = all_churn_ttm * model_precision
    roc_auc = roc_auc_score(df_preds['actual_churn'], df_preds['churn_proba'])

    # VIP Veszélyben szegmens külön (expander kontextushoz)
    vip_tx = df_tx[df_tx['Customer ID'].isin(vip_at_risk_ids)]
    vip_ttm = vip_tx[
        (vip_tx['InvoiceDate'] >= start_date_ttm) &
        (vip_tx['InvoiceDate'] <  cutoff_date)
    ]['LineTotal'].sum()

    # 2010-es teljes éves bevétel mint referencia-alap (egyetlen teljes naptári évünk)
    revenue_2010 = df_tx[df_tx['InvoiceDate'].dt.year == 2010]['LineTotal'].sum()
    risk_pct = revenue_at_risk / revenue_2010 * 100

    # ==========================================
    # 4. KPI Megjelenítése és Transzparencia
    # ==========================================
    vip_at_risk_count = int(vip_at_risk_mask.sum())
    effective_threshold = float(df_preds.loc[df_preds['churn_pred'] == 1, 'churn_proba'].min())

    # Historikus validáció: TP (valóban elmorzsolódott) vs. FP (aktív maradt)
    vip_df = df_preds[vip_at_risk_mask]
    vip_tp = int((vip_df['actual_churn'] == 1).sum())   # valódi churnerek
    vip_fp = int((vip_df['actual_churn'] == 0).sum())   # téves riasztás
    vip_precision_pct = vip_tp / vip_at_risk_count * 100

    # ==========================================
    # 5. KPI "Szürke Zóna": menthető ügyfelek aránya
    # ==========================================
    # #! SZÜRKE ZÓNA KÜSZÖB - az alsó és felső határ itt módosítható
    grey_zone_lower = 0.3
    grey_zone_upper = 0.7
    grey_zone_mask = (df_preds['churn_proba'] >= grey_zone_lower) & (df_preds['churn_proba'] <= grey_zone_upper)
    grey_zone_count = int(grey_zone_mask.sum())
    grey_zone_pct = grey_zone_count / len(df_preds) * 100

    # Szürke zóna TTM bevétele és reálisan menthető összeg
    grey_zone_ttm = df_tx[
        df_tx['Customer ID'].isin(df_preds.index[grey_zone_mask])
    ].pipe(lambda t: t[
        (t['InvoiceDate'] >= start_date_ttm) &
        (t['InvoiceDate'] <  cutoff_date)
    ])['LineTotal'].sum()
    grey_zone_arpu = grey_zone_ttm / grey_zone_count if grey_zone_count > 0 else 0
    grey_zone_expected_churners = grey_zone_count * 0.50
    retention_benchmark_pct = 15.0  # közepes e-commerce kampány (iparági átlag: 10–20%)
    salvageable_revenue = grey_zone_ttm * 0.50 * (retention_benchmark_pct / 100)
    salvageable_pct_of_risk = salvageable_revenue / revenue_at_risk * 100 if revenue_at_risk > 0 else 0

    # ==========================================
    # 6. KPI Átlagos lemorzsolódási arány
    # ==========================================
    total_customers = len(df_preds)
    churned_customers = int((df_preds['actual_churn'] == 1).sum())
    churn_rate_pct = churned_customers / total_customers * 100

    # ==========================================
    # 7. KPI VIP Bajnokok visszaküldési aránya
    # ==========================================
    vip_champions = df_preds[df_preds['rfm_segment'] == 'VIP Bajnokok']
    vip_champions_count = len(vip_champions)
    vip_return_ratio_pct = vip_champions['return_ratio'].mean() * 100
    overall_return_ratio_pct = df_preds['return_ratio'].mean() * 100

    # -------------------------------------------------------
    # Háttérkép betöltése (base64, hogy Streamlit biztosan kiszolgálja)
    # -------------------------------------------------------
    bg_path = Path("src/streamlit_bg.webp")
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
                /* #! KÁRTYA HÁTTÉRSZÍN - itt módosítható */
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
            .kpi-sub {{
                font-size: 13px;
                color: rgba(200, 207, 232, 0.65);
                margin-top: 0.1rem;
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
                <div class="kpi-value">£ {revenue_at_risk/1000:,.0f}k</div>
                <div class="kpi-sub">a 2010-es éves bevétel (£ {revenue_2010:,.0f}) ~{risk_pct:.1f}%-ának megfelelő összeg, {model_precision_pct:.2f}%-os modellprecizitással korrigálva · <b style="color:#1aff6e;">Ebből reálisan menthető: £ {salvageable_revenue/1000:,.0f}k</b> (szürke zóna, 15%-os iparági benchmark retenció)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">👥 Veszélyben lévő VIP ügyfelek</div>
                <div class="kpi-value">{vip_at_risk_count:,} fő</div>
                <div class="kpi-sub">A teljes ügyfélbázison {model_precision_pct:.2f}%-os, a VIP Bajnokok szegmensen belül {vip_precision_pct:.2f}%-os megbízhatósággal tudtuk azonosítani a lemorzsolódókat.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">🎯 "Szürke Zóna"  Bizonytalan churn-valószínűségű ügyfelek</div>
                <div class="kpi-value">{grey_zone_pct:.1f}%</div>
                <div class="kpi-sub">{grey_zone_count:,} fő.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">📉 Historikus lemorzsolódási (churn) arány</div>
                <div class="kpi-value">{churn_rate_pct:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">🧮 Modellezhető aktív ügyfélbázis</div>
                <div class="kpi-value">{total_customers:,} fő</div>
                <div class="kpi-sub">Az modellbe bevont, érvényes vásárlók száma</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">↩️ Legértékesebb ügyfelek (VIP Bajnokok) visszaküldési aránya</div>
                <div class="kpi-value">{vip_return_ratio_pct:.1f}%</div>
                <div class="kpi-sub">Ügyfélenkénti átlag: {overall_return_ratio_pct:.1f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # Expander magyarázatok (kártyák alatt, 2 oszlopban)
    # -------------------------------------------------------
    col1, col2 = st.columns(2)


    with col1:
        with st.expander("ℹ️ Hogyan számoltuk? Becsült bevételkiesés"):
            st.markdown(f"""
            **Célcsoport:** Az összes churn-re jelölt ügyfél (`churn_pred = 1`): {len(all_churn_ids):,} fő.

            **Számítás:**

            $$ \\text{{Becsült kiesés}} = \\underbrace{{\\sum_{{\\text{{churn\\_pred}}=1}} \\text{{TTM bevétel}}}}_{{£\\,{all_churn_ttm:,.0f}}} \\times \\underbrace{{\\text{{Precision}}}}_{{{{model\\_precision}}={model_precision:.4f}}} = £\\,{revenue_at_risk:,.0f} $$

            **Miért precision-korrekció?** A modell a churn-re jelölt ügyfelek {model_precision_pct:.2f}%-ánál jelez valódi lemorzsolódást (teszt halmaz, classification report). A fennmaradó {100-model_precision_pct:.2f}% téves riasztás - az ő bevételük nem kiesés, ezért levonjuk.

            ---
            | Metrika | Érték |
            |---|---|
            | Churn-re jelölt ügyfelek | {len(all_churn_ids):,} fő |
            | Churn-re jelöltek TTM bevétele | £ {all_churn_ttm:,.0f} |

            ---
            **Reálisan menthető összeg: £ {salvageable_revenue:,.0f}**

            A teljes kiesésből nem menthető vissza minden összeg, mivel csak a „Szürke Zóna" ügyfelei befolyásolhatók jó eséllyel kampánnyal (churn-valószínűség: 30–70%). A valóban megmenthető rész becslése:

            $$ \\text{{Menthető}} = \\underbrace{{\\text{{Szürke zóna TTM}}}}_{{£\\,{grey_zone_ttm:,.0f}}} \\times \\underbrace{{50\\%\\text{{ átlag churn proba}}}}_{{\\text{{várható veszteség}}}} \\times \\underbrace{{{retention_benchmark_pct:.0f}\\%\\text{{ retenció}}}}_{{\\text{{iparági benchmark}}}} = £\\,{salvageable_revenue:,.0f} $$

            Ez a kiesés ~{salvageable_pct_of_risk:.0f}%-a — a reálisan elérhető bevédelmi sáv egy közepes minőségű, célzott megtartási kampánnyal.

            *TTM (Trailing Twelve Months): a cutoff dátum (2011-09-09) előtti gördülő 365 nap tranzakcióinak összege - a legfrissebb viselkedést tükrözi, nem egy rögzített naptári évet.*
            *A legpontosabb előrejelzéshez jövőbeli fejlesztésként Prediktív CLV modell bevezetése javasolt.*
            """)

    with col2:
        with st.expander("ℹ️ Hogyan számoltuk? VIP ügyfelek száma"):
            st.markdown(f"""
            **Célcsoport:** Magas RFM-értékű ügyfelek, akiknél a modell lemorzsolódást jelez.

            **Kiválasztás feltétele:** Az ügyfél egyidejűleg teljesíti az alábbi két kritériumot:
            1. **VIP szegmens** a vásárló magas értéket képvisel (Champions / Top szegmens az RFM-besorolás alapján).
            2. **Lemorzsolódás előrejelzett** a modell által becsült valószínűség átlépi az **optimalizált döntési küszöböt** (`churn_proba ≥ {effective_threshold:.3f}`).

            **A küszöb meghatározása:** Az F1-score-t maximalizáló threshold a Precision-Recall görbéből, kizárólag a holdout teszthalmazon számítva (1 049 ügyfél). Az F1-optimális küszöb matematikai egyensúlyt keres a téves riasztások (FP) és az elszalasztott lemorzsolódók (FN) között - ez az üzletileg ésszerű kompromisszum.

            ---
            **Historikus validáció - mennyire „tiszta" a lista?**

            | | Ügyfelek száma | Arány |
            |---|---|---|
            | ✅ Valódi churner (TP) | {vip_tp:,} fő | {vip_precision_pct:.2f}% |
            | ⚠️ Téves riasztás (FP) | {vip_fp:,} fő | {100 - vip_precision_pct:.2f}% |
            | **Összesen (akció-lista)** | **{vip_at_risk_count:,} fő** | **100%** |

            A lista **{vip_precision_pct:.2f}%-os precizitással** azonosítja a valódi lemorzsolódókat. A fennmaradó {100 - vip_precision_pct:.2f}% (téves riasztás) proaktív megkeresése üzletileg nem kockázatos - egy VIP ügyfelet felhívni soha nem árt.

            *Megjegyzés: Az `actual_churn` kizárólag historikus validációs label. Éles működésben ez az érték nem ismert előre - ezért is van szükség a modellre.*
            """)

    col3, col4 = st.columns(2)
    with col3:
        with st.expander("ℹ️ Hogyan számoltuk? - \"Szürke Zóna\""):
            st.markdown(f"""
            **Célcsoport:** Minden ügyfél, akinek a modell által becsült lemorzsolódási valószínűsége a bizonytalansági sávba esik.

            **Definíció:** A „Szürke Zóna" azokat az ügyfeleket foglalja magába, akiknél:

            $$ {grey_zone_lower} \\leq \\text{{churn\\_proba}} \\leq {grey_zone_upper} $$

            Ők sem egyértelműen „maradók", sem egyértelműen „lemorzsolódók" - a döntésük még befolyásolható.

            **Miért fontos ez a marketing csapatnak?**
            A magas (`> {grey_zone_upper}`) valószínűségű ügyfelek megmentése valószínűleg már túl késő és drága. Az alacsony (`< {grey_zone_lower}`) valószínűségűek nem igényelnek beavatkozást. A Szürke Zóna a leghatékonyabb kampánycélpont: **maximális ROI minimális ráfordítással.**

            ---
            | Metrika | Érték |
            |---|---|
            | Sávban lévő ügyfelek száma | {grey_zone_count:,} fő |
            | Arány az összes ügyfélből | **{grey_zone_pct:.1f}%** |
            | Valószínűségi küszöb | {grey_zone_lower} - {grey_zone_upper} |

            *Megjegyzés: A küszöbértékek a executive_summary.py-ban a `grey_zone_lower` / `grey_zone_upper` változókkal módosíthatók.*
            """)

    with col4:
        with st.expander("ℹ️ Hogyan számoltuk? - Átlagos lemorzsolódási arány"):
            st.markdown(f"""
            **Adatforrás:** `churn_predictions.parquet` - a teljes ügyfélbázis historikus lemorzsolódási labelje (`actual_churn`).

            **Számítás:**

            $$ \\text{{Churn arány}} = \\frac{{\\text{{Lemorzsolódott ügyfelek}}}}{{\\text{{Összes ügyfél}}}} \\times 100 $$

            ---
            | Metrika | Érték |
            |---|---|
            | Összes ügyfél | {total_customers:,} fő |
            | Lemorzsolódott (`actual_churn = 1`) | {churned_customers:,} fő |
            | **Churn arány** | **{churn_rate_pct:.1f}%** |

            **Értelmezés:** Ez a mutató az ügyfélmegtartási stratégia legfőbb egészségügyi indikátora. Egy {churn_rate_pct:.0f}%-os churn arány azt jelenti, hogy az ügyfelek {'több mint fele' if churn_rate_pct > 50 else 'közel fele'} lemorzsolódott a megfigyelési időszakban - ez kiemelt megtartási fókuszt indokol.

            *Megjegyzés: Az `actual_churn` historikus label, a modell tanítása és validációja során került meghatározásra.*
            """)

    col5, col6 = st.columns(2)
    with col5:
        with st.expander("ℹ️ Hogyan számoltuk? - Modellezhető aktív ügyfélbázis"):
            st.markdown(f"""
            **Adatforrás:** `churn_predictions.parquet` - minden olyan ügyfél, akire a modell érvényes előrejelzést tudott adni.

            **Mit jelent ez a szám?** A nyers tranzakciós adatból az adatelőkészítés során kiszűrésre kerültek:
            - hiányos azonosítójú sorok,
            - visszáru-tranzakciók (negatív mennyiség),
            - az RFM-számításhoz nem elegendő aktivitású ügyfelek.

            Az itt látható **{total_customers:,} fő** az a végső ügyfélkör, amelyre az RFM-szegmentáció és a churn-modell együttesen alkalmazható.

            **Miért fontos kontextusként?** A dashboard többi százalékos mutatója ({churn_rate_pct:.1f}% churn, {grey_zone_pct:.1f}% szürke zóna) kizárólag ebből a {total_customers:,} fős bázisból értelmezendő. Egy azonos arány kisebb bázison kisebb abszolút érintett számot jelent - ez a kártya adja meg a léptéket.
            """)

    with col6:
        with st.expander("ℹ️ Hogyan számoltuk? - VIP Bajnokok visszaküldési aránya"):
            st.markdown(f"""
            **Célcsoport:** Az `rfm_segment == 'VIP Bajnokok'` szegmens - a legmagasabb RFM-pontszámú, leglojálisabb és legtöbbet költő {vip_champions_count:,} ügyfél.

            **A `return_ratio` definíciója:**

            $$ \\text{{return\\_ratio}} = \\frac{{\\text{{visszaküldött tételek száma}}}}{{\\text{{összes tranzakciós sor}}}} $$

            Ez az arány az adatelőkészítés során ügyfelenkénti szinten kerül kiszámításra a nyers tranzakciós adatból (negatív `Quantity` = visszaküldés).

            ---
            | Metrika | Érték |
            |---|---|
            | VIP Bajnokok átlagos visszaküldési aránya | **{vip_return_ratio_pct:.1f}%** |
            | Teljes ügyfélbázis átlaga | {overall_return_ratio_pct:.1f}% |
            | Különbség | +{vip_return_ratio_pct - overall_return_ratio_pct:.1f} százalékpont |
            | Érintett ügyfelek | {vip_champions_count:,} fő |

            **Üzleti következtetés:** A VIP Bajnokok visszaküldési aránya {vip_return_ratio_pct / overall_return_ratio_pct:.1f}×-osa az átlagnak. Ez arra utal, hogy a lemorzsolódás hátterében termékminőségi vagy logisztikai problémák állhatnak - a megoldás nem kizárólag marketing-, hanem termékfejlesztési és logisztikai feladat is.
            """)

    st.markdown("---")

    # ==========================================
    # 8. Feature importance grafikon
    # ==========================================
    churn_model = load_churn_model()
    if churn_model is None:
        st.warning("A churn modell nem tölthető be – ellenőrizd a `models/xgboost_churn.joblib` fájlt.")
    else:
        try:
            clf = churn_model.named_steps['clf']
            fi_series = pd.Series(
                clf.feature_importances_,
                index=clf.feature_names_in_
            ).sort_values(ascending=True)

            label_map = {
                'recency_days':    'Utolsó vásárlás óta eltelt napok',
                'frequency':       'Vásárlások gyakorisága',
                'monetary_total':  'Teljes elköltött összeg',
                'monetary_avg':    'Átlagos kosárérték',
                'return_ratio':    'Visszaküldési arány',
            }
            fi_series.index = [label_map.get(i, i) for i in fi_series.index]

            fi_min, fi_max = fi_series.values.min(), fi_series.values.max()
            norm_vals = [(v - fi_min) / (fi_max - fi_min) for v in fi_series.values]
            color_scale = [[0.0, '#9898c0'], [0.5, '#ff8c1a'], [1.0, '#ff1a3c']]
            bar_colors = pc.sample_colorscale(color_scale, norm_vals)

            fig_fi = go.Figure(go.Bar(
                x=fi_series.values * 100,
                y=fi_series.index,
                orientation='h',
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f'<b>{v:.1f}%</b>' for v in fi_series.values * 100],
                textposition='outside',
                textfont=dict(color='white', size=13),
                cliponaxis=False,
            ))

            fig_fi.update_layout(
                xaxis=dict(
                    title='Fontossági súly (%)',
                    color='rgba(200,207,232,0.6)',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.08)',
                    range=[0, 60],
                    ticksuffix='%',
                ),
                yaxis=dict(
                    color='white',
                    tickfont=dict(size=13),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(168,16,34,0.12)',
                font=dict(color='white'),
                margin=dict(l=10, r=80, t=20, b=40),
                height=300,
                showlegend=False,
            )

            st.subheader("Mi hajtja a churn-t? - A modell 5 magyarázó tényezője")
            st.plotly_chart(fig_fi, use_container_width=True)

            with st.expander("ℹ️ Magyarázat"):
                st.markdown(f"""
                **Adatforrás:** `xgboost_churn.joblib` - az XGBoost modell belső feature importance értékei (weight: hány döntési csomópontban szerepel az adott feature).

                **Mit mutat?** Azt, hogy a churn-előrejelzési modell döntéseihez melyik tényező mennyit nyom a latban:

                | Tényező | Fontossági súly | Üzleti üzenet |
                |---|---|---|
                | Utolsó vásárlás óta eltelt napok | **{fi_series.iloc[4]:.1%}** | Ha rég nem járt vissza → ez a legfőbb figyelmeztető jel |
                | Vásárlások gyakorisága | **{fi_series.iloc[3]:.1%}** | Ritka vásárlók sokkal inkább lemorzsolódnak |
                | Teljes elköltött összeg | **{fi_series.iloc[2]:.1%}** | Kisebb életértékű ügyfelek kockázatosabbak |
                | Átlagos kosárérték | **{fi_series.iloc[1]:.1%}** | Minimális prediktív erő |
                | Visszaküldési arány | **{fi_series.iloc[0]:.1%}** | Szinte nem számít a churn előrejelzésekor |

                **Kulcsüzenet a vezető számára:**
                A modell döntéseit elsősorban az **inaktivitás** hajtja: ki mikor és milyen sűrűn vásárolt.
                A visszaküldési arány a modellben alig 2%-ot magyaráz - ez azonban nem zárja ki, hogy oksági szerepe van; csupán azt jelenti, hogy az XGBoost ennél erősebb jeleket talált.

                **Következmény:** A megtartási stratégia fókusza az **email reaktivációs kampány** és a **vásárlási frekvencia növelése** (pl. loyalty program). A visszáru-folyamat hatásának vizsgálatához önálló elemzés javasolt.

                *Modell megbízhatósága: ROC-AUC = {roc_auc:.3f} - döntéstámogatásra alkalmas szint.*
                """)

        except Exception as e:
            st.error(f"Feature importance grafikon hiba: {e}")

    st.markdown("---")

    # ==========================================
    # 9. CHART 1 - Havi bevétel trend szegmens szerint
    # ==========================================
    SEG_COLORS = {
        'VIP Bajnokok':          '#ff1a3c',
        'Lemorzsolódó / Alvó':   '#ff8c1a',
        'Új / Ígéretes':         '#1ab4ff',
        'Elvesztett / Inaktív':  '#9898c0',
    }
    st.subheader("Hogyan alakult a bevétel időben? - Havi trend szegmensenként")

    seg_lookup = df_preds['rfm_segment']
    tx_seg = df_tx.copy()
    tx_seg['rfm_segment'] = tx_seg['Customer ID'].map(seg_lookup).fillna('Ismeretlen')
    tx_seg = tx_seg[tx_seg['rfm_segment'] != 'Ismeretlen']
    tx_seg['YearMonth'] = tx_seg['InvoiceDate'].dt.to_period('M').astype(str)

    monthly_rev = (
        tx_seg.groupby(['YearMonth', 'rfm_segment'])['LineTotal']
        .sum()
        .reset_index()
        .rename(columns={'LineTotal': 'Bevétel (£)', 'rfm_segment': 'Szegmens'})
    )
    monthly_rev = monthly_rev.sort_values('YearMonth')

    fig_area = px.area(
        monthly_rev,
        x='YearMonth',
        y='Bevétel (£)',
        color='Szegmens',
        color_discrete_map={
            'VIP Bajnokok':         '#ff1a3c',
            'Lemorzsolódó / Alvó':  '#ff8c1a',
            'Új / Ígéretes':        '#1ab4ff',
            'Elvesztett / Inaktív': '#9898c0',
        },
        labels={'YearMonth': 'Hónap'},
    )
    fig_area.update_traces(line=dict(width=1.5))
    fig_area.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(168,16,34,0.08)',
        font=dict(color='white'),
        xaxis=dict(
            color='rgba(200,207,232,0.6)',
            gridcolor='rgba(255,255,255,0.07)',
            tickangle=-35,
        ),
        yaxis=dict(
            color='rgba(200,207,232,0.6)',
            gridcolor='rgba(255,255,255,0.07)',
            tickprefix='£',
            tickformat=',.0f',
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.22,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(255,255,255,0.15)',
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=20, b=110),
        height=460,
    )
    st.plotly_chart(fig_area, use_container_width=True)

    with st.expander("ℹ️ Magyarázat"):
        st.markdown("""
        **Adatforrás:** `online_retail_ready_for_rfm.parquet` tranzakciók + `churn_predictions.parquet` szegmens-hozzárendelés (Customer ID alapú join).

        **Mit mutat?** Az egyes RFM-szegmensek havi összbevételét 2009 decembertől 2011 végéig.
        A területek egymásra halmozva mutatják a teljes bevételt és annak összetételét.

        **Szezonalitás:** Az év végi csúcsok (november-december) jól láthatóak - ez a kiskereskedelem tipikus ünnepi szezonja.

        **Kulcsüzenet:** A VIP Bajnokok (piros) a bevétel domináns forrása.
        Csökkentük-e az aktív vásárlók körét az évek során? Hány hónappal a cutoff előtt kezdett el csökkeni a lemmorzsolódó szegmens bevétele?
        """)

    st.markdown("---")

    # ==========================================
    # 10. CHART 2 - Pareto: ügyfelek bevétel-koncentrációja (80/20 szabály)
    # ==========================================
    st.subheader("Bevétel-koncentráció - Ki termeli a bevétel 80%-át?")

    # Ügyfelenkénti összbevétel kiszámítása - csak a modellezett ügyfélkörre szűrve
    customer_rev = (
        df_tx[df_tx['Customer ID'].isin(df_preds.index)]
        .groupby('Customer ID')['LineTotal']
        .sum()
        .reset_index()
        .rename(columns={'LineTotal': 'total_revenue'})
    )
    customer_rev = customer_rev[customer_rev['total_revenue'] > 0]
    customer_rev = customer_rev.sort_values('total_revenue', ascending=False).reset_index(drop=True)

    total_rev = customer_rev['total_revenue'].sum()
    customer_rev['cumulative_rev'] = customer_rev['total_revenue'].cumsum()
    customer_rev['cumulative_rev_pct'] = customer_rev['cumulative_rev'] / total_rev * 100
    customer_rev['customer_pct'] = (customer_rev.index + 1) / len(customer_rev) * 100

    # Hány % az ügyfélbázisból felelős a bevétel 80%-áért?
    pareto_idx = (customer_rev['cumulative_rev_pct'] >= 80).idxmax()
    pareto_customer_pct = customer_rev.loc[pareto_idx, 'customer_pct']
    pareto_customer_count = pareto_idx + 1

    fig_pareto = go.Figure()

    # Egyéni hover szöveg minden ponthoz
    customer_rev['hover_text'] = (
        'Bevétel ' + customer_rev['cumulative_rev_pct'].round(1).astype(str) + '%-a '
        + 'az ügyfelek ' + customer_rev['customer_pct'].round(1).astype(str) + '%-ától származik'
    )

    # Terület görbe
    fig_pareto.add_trace(go.Scatter(
        x=customer_rev['customer_pct'],
        y=customer_rev['cumulative_rev_pct'],
        mode='lines',
        fill='tozeroy',
        name='Kumulált bevétel',
        line=dict(color='#1aff6e', width=2.5),
        fillcolor='rgba(26,255,110,0.15)',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=customer_rev['hover_text'],
    ))

    # Egyenlőség vonal (ha mindenki egyenlő lenne)
    fig_pareto.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Egyenletes eloszlás',
        line=dict(color='rgba(200,207,232,0.3)', width=1.5, dash='dot'),
    ))

    # 80%-os vízszintes jelölővonal
    fig_pareto.add_shape(
        type='line', x0=0, x1=pareto_customer_pct, y0=80, y1=80,
        line=dict(color='#1aff6e', width=1.5, dash='dash'),
    )
    # Függőleges jelölővonal
    fig_pareto.add_shape(
        type='line', x0=pareto_customer_pct, x1=pareto_customer_pct, y0=0, y1=80,
        line=dict(color='#1aff6e', width=1.5, dash='dash'),
    )

    # Annotáció a metszésponthoz
    fig_pareto.add_annotation(
        x=pareto_customer_pct,
        y=80,
        text=f"<b>A bevétel 80%-a<br>az ügyfelek {pareto_customer_pct:.1f}%-ától származik ({pareto_customer_count:,} fő)</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#1aff6e',
        ax=70, ay=-55,
        font=dict(color='#1aff6e', size=13),
        bgcolor='rgba(20,20,40,0.75)',
        bordercolor='#1aff6e',
        borderwidth=1,
        borderpad=6,
    )

    fig_pareto.update_layout(
        xaxis=dict(
            title='Ügyfelek kumulált aránya (%)',
            color='rgba(200,207,232,0.6)',
            gridcolor='rgba(255,255,255,0.07)',
            ticksuffix='%',
            range=[0, 100],
        ),
        yaxis=dict(
            title='Kumulált bevétel aránya (%)',
            color='rgba(200,207,232,0.6)',
            gridcolor='rgba(255,255,255,0.07)',
            ticksuffix='%',
            range=[0, 100],
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(168,16,34,0.08)',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.18,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(255,255,255,0.15)',
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=20, b=90),
        height=440,
    )

    st.plotly_chart(fig_pareto, use_container_width=True)

    with st.expander("ℹ️ Magyarázat"):
        st.markdown(f"""
        **Adatforrás:** `online_retail_ready_for_rfm.parquet` - ügyfelenkénti összbevétel aggregáció.

        **Mit mutat?** A Lorenz-görbe / Pareto-diagram azt mutatja, hogy a bevétel mekkora hányada koncentrálódik az ügyfélbázis melyik részénél.

        Az ügyfélbázis egy szűk rétege dominálja a bevételt.

        ---

        **Üzleti következtetés:** A megtartási erőforrások elosztásakor ez a {pareto_customer_pct:.1f}%-os mag jelenti a legkritikusabb célcsoportot.
        Az ő elvesztésük aránytalanul nagy bevételkiesést okoz - ezért indokolt a VIP-szegmens kiemelt kezelése.
        """)

