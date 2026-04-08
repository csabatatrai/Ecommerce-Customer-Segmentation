"""
pages/98_🚀_Advanced_Streamlit.py
Haladó Streamlit funkciók puskázó oldala.
Minden, ami az alap Showcase-ből kimaradt: cache, session state,
fragmentek, chat UI, callback-ek, plotly haladó, st.query_params,
dinamikus oldalak, egyedi CSS és még sok más.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json

# ── Oldal konfiguráció ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Haladó Streamlit",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": None,
        "About": "# Haladó Streamlit Showcase\nEz a fájl a Streamlit ritkábban használt, de üzletileg értékes funkcióit mutatja be.",
    },
)

# ── Egyedi CSS injektálás ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Kártya stílus konténerekhez */
    div[data-testid="stVerticalBlock"] > div:has(> div.card-custom) {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(100,100,200,0.15);
    }
    /* Pulzáló badge */
    .pulse-badge {
        display: inline-block;
        background: #2ecc71;
        color: white;
        padding: 4px 14px;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(46,204,113,0.5); }
        50% { box-shadow: 0 0 0 8px rgba(46,204,113,0); }
    }
    /* Metric kártya hover effekt */
    div[data-testid="stMetric"] {
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Haladó Streamlit Funkciók")
st.markdown(
    'Ez az oldal azokat a Streamlit képességeket mutatja be, amik az alapverzióból kimaradtak. '
    '<span class="pulse-badge">ÉLŐ</span>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Globális szűrők (üzleti dashboard stílus)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🎛️ Globális szűrők")
    st.caption("Ezek a szűrők minden grafikont és táblázatot befolyásolnak – pont mint egy BI dashboardon.")

    date_range = st.date_input(
        "Dátumtartomány",
        value=(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)),
        help="Két dátumot adj meg a tartomány kijelöléséhez.",
    )

    region = st.selectbox("Régió", ["Összes", "Budapest", "Debrecen", "Szeged", "Pécs"])

    revenue_range = st.slider("Bevételi sáv (M Ft)", 0, 100, (10, 80))

    st.divider()
    st.subheader("📌 Session State állapot")
    st.json(
        {k: str(v)[:60] for k, v in st.session_state.items()},
        expanded=False,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_cache, tab_state, tab_charts, tab_chat, tab_forms, tab_advanced = st.tabs(
    [
        "⚡ Cache és Teljesítmény",
        "🧠 Session State",
        "📈 Haladó Grafikonok",
        "💬 Chat UI",
        "📋 Űrlapok és Callback",
        "🧩 Haladó Trükkök",
    ]
)


# =====================================================================
# TAB 1: Cache és Teljesítmény
# =====================================================================
with tab_cache:
    st.header("⚡ Gyorsítótárazás (Caching)")
    st.info(
        "A Streamlit minden interakciónál újrafuttatja az egész scriptet. "
        "A `@st.cache_data` és `@st.cache_resource` nélkül a drága műveletek minden kattintásnál lefutnának."
    )

    # ── @st.cache_data ──────────────────────────────────────────────
    st.subheader("1. `@st.cache_data` – Adatok cache-elése")
    st.caption("Soros adatokra (DataFrame, lista, dict). Minden híváskor **másolatot** ad vissza, így biztonságos.")

    with st.echo():
        @st.cache_data(ttl=datetime.timedelta(minutes=30), show_spinner="Adatok betöltése...")
        def load_large_dataset(rows: int = 50_000) -> pd.DataFrame:
            """Szimulált nagy adathalmaz – valójában API / DB lekérés lenne."""
            np.random.seed(42)
            return pd.DataFrame(
                {
                    "dátum": pd.date_range("2024-01-01", periods=rows, freq="h"),
                    "bevétel": np.random.lognormal(10, 1, rows).round(0),
                    "régió": np.random.choice(["Budapest", "Debrecen", "Szeged", "Pécs"], rows),
                    "kategória": np.random.choice(["SaaS", "Tanácsadás", "Licenc"], rows),
                }
            )

    c1, c2 = st.columns([2, 1])
    with c1:
        row_count = st.select_slider("Sorok száma", options=[1_000, 10_000, 50_000, 200_000], value=50_000)
        t0 = time.perf_counter()
        df_big = load_large_dataset(row_count)
        elapsed = time.perf_counter() - t0
        st.success(f"✅ {len(df_big):,} sor betöltve **{elapsed*1000:.1f} ms** alatt (cache-ből: {'igen' if elapsed < 0.01 else 'nem'})")
    with c2:
        st.metric("Cache méret (hozzávetőleges)", f"{df_big.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ── @st.cache_resource ──────────────────────────────────────────
    st.divider()
    st.subheader("2. `@st.cache_resource` – Erőforrások cache-elése")
    st.caption("ML modellekre, DB kapcsolatokra. **Nem** másolja, hanem megosztja a példányt az összes felhasználóval.")

    with st.echo():
        @st.cache_resource
        def get_expensive_model():
            """Szimulált modell betöltés (pl. scikit-learn, ONNX, stb.)"""
            time.sleep(0.1)  # valójában: joblib.load("model.pkl")
            return {"name": "GradientBoosting", "accuracy": 0.934, "loaded_at": datetime.datetime.now().isoformat()}

    model_info = get_expensive_model()
    st.json(model_info)

    st.divider()
    st.subheader("3. `st.cache_data.clear()` – Cache ürítés")
    if st.button("🗑️ Adatcache ürítése"):
        load_large_dataset.clear()
        st.toast("Adatcache törölve!", icon="🗑️")
        st.rerun()


# =====================================================================
# TAB 2: Session State haladó
# =====================================================================
with tab_state:
    st.header("🧠 Session State – Állapotkezelés")
    st.markdown(
        "A `st.session_state` egy dictionary, ami **megmarad** a rerundok között. "
        "A widgetek `key=` paraméterén keresztül automatikusan is használhatod."
    )

    # ── Számláló példa callback-kel ────────────────────────────────
    st.subheader("Callback alapú számláló")

    if "advanced_counter" not in st.session_state:
        st.session_state.advanced_counter = 0
        st.session_state.history = []

    def _increment(amount: int):
        st.session_state.advanced_counter += amount
        st.session_state.history.append(
            {"idő": datetime.datetime.now().strftime("%H:%M:%S"), "változás": amount}
        )

    def _reset():
        st.session_state.advanced_counter = 0
        st.session_state.history = []

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.button("➕ +1", on_click=_increment, args=(1,), use_container_width=True)
    col_b.button("➕ +10", on_click=_increment, args=(10,), use_container_width=True)
    col_c.button("➖ -5", on_click=_increment, args=(-5,), use_container_width=True)
    col_d.button("🔄 Reset", on_click=_reset, use_container_width=True, type="secondary")

    st.metric("Aktuális érték", st.session_state.advanced_counter)

    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history).tail(10),
            use_container_width=True,
            hide_index=True,
        )

    # ── Widget ↔ Session State szinkron ────────────────────────────
    st.divider()
    st.subheader("Widget ↔ Session State szinkron")
    st.caption("Ha `key=`-t adsz meg, a widget értéke automatikusan `st.session_state[key]`-ben landol.")

    with st.echo():
        st.text_input("Név", key="user_name", placeholder="Gipsz Jakab")
        st.write(f"Üdvözöllek, **{st.session_state.get('user_name', '???')}**!")

    # ── Multi-page state megosztás ─────────────────────────────────
    st.divider()
    st.subheader("State megosztás oldalak között")
    st.code(
        """
# Bármely pages/*.py fájlban elérhető:
if "global_filter" not in st.session_state:
    st.session_state.global_filter = "Budapest"

# Másik oldalon olvashatod:
selected = st.session_state.global_filter
        """,
        language="python",
    )


# =====================================================================
# TAB 3: Haladó Grafikonok
# =====================================================================
with tab_charts:
    st.header("📈 Haladó Plotly & Streamlit Grafikonok")

    # Demo adatok
    np.random.seed(42)
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    df_sales = pd.DataFrame(
        {
            "hónap": months,
            "SaaS": np.cumsum(np.random.normal(50, 15, 12)).round(0),
            "Tanácsadás": np.cumsum(np.random.normal(30, 10, 12)).round(0),
            "Licenc": np.cumsum(np.random.normal(20, 8, 12)).round(0),
        }
    )

    # ── Kombó grafikon (vonal + oszlop) ────────────────────────────
    st.subheader("1. Kombó grafikon – Kettős tengely")
    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_combo.add_trace(
        go.Bar(x=df_sales["hónap"], y=df_sales["SaaS"], name="SaaS bevétel", marker_color="#667eea", opacity=0.7),
        secondary_y=False,
    )
    fig_combo.add_trace(
        go.Scatter(
            x=df_sales["hónap"],
            y=(df_sales["SaaS"].pct_change() * 100).round(1),
            name="Havi növekedés %",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2.5),
        ),
        secondary_y=True,
    )
    fig_combo.update_layout(
        template="plotly_white", height=380, margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    fig_combo.update_yaxes(title_text="Bevétel (kumulált)", secondary_y=False)
    fig_combo.update_yaxes(title_text="Növekedés %", secondary_y=True)
    st.plotly_chart(fig_combo, use_container_width=True)

    # ── Heatmap ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2. Heatmap – Korreláció / Kohorsz")
        heatmap_data = np.random.rand(7, 12) * 100
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Hónap", y="Kohorsz", color="Retenciós %"),
            x=[f"{i+1}. hónap" for i in range(12)],
            y=[f"2024 Q{i//2+1} – Kohorsz {i+1}" for i in range(7)],
            color_continuous_scale="YlGnBu",
            aspect="auto",
        )
        fig_heat.update_layout(height=350, margin=dict(t=30, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with c2:
        st.subheader("3. Sunburst – Hierarchikus adat")
        df_sun = pd.DataFrame(
            {
                "régió": ["Budapest"] * 3 + ["Vidék"] * 3,
                "kategória": ["SaaS", "Tanácsadás", "Licenc"] * 2,
                "bevétel": [120, 80, 40, 60, 90, 30],
            }
        )
        fig_sun = px.sunburst(df_sun, path=["régió", "kategória"], values="bevétel", color="bevétel", color_continuous_scale="Viridis")
        fig_sun.update_layout(height=350, margin=dict(t=30, b=10))
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Funnel (értékesítési tölcsér) ──────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("4. Értékesítési tölcsér")
        fig_funnel = go.Figure(
            go.Funnel(
                y=["Weboldal látogatók", "Regisztráltak", "Demo kérés", "Ajánlat kiküldve", "Lezárt üzlet"],
                x=[12000, 5400, 2100, 860, 340],
                textinfo="value+percent initial",
                marker=dict(color=["#667eea", "#764ba2", "#e74c3c", "#f39c12", "#2ecc71"]),
            )
        )
        fig_funnel.update_layout(height=380, margin=dict(t=30, b=10))
        st.plotly_chart(fig_funnel, use_container_width=True)

    with c4:
        st.subheader("5. Gauge (mérőóra) – KPI cél")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=73,
                delta={"reference": 60, "increasing": {"color": "#2ecc71"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 40], "color": "#fadbd8"},
                        {"range": [40, 70], "color": "#fdebd0"},
                        {"range": [70, 100], "color": "#d5f5e3"},
                    ],
                    "threshold": {"line": {"color": "#e74c3c", "width": 3}, "thickness": 0.8, "value": 90},
                },
                title={"text": "NPS Score"},
            )
        )
        fig_gauge.update_layout(height=380, margin=dict(t=60, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Plotly animáció ────────────────────────────────────────────
    st.divider()
    st.subheader("6. Animált grafikon – Plotly animation_frame")
    np.random.seed(42)
    anim_data = []
    for year in range(2020, 2026):
        for cat in ["SaaS", "Tanácsadás", "Licenc"]:
            anim_data.append({"Év": str(year), "Kategória": cat, "Bevétel": np.random.randint(20, 150), "Ügyfelek": np.random.randint(5, 60)})
    df_anim = pd.DataFrame(anim_data)
    fig_anim = px.bar(
        df_anim, x="Kategória", y="Bevétel", color="Kategória",
        animation_frame="Év", range_y=[0, 160],
        title="Éves bevétel kategóriánként (nyomd meg a Play-t!)",
    )
    fig_anim.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_anim, use_container_width=True)


# =====================================================================
# TAB 4: Chat UI
# =====================================================================
with tab_chat:
    st.header("💬 Chat felület – `st.chat_message` & `st.chat_input`")
    st.caption(
        "LLM-alapú chatbotok, ügyfélszolgálati asszisztensek vagy belső tudástárak építéséhez. "
        "A session_state-ben tároljuk a chat előzményeket."
    )

    # ── Chat történet inicializálás ────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Üdv! Kérdezz bármit a Streamlit-ről, és megpróbálok segíteni. 🤖"}
        ]

    # ── Előzmények megjelenítése ───────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Felhasználói bemenet ───────────────────────────────────────
    if prompt := st.chat_input("Írd be a kérdésed..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Szimulált „streaming" válasz
        with st.chat_message("assistant"):
            placeholder = st.empty()
            fake_response = (
                f"Köszönöm a kérdésed! Sajnos ez egy demo, valódi LLM nincs a háttérben. "
                f"Amit írtál: *«{prompt}»*. Egy éles alkalmazásban itt hívnád meg az OpenAI / Anthropic API-t, "
                f"és a `st.write_stream()` segítségével streamelnéd a választ token-ről tokenre."
            )
            streamed = ""
            for char in fake_response:
                streamed += char
                placeholder.markdown(streamed + "▌")
                time.sleep(0.008)
            placeholder.markdown(streamed)

        st.session_state.chat_messages.append({"role": "assistant", "content": fake_response})

    # ── Chat API integráció minta ──────────────────────────────────
    with st.expander("📌 Valódi LLM integráció minta kód"):
        st.code(
            """
import anthropic

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Streaming válasz:
with st.chat_message("assistant"):
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=st.session_state.chat_messages,
    ) as stream:
        response = st.write_stream(stream.text_stream)

st.session_state.chat_messages.append(
    {"role": "assistant", "content": response}
)
            """,
            language="python",
        )


# =====================================================================
# TAB 5: Űrlapok és Callback
# =====================================================================
with tab_forms:
    st.header("📋 Űrlapok és Callback-ek")

    # ── st.form ────────────────────────────────────────────────────
    st.subheader("1. `st.form` – Több mező → egyetlen submit")
    st.caption(
        "Az űrlapon belüli widgetek NEM indítják újra a scriptet, "
        "csak a Submit gomb kattintásakor fut le minden egyszerre."
    )

    with st.form("demo_form", clear_on_submit=False, border=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            name = st.text_input("Cégnév", placeholder="Példa Kft.")
            email = st.text_input("Email", placeholder="info@pelda.hu")
            plan = st.selectbox("Csomag", ["Starter – 49€/hó", "Pro – 149€/hó", "Enterprise – egyedi"])
        with fc2:
            seats = st.number_input("Felhasználók száma", 1, 1000, 5)
            start = st.date_input("Indulás dátuma")
            notes = st.text_area("Megjegyzés", height=115)

        submitted = st.form_submit_button("🚀 Ajánlat küldése", type="primary", use_container_width=True)

    if submitted:
        st.success(f"Ajánlat elküldve: **{name}** ({email}) – {plan}, {seats} fő, indulás: {start}")
        with st.expander("Elküldött JSON payload"):
            st.json({"company": name, "email": email, "plan": plan, "seats": seats, "start": str(start), "notes": notes})

    # ── Callback haladó ────────────────────────────────────────────
    st.divider()
    st.subheader("2. `on_change` Callback – Azonnali reakció")
    st.caption("A callback a widget értékváltozásakor **a rerun elején** fut le – gyorsabb, mint utólag ellenőrizni.")

    if "color_log" not in st.session_state:
        st.session_state.color_log = []

    def _on_color_change():
        c = st.session_state.get("demo_color", "#000000")
        st.session_state.color_log.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} → {c}")

    st.color_picker("Válassz színt", "#3498db", key="demo_color", on_change=_on_color_change)

    if st.session_state.color_log:
        st.caption("Színváltozás napló (on_change callback):")
        for entry in st.session_state.color_log[-5:]:
            st.text(entry)

    # ── Dinamikus sorok hozzáadása ─────────────────────────────────
    st.divider()
    st.subheader("3. Dinamikus sorok – Termékek hozzáadása")

    if "products" not in st.session_state:
        st.session_state.products = [{"name": "", "price": 0, "qty": 1}]

    def _add_product():
        st.session_state.products.append({"name": "", "price": 0, "qty": 1})

    def _remove_product(idx):
        st.session_state.products.pop(idx)

    for i, prod in enumerate(st.session_state.products):
        pc1, pc2, pc3, pc4 = st.columns([3, 2, 1, 0.5])
        st.session_state.products[i]["name"] = pc1.text_input("Termék", value=prod["name"], key=f"pn_{i}", label_visibility="collapsed" if i > 0 else "visible")
        st.session_state.products[i]["price"] = pc2.number_input("Ár (Ft)", value=prod["price"], key=f"pp_{i}", label_visibility="collapsed" if i > 0 else "visible")
        st.session_state.products[i]["qty"] = pc3.number_input("Db", value=prod["qty"], min_value=1, key=f"pq_{i}", label_visibility="collapsed" if i > 0 else "visible")
        if i > 0:
            pc4.button("❌", key=f"pdel_{i}", on_click=_remove_product, args=(i,))

    bc1, bc2, _ = st.columns([1, 1, 3])
    bc1.button("➕ Új sor", on_click=_add_product, use_container_width=True)
    total = sum(p["price"] * p["qty"] for p in st.session_state.products)
    bc2.metric("Összesen", f"{total:,.0f} Ft")


# =====================================================================
# TAB 6: Haladó Trükkök
# =====================================================================
with tab_advanced:
    st.header("🧩 Haladó Streamlit Trükkök")

    # ── st.query_params ────────────────────────────────────────────
    st.subheader("1. `st.query_params` – URL paraméterek")
    st.caption("Megosztható linkek szűrőkkel! `?city=Budapest&metric=revenue` → a szűrők automatikusan beállnak.")

    with st.echo():
        qp = st.query_params
        city_from_url = qp.get("city", "Budapest")
        st.write(f"URL-ből kapott város: **{city_from_url}**")
        if st.button("Állítsd Debrecenre az URL-t"):
            st.query_params["city"] = "Debrecen"
            st.rerun()

    # ── st.fragment (részleges rerun) ──────────────────────────────
    st.divider()
    st.subheader("2. `@st.fragment` – Részleges újrafuttatás")
    st.caption("Csak az adott blokk fut újra, nem az egész oldal. Ideális élő dashboard-okhoz.")

    st.code(
        """
@st.fragment(run_every=datetime.timedelta(seconds=5))
def live_kpi_panel():
    import random
    val = random.randint(80, 120)
    st.metric("Élő rendelés/perc", val, delta=val - 100)

live_kpi_panel()
        """,
        language="python",
    )

    # Egyszerűsített, nem-dekorátor verziójú demo (mert a rerun_every
    # valódi működéséhez a fragment dekorátor kellene)
    st.info("Éles alkalmazásban a `@st.fragment(run_every=5)` a megadott blokot 5 másodpercenként frissíti a teljes oldal nélkül.")

    # ── Többlépcsős wizard ─────────────────────────────────────────
    st.divider()
    st.subheader("3. Többlépcsős Wizard (lépésről lépésre)")

    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
        st.session_state.wizard_data = {}

    step = st.session_state.wizard_step

    # Progressz bar
    st.progress(step / 3, text=f"Lépés {step} / 3")

    if step == 1:
        with st.container(border=True):
            st.markdown("#### 📝 1. lépés – Alapadatok")
            st.session_state.wizard_data["company"] = st.text_input("Cég neve", value=st.session_state.wizard_data.get("company", ""))
            st.session_state.wizard_data["industry"] = st.selectbox("Iparág", ["Tech", "Pénzügy", "Egészségügy", "Gyártás"], index=0)
            if st.button("Tovább →", type="primary"):
                st.session_state.wizard_step = 2
                st.rerun()

    elif step == 2:
        with st.container(border=True):
            st.markdown("#### 💰 2. lépés – Üzleti adatok")
            st.session_state.wizard_data["revenue"] = st.number_input("Éves árbevétel (M Ft)", 0, 10000, 100)
            st.session_state.wizard_data["employees"] = st.slider("Létszám", 1, 5000, 50)
            bc1, bc2 = st.columns(2)
            if bc1.button("← Vissza"):
                st.session_state.wizard_step = 1
                st.rerun()
            if bc2.button("Tovább →", type="primary"):
                st.session_state.wizard_step = 3
                st.rerun()

    elif step == 3:
        with st.container(border=True):
            st.markdown("#### ✅ 3. lépés – Összegzés")
            st.json(st.session_state.wizard_data)
            bc1, bc2 = st.columns(2)
            if bc1.button("← Vissza"):
                st.session_state.wizard_step = 2
                st.rerun()
            if bc2.button("🚀 Beküldés", type="primary"):
                st.balloons()
                st.success("Sikeresen beküldve!")
                st.session_state.wizard_step = 1
                st.session_state.wizard_data = {}

    # ── st.status – Többlépcsős folyamat jelzés ───────────────────
    st.divider()
    st.subheader("4. `st.status` – Összetett folyamatjelző")

    if st.button("🔬 Pipeline indítása"):
        with st.status("Pipeline futtatása...", expanded=True) as status:
            st.write("📥 Adatok letöltése...")
            time.sleep(1)
            st.write("🧹 Tisztítás és transzformáció...")
            time.sleep(1)
            st.write("🤖 Modell betanítása...")
            time.sleep(1.5)
            st.write("📊 Riport generálás...")
            time.sleep(0.5)
            status.update(label="✅ Pipeline kész!", state="complete", expanded=False)

    # ── Egyedi HTML + st.components.v1.html ────────────────────────
    st.divider()
    st.subheader("5. Egyedi HTML injektálás")
    st.caption("`st.html()` vagy `st.components.v1.html()` – bármilyen HTML/CSS/JS-t beágyazhatsz.")

    st.html(
        """
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            font-family: system-ui;
        ">
            <h2 style="margin:0 0 0.5rem 0;">🎯 Éves célkitűzés</h2>
            <p style="font-size:3rem; font-weight:800; margin:0;">142%</p>
            <p style="opacity:0.8; margin:0.5rem 0 0 0;">Az éves bevételi cél teljesítése</p>
        </div>
        """
    )

    # ── Secrets és konfigurációs minta ─────────────────────────────
    st.divider()
    st.subheader("6. `st.secrets` – Titkok kezelése")
    st.code(
        """
# .streamlit/secrets.toml:
# ANTHROPIC_API_KEY = "sk-ant-..."
# [database]
# host = "db.example.com"
# port = 5432

api_key = st.secrets["ANTHROPIC_API_KEY"]
db_host = st.secrets.database.host
        """,
        language="toml",
    )

    # ── st.dialog (modal) ──────────────────────────────────────────
    st.divider()
    st.subheader("7. `@st.dialog` – Modális ablak")
    st.code(
        """
@st.dialog("Visszajelzés küldése")
def feedback_dialog():
    rating = st.slider("Értékelés", 1, 5, 3)
    comment = st.text_area("Megjegyzés")
    if st.button("Küldés"):
        save_feedback(rating, comment)
        st.rerun()

if st.button("💬 Visszajelzés"):
    feedback_dialog()
        """,
        language="python",
    )
    st.info("A `@st.dialog` egy felugró modális ablakot hoz létre – tökéletes megerősítő kérdésekhez, űrlapokhoz.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    <div style="text-align:center; opacity:0.5; padding: 1rem 0;">
        Készítette: Streamlit Haladó Showcase &nbsp;•&nbsp; Streamlit v1.40+ ajánlott
    </div>
    """,
    unsafe_allow_html=True,
)