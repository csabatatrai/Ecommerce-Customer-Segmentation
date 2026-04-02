"""
pages/03_action_planner.py
───────────────────────────
Akciótervező – RFM × Churn mátrix, szegmens-szintű prioritizálás,
exportálható ügyféllisták, üzleti ROI becslés
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

# ── Konfiguráció ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Akciótervező",
    page_icon="📋",
    layout="wide",
)

BG_DARK  = "#0F1117"
BG_CARD  = "#1A1D27"
BG_CARD2 = "#13151F"
ACCENT   = "#E8472A"
AMBER    = "#F0A500"
TEAL     = "#00C9A7"
BLUE     = "#2AB5E8"
TEXT_PRI = "#F0F0F0"
TEXT_SEC = "#9FA3B1"

SEGMENT_COLORS = {
    "VIP Bajnokok":          "#E8472A",
    "Új / Ígéretes":         "#2AB5E8",
    "Lemorzsolódó / Alvó":   "#F0A500",
    "Elvesztett / Inaktív":  "#8A8A8A",
}

ACTION_META = {
    "🚨 VIP Veszélyben – Azonnali Retenció": {
        "color": "#E8472A",
        "icon": "🚨",
        "priority": 1,
        "playbook": [
            "Személyes account manager megkeresés (24 órán belül)",
            "Exkluzív visszatérési kupon (15–20%, 30 napos lejárat)",
            "3 részes win-back email sorozat (2 hetes intervallum)",
            "NPS felmérés küldése és proaktív panaszkezelés",
            "Ingyenes szállítás és prémium ügyfélszolgálat biztosítása",
        ],
        "expected_retention": 0.35,
        "cost_per_customer": 45,
    },
    "💎 VIP Stabil – Lojalitás Program": {
        "color": "#00C9A7",
        "icon": "💎",
        "priority": 2,
        "playbook": [
            "Lojalitáspontok és exkluzív tagság felajánlása",
            "Early access az új termékekhez és szezonális kollekcióhoz",
            "Születésnapi / évfordulós személyes meglepetés",
            "VIP ügyfél-rendezvény meghívó (B2B esetén partnerprogram)",
            "Referral program ösztönzése (ajánlói kedvezmény)",
        ],
        "expected_retention": 0.85,
        "cost_per_customer": 20,
    },
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back": {
        "color": "#F0A500",
        "icon": "⚠️",
        "priority": 3,
        "playbook": [
            "Automatizált win-back email kampány (3 üzenet)",
            "10%-os visszatérési kedvezmény + ingyenes szállítás",
            "Termékkiegészítő (cross-sell) ajánlók a korábbi rendelések alapján",
            "Reaktiváló SMS/push értesítő (GDPR-kompliáns)",
            "Kosárelhagyó emlékeztető beállítása",
        ],
        "expected_retention": 0.15,
        "cost_per_customer": 8,
    },
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": {
        "color": "#8A8A8A",
        "icon": "✅",
        "priority": 4,
        "playbook": [
            "Heti/havi hírlevél és szezonális promóciók",
            "Termékajánló algoritmus személyre szabása",
            "Egyszerű hűségpont rendszer bevezetése",
            "Ügyfél-elégedettség kérdőív (rövid, 3 kérdéses)",
        ],
        "expected_retention": 0.60,
        "cost_per_customer": 3,
    },
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG_DARK}; color: {TEXT_PRI};
}}
.stApp {{ background-color: {BG_DARK}; }}

.page-header {{
    display: flex; align-items: flex-end; gap: 20px;
    padding: 32px 0 24px; border-bottom: 1px solid #2A2D3A; margin-bottom: 32px;
}}
.page-header h1 {{
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: {TEXT_PRI}; margin: 0; line-height: 1;
}}
.page-header .sub {{ font-size: 0.9rem; color: {TEXT_SEC}; padding-bottom: 4px; }}

.kpi-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A; border-radius: 12px;
    padding: 20px 22px; position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, {ACCENT});
}}
.kpi-label {{ font-size: 0.75rem; text-transform: uppercase;
              letter-spacing: .08em; color: {TEXT_SEC}; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
              color: {TEXT_PRI}; line-height: 1; }}
.kpi-delta {{ font-size: 0.78rem; color: {TEXT_SEC}; margin-top: 4px; }}

.section-title {{
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: {TEXT_PRI}; margin: 0 0 14px; letter-spacing: .03em;
}}
.chart-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A;
    border-radius: 14px; padding: 22px 20px; margin-bottom: 20px;
}}

/* Akció kártyák */
.action-card {{
    background: {BG_CARD}; border: 1px solid #2A2D3A; border-radius: 14px;
    padding: 20px; margin-bottom: 16px; position: relative;
}}
.action-card-header {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
}}
.action-title {{
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: {TEXT_PRI};
}}
.action-stat {{
    display: flex; gap: 20px; flex-wrap: wrap;
    padding: 10px 0; border-top: 1px solid #2A2D3A;
    border-bottom: 1px solid #2A2D3A; margin: 10px 0;
}}
.action-stat-item {{ text-align: center; }}
.action-stat-val {{
    font-family: 'Syne', sans-serif; font-size: 1.2rem;
    font-weight: 700; color: {TEXT_PRI};
}}
.action-stat-lbl {{ font-size: 0.7rem; color: {TEXT_SEC}; text-transform: uppercase; }}
.playbook-item {{
    display: flex; gap: 10px; align-items: flex-start;
    padding: 5px 0; font-size: 0.86rem; color: {TEXT_SEC}; line-height: 1.5;
}}
.playbook-num {{
    min-width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; background: var(--seg-color);
    color: white; flex-shrink: 0; margin-top: 1px;
}}
</style>
""", unsafe_allow_html=True)

# ── Adatbetöltés ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    p = Path("data/processed/churn_predictions.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


df_raw = load_data()

if df_raw is None:
    rng  = np.random.default_rng(42)
    n    = 5243
    segs = rng.choice(
        ["VIP Bajnokok", "Új / Ígéretes", "Lemorzsolódó / Alvó", "Elvesztett / Inaktív"],
        size=n, p=[0.09, 0.25, 0.28, 0.38]
    )
    monetary = rng.exponential(2000, n).clip(10, 450000)
    recency  = rng.integers(0, 646, n)
    freq     = rng.integers(1, 100, n)
    churn_proba = np.clip(0.3 + 0.0007*recency - 0.008*freq + rng.normal(0,.15,n), 0.01, 0.99)
    median_m = np.median(monetary)

    def act(m, cp):
        if m > median_m and cp >= 0.5:   return "🚨 VIP Veszélyben – Azonnali Retenció"
        elif m > median_m:               return "💎 VIP Stabil – Lojalitás Program"
        elif cp >= 0.5:                  return "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back"
        else:                            return "✅ Alacsony Értékű, Stabil – Standard Kommunikáció"

    df_raw = pd.DataFrame({
        "Customer ID":    [f"C{i:05d}" for i in range(n)],
        "rfm_segment":    segs,
        "monetary_total": monetary,
        "frequency":      freq,
        "recency_days":   recency,
        "churn_proba":    churn_proba,
        "churn_pred":     (churn_proba >= 0.5).astype(int),
        "actual_churn":   rng.binomial(1, churn_proba),
        "return_ratio":   rng.uniform(0, 0.5, n),
        "action":         [act(m, cp) for m, cp in zip(monetary, churn_proba)],
    })
    st.info("ℹ️  Demo mód: `data/processed/churn_predictions.parquet` nem található.", icon="🔔")

df = df_raw.copy()

# Oszlopnév fallback
if "rfm_segment" not in df.columns:
    for c in df.columns:
        if "segment" in c.lower() or "szegmens" in c.lower():
            df = df.rename(columns={c: "rfm_segment"})
            break
    if "rfm_segment" not in df.columns:
        df["rfm_segment"] = "Ismeretlen"

if "action" not in df.columns:
    df["action"] = "Ismeretlen"

# ── FEJLÉC ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div><h1>📋 Akciótervező</h1></div>
  <div class="sub">Szegmens × Churn prioritizálás · Üzleti playbook · Exportálható listák</div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    churn_thresh = st.slider("Churn küszöb", 0.1, 0.9, 0.5, 0.05)
    retention_invest = st.number_input("Retenciós büdzsé (£)", value=50000, step=5000)
    st.markdown("---")
    st.markdown("### 🎯 Fókusz-szegmens")
    focus_action = st.selectbox(
        "Melyik szegmenst exportálod?",
        options=list(ACTION_META.keys()),
        index=0,
    )

# ── KPI felső sáv ─────────────────────────────────────────────────────────────
df["churn_pred_t"] = (df["churn_proba"] >= churn_thresh).astype(int)
n_churn = df["churn_pred_t"].sum()
lost_rev = df[df["churn_pred_t"] == 1]["monetary_total"].sum()
vip_at_risk = df[df["action"] == "🚨 VIP Veszélyben – Azonnali Retenció"]

# ROI becslés: ha megmentünk 35%-ot a VIP veszélyben szegmensből
meta_vip = ACTION_META["🚨 VIP Veszélyben – Azonnali Retenció"]
expected_saved = vip_at_risk["monetary_total"].sum() * meta_vip["expected_retention"]
cost_total     = len(vip_at_risk) * meta_vip["cost_per_customer"]
roi_pct        = (expected_saved - cost_total) / cost_total * 100 if cost_total > 0 else 0

kpis = [
    ("Előrejelzett churnerek",    f"{n_churn:,}",                    f"küszöb: {churn_thresh}",       ACCENT),
    ("Veszélyeztetett bevétel",   f"£{lost_rev/1e3:.0f}K",            "ha senkit sem mentenek meg",    AMBER),
    ("VIP megmenthető bevétel",   f"£{expected_saved/1e3:.0f}K",      f"{meta_vip['expected_retention']:.0%} retenciós arány",  TEAL),
    ("Várható kampány ROI",        f"{roi_pct:.0f}%",                 f"£{cost_total:,.0f} befektetésnél", BLUE),
]

cols = st.columns(len(kpis))
for col, (label, val, delta, color) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# ── PRIORITÁSI MÁTRIX (heatmap) ───────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">RFM-szegmens × Akció-kategória mátrix (ügyfelek száma)</div>', unsafe_allow_html=True)

cross = pd.crosstab(df["rfm_segment"], df["action"])
# Rövid oszlopnevek
short_map = {
    "🚨 VIP Veszélyben – Azonnali Retenció":           "🚨 VIP Veszélyben",
    "💎 VIP Stabil – Lojalitás Program":               "💎 VIP Stabil",
    "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back":     "⚠️ Win-Back",
    "✅ Alacsony Értékű, Stabil – Standard Kommunikáció": "✅ Stabil",
}
cross.columns = [short_map.get(c, c) for c in cross.columns]

z     = cross.values
x_lab = list(cross.columns)
y_lab = list(cross.index)

# Szöveg annotáció
text  = [[str(v) if v > 0 else "" for v in row] for row in z]

fig_heat = go.Figure(go.Heatmap(
    z=z, x=x_lab, y=y_lab, text=text,
    texttemplate="%{text}",
    textfont=dict(size=14, color="white", family="Syne"),
    colorscale=[
        [0.0, "#1A1D27"],
        [0.3, "#4A1A10"],
        [0.7, "#A03020"],
        [1.0, "#E8472A"],
    ],
    showscale=True,
    colorbar=dict(
        tickfont=dict(color=TEXT_SEC, size=10),
        bgcolor="rgba(0,0,0,0)",
        thickness=12, len=0.8,
    ),
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>%{z} ügyfél<extra></extra>",
))
fig_heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="DM Sans"),
    xaxis=dict(side="top", showgrid=False, tickfont=dict(size=12)),
    yaxis=dict(showgrid=False, tickfont=dict(size=12)),
    margin=dict(l=0, r=0, t=60, b=0), height=280,
)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── ROI WATERFALL ─────────────────────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Becsült kampány ROI – szegmensenként</div>', unsafe_allow_html=True)

roi_rows = []
for action_key, meta in ACTION_META.items():
    seg_df = df[df["action"] == action_key]
    seg_rev = seg_df["monetary_total"].sum()
    n_seg   = len(seg_df)
    saved   = seg_rev * meta["expected_retention"]
    cost    = n_seg   * meta["cost_per_customer"]
    net     = saved - cost
    roi_rows.append({
        "Szegmens":       action_key,
        "Ügyfelek":       n_seg,
        "Bevétel (£)":    seg_rev,
        "Megmentett (£)": saved,
        "Kampány ktg (£)": cost,
        "Nettó haszon (£)": net,
        "ROI (%)":        (net / cost * 100) if cost > 0 else 0,
        "color":          meta["color"],
    })

roi_df = pd.DataFrame(roi_rows)
short_labels2 = {v: k for k, v in {
    "🚨 VIP Veszélyben": "🚨 VIP Veszélyben – Azonnali Retenció",
    "💎 VIP Stabil": "💎 VIP Stabil – Lojalitás Program",
    "⚠️ Win-Back": "⚠️  Alacsony Értékű, Lemorzsolódó – Win-Back",
    "✅ Stabil": "✅ Alacsony Értékű, Stabil – Standard Kommunikáció",
}.items()}
roi_df["Szegmens_rövid"] = roi_df["Szegmens"].map(
    {v: k for k, v in short_map.items()}
).fillna(roi_df["Szegmens"])

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=roi_df["Szegmens_rövid"],
    y=roi_df["Megmentett (£)"],
    name="Megmentett bevétel",
    marker_color=[c for c in roi_df["color"]],
    opacity=0.85,
    hovertemplate="<b>%{x}</b><br>Megmentett: £%{y:,.0f}<extra></extra>",
))
fig_bar.add_trace(go.Bar(
    x=roi_df["Szegmens_rövid"],
    y=-roi_df["Kampány ktg (£)"],
    name="Kampány költség",
    marker_color="#2A2D3A",
    hovertemplate="<b>%{x}</b><br>Ktg: £%{customdata:,.0f}<extra></extra>",
    customdata=roi_df["Kampány ktg (£)"],
))

fig_bar.update_layout(
    barmode="relative",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="DM Sans"),
    xaxis=dict(showgrid=False, tickfont=dict(size=12)),
    yaxis=dict(showgrid=True, gridcolor="#2A2D3A", zeroline=True,
               zerolinecolor="#3A3D4A", title="£"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=11)),
    margin=dict(l=0, r=0, t=10, b=0), height=300,
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── PLAYBOOK KÁRTYÁK ──────────────────────────────────────────────────────────
st.markdown(f'<div class="section-title" style="margin-bottom:16px">📘 Szegmens Playbook</div>', unsafe_allow_html=True)

tab_names = [f"{m['icon']} {k.split('–')[0].strip()}" for k, m in ACTION_META.items()]
tabs = st.tabs(tab_names)

for tab, (action_key, meta) in zip(tabs, ACTION_META.items()):
    with tab:
        seg_df    = df[df["action"] == action_key]
        seg_rev   = seg_df["monetary_total"].sum()
        n_seg     = len(seg_df)
        saved     = seg_rev * meta["expected_retention"]
        cost      = n_seg * meta["cost_per_customer"]
        color     = meta["color"]

        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val) in zip(
            [c1, c2, c3, c4],
            [
                ("Érintett ügyfelek", f"{n_seg:,}"),
                ("Szegmens bevétel", f"£{seg_rev/1e3:.1f}K"),
                ("Várható megmentés", f"£{saved/1e3:.1f}K"),
                ("Ügyfélenkénti ktg.", f"£{meta['cost_per_customer']}"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="--accent-color:{color}">
                  <div class="kpi-label">{lbl}</div>
                  <div class="kpi-value" style="font-size:1.4rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Akciólépések</div>', unsafe_allow_html=True)
        for i, step in enumerate(meta["playbook"], 1):
            st.markdown(f"""
            <div class="playbook-item">
              <div class="playbook-num" style="--seg-color:{color}">{i}</div>
              <div>{step}</div>
            </div>""", unsafe_allow_html=True)

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">📥 Lista exportálás – {focus_action.split("–")[0].strip()}</div>', unsafe_allow_html=True)

export_df = df[df["action"] == focus_action].copy()
display_export_cols = {
    "Customer ID": "Ügyfél ID",
    "rfm_segment": "RFM szegmens",
    "monetary_total": "Bevétel (£)",
    "frequency": "Vásárlás (db)",
    "recency_days": "Inaktivitás (nap)",
    "churn_proba": "Churn valószínűség",
    "return_ratio": "Visszaküldési arány",
    "action": "Ajánlott akció",
}
avail = {k: v for k, v in display_export_cols.items() if k in export_df.columns}
show_export = export_df[list(avail.keys())].rename(columns=avail).sort_values(
    "Bevétel (£)", ascending=False
)

fmt = {
    "Bevétel (£)": "£{:,.0f}",
    "Churn valószínűség": "{:.1%}",
    "Visszaküldési arány": "{:.1%}",
}
fmt = {k: v for k, v in fmt.items() if k in show_export.columns}

st.dataframe(
    show_export.style.format(fmt).background_gradient(subset=["Bevétel (£)"], cmap="YlOrRd"),
    use_container_width=True, hide_index=True, height=300,
)

col_exp1, col_exp2 = st.columns([1, 3])
with col_exp1:
    csv_buf = io.StringIO()
    show_export.to_csv(csv_buf, index=False)
    st.download_button(
        label=f"⬇️ CSV letöltés ({len(show_export):,} ügyfél)",
        data=csv_buf.getvalue(),
        file_name=f"action_list_{focus_action[:20].replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── ÖSSZEFOGLALÓ INSIGHT ──────────────────────────────────────────────────────
total_saved_all = sum(
    df[df["action"] == ak]["monetary_total"].sum() * m["expected_retention"]
    for ak, m in ACTION_META.items()
)
total_cost_all = sum(
    len(df[df["action"] == ak]) * m["cost_per_customer"]
    for ak, m in ACTION_META.items()
)
total_roi = (total_saved_all - total_cost_all) / total_cost_all * 100 if total_cost_all > 0 else 0

st.markdown(f"""
<div style="background:#1A1D27;border:1px solid #2A2D3A;border-left:4px solid {TEAL};
            border-radius:10px;padding:18px 22px;margin-top:8px;">
  <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:8px;color:{TEXT_PRI}">
    📊 Teljes kampány összefoglaló
  </div>
  <div style="display:flex;gap:32px;flex-wrap:wrap">
    <div>
      <div style="font-size:0.75rem;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.06em">Összes megmenthető bevétel</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{TEAL}">£{total_saved_all/1e3:.0f}K</div>
    </div>
    <div>
      <div style="font-size:0.75rem;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.06em">Teljes kampányköltség</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{AMBER}">£{total_cost_all:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.75rem;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.06em">Össz. kampány ROI</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{ACCENT}">{total_roi:.0f}%</div>
    </div>
  </div>
  <div style="margin-top:10px;color:{TEXT_SEC};font-size:0.86rem;line-height:1.6">
    A retenciós értékek a szegmens-specifikus iparági benchmark alapján becsültek.
    A tényleges eredmények az A/B tesztek és a kampány-optimalizáció alapján pontosítandók.
  </div>
</div>
""", unsafe_allow_html=True)
