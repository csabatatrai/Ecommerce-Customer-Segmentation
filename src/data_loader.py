"""Megosztott adatbetöltés – cache_resource egyszer tölt be, minden munkamenet osztja.

cache_data vs cache_resource:
  cache_data  → minden hívásnál másolatot készít  → sok user = sok memória
  cache_resource → egyetlen objektumot ad vissza  → sok user = 1× memória
Az adatok csak olvasásra használtak, ezért cache_resource biztonságos.
"""
import warnings
import pandas as pd
import streamlit as st
from pathlib import Path

_TX_COLS = [
    "InvoiceDate", "Customer ID", "Invoice",
    "Description", "Quantity", "Price", "LineTotal",
]


def _find_path(*candidates: str) -> Path | None:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    return None


def _compute_action(rfm_segment: str, churn_pred: int) -> str:
    """Akció újraszámítása aktuális szegmens + churn előrejelzés alapján."""
    if rfm_segment == "VIP Bajnokok":
        return (
            "🚨 VIP Veszélyben - Azonnali Retenció"
            if churn_pred == 1
            else "💎 VIP Stabil - Lojalitás Program"
        )
    if rfm_segment in ("Lemorzsolódó / Alvó", "Elvesztett / Inaktív"):
        return "⚠️  Lemorzsolódó - Win-Back Kampány"
    # Új / Ígéretes
    return (
        "⚠️  Lemorzsolódó - Win-Back Kampány"
        if churn_pred == 1
        else "✅ Stabil - Standard Kommunikáció"
    )


@st.cache_resource(show_spinner="Adatok betöltése...")
def load_churn_predictions() -> pd.DataFrame:
    """Churn előrejelzések + aktuális RFM szegmensek, Customer ID az index.

    Az rfm_segment és action oszlopok a teljes adatbázison futtatott
    customer_segments_current.parquet alapján kerülnek frissítésre, hogy
    a dashboard minden oldala az aktuális szegmentációt tükrözze.
    Azon ügyfelekre, akikre nincs aktuális szegmens, az eredeti érték marad.
    """
    path = _find_path(
        "data/processed/churn_predictions.parquet",
        "../data/processed/churn_predictions.parquet",
    )
    if path is None:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_parquet(path)

    if df.index.name != "Customer ID":
        if "Customer ID" in df.columns:
            df = df.set_index("Customer ID")
        elif "CustomerID" in df.columns:
            df = df.set_index("CustomerID")
            df.index.name = "Customer ID"
    df.index = df.index.astype(str)

    # ── Aktuális szegmens felülírása ───────────────────────────────────────────
    seg_path = _find_path(
        "data/processed/customer_segments_current.parquet",
        "../data/processed/customer_segments_current.parquet",
    )
    if seg_path is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg = pd.read_parquet(seg_path, columns=["Segment"])
        seg.index = seg.index.astype(str)

        # Csak azokat az ügyfeleket frissítjük, akik az aktuális fájlban szerepelnek
        common = df.index.intersection(seg.index)
        df.loc[common, "rfm_segment"] = seg.loc[common, "Segment"]

        # action újraszámítása az aktuális szegmens alapján
        df.loc[common, "action"] = [
            _compute_action(str(df.at[cid, "rfm_segment"]), int(df.at[cid, "churn_pred"]))
            for cid in common
        ]

    return df


@st.cache_resource(show_spinner="Aktuális szegmensek betöltése...")
def load_current_segments() -> pd.DataFrame:
    """Teljes adatbázison alapuló klaszterezés – operatív szegmentáció.

    A 02-es notebook 6. szekciójában generált customer_segments_current.parquet
    tartalmazza a teljes tranzakcióhistórián futtatott RFM klaszterezést.
    """
    path = _find_path(
        "data/processed/customer_segments_current.parquet",
        "../data/processed/customer_segments_current.parquet",
    )
    if path is None:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_parquet(path)

    if df.index.name != "Customer ID":
        if "Customer ID" in df.columns:
            df = df.set_index("Customer ID")
    df.index = df.index.astype(str)
    return df


@st.cache_resource(show_spinner="Tranzakciók betöltése...")
def load_transactions() -> pd.DataFrame:
    """Nyers tranzakciók – csak a szükséges oszlopok töltődnek be."""
    path = _find_path(
        "data/processed/online_retail_ready_for_rfm.parquet",
        "../data/processed/online_retail_ready_for_rfm.parquet",
    )
    if path is None:
        return pd.DataFrame()

    df = pd.read_parquet(path, columns=_TX_COLS)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(str)
    return df
