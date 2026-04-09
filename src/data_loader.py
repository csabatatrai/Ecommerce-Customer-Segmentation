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


@st.cache_resource(show_spinner="Adatok betöltése...")
def load_churn_predictions() -> pd.DataFrame:
    """Churn előrejelzések + RFM szegmensek, Customer ID az index."""
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
