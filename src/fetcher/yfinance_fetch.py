# src/fetcher/yfinance_fetch.py
"""
Incremental EOD fetcher (CSV-only)
- Stores raw CSVs in data/raw/<SYMBOL>.csv
- Writes standardized CSVs to data/standard/auto/<SYMBOL>.csv
- Keeps meta in data/meta/last_fetch.json
"""
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import logging
from typing import List, Optional
import pandas as pd
import yfinance as yf

# project-root aware paths
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STANDARD_DIR = PROJECT_ROOT / "data" / "standard" / "auto"
META_FILE = PROJECT_ROOT / "data" / "meta" / "last_fetch.json"
MASTER_CSV = PROJECT_ROOT / "data" / "EQUITY_L.csv"

# logging
LOGGER = logging.getLogger("yfinance_fetch")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")

def _load_meta():
    try:
        if META_FILE.exists():
            return json.loads(META_FILE.read_text())
    except Exception:
        LOGGER.exception("Failed loading meta file")
    return {}

def _save_meta(meta: dict):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2))

def _standardize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.reset_index() if "Date" in df.columns else df
    df = df.rename(columns={
        "Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Missing date column")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["symbol"] = symbol
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]]
    return df

def _read_existing_raw(symbol: str) -> pd.DataFrame:
    p = RAW_DIR / f"{symbol}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=["date"])
    except Exception:
        LOGGER.exception("Failed reading existing raw for %s", symbol)
        return pd.DataFrame()

def _append_raw(symbol: str, new_df: pd.DataFrame):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{symbol}.csv"
    if out.exists():
        try:
            old = pd.read_csv(out, parse_dates=["date"])
        except Exception:
            LOGGER.exception("Failed reading old raw for %s; overwrite", symbol)
            old = pd.DataFrame()
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
        combined = pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=["date"]).dropna(subset=["date"]).sort_values("date")
    else:
        new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
        combined = new_df.dropna(subset=["date"]).sort_values("date")
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(out, index=False)
    LOGGER.info("Saved raw csv for %s (%d rows)", symbol, len(combined))
    return combined

def _write_standard(df: pd.DataFrame, symbol: str):
    STANDARD_DIR.mkdir(parents=True, exist_ok=True)
    out = STANDARD_DIR / f"{symbol}.csv"
    df.to_csv(out, index=False)
    LOGGER.info("Wrote standard csv: %s (%d rows)", out, len(df))

def _yf_symbol(symbol: str):
    s = symbol.strip().upper()
    return s if "." in s else f"{s}.NS"

def fetch_history(symbol: str, start: str, end: Optional[str] = None, interval="1d") -> pd.DataFrame:
    yf_symbol = _yf_symbol(symbol)
    if end is None:
        end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    LOGGER.info("Fetching %s from %s to %s", symbol, start, end)
    t = yf.Ticker(yf_symbol)
    for attempt in range(3):
        try:
            df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
            if df is None or df.empty:
                LOGGER.info("No data returned for %s", symbol)
                return pd.DataFrame()
            df = df.reset_index()
            return df
        except Exception:
            LOGGER.exception("Fetch attempt failed for %s", symbol)
            time.sleep(2 + attempt)
    return pd.DataFrame()

def fetch_and_store_symbol(symbol: str, start: str = "2000-01-01"):
    base = symbol.split(".")[0].upper()
    existing = _read_existing_raw(base)
    if not existing.empty:
        last_date = pd.to_datetime(existing["date"]).max().date()
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        LOGGER.info("%s last date %s -> fetching from %s", base, last_date, fetch_start)
    else:
        fetch_start = start
    today = datetime.utcnow().date()
    if not existing.empty and last_date >= today:
        LOGGER.info("%s already up to date; skipping", base)
        return
    df_new = fetch_history(base, start=fetch_start)
    if df_new.empty:
        LOGGER.info("No new rows for %s", base)
        return
    try:
        df_std = _standardize_df(df_new, base)
    except Exception:
        LOGGER.exception("Standardization failed for %s", base)
        return
    combined = _append_raw(base, df_std)
    _write_standard(combined, base)
    # update meta
    meta = _load_meta()
    meta[base] = {"last_fetched": datetime.utcnow().isoformat(), "last_date_in_data": combined["date"].max()}
    _save_meta(meta)

def _read_master_symbols(master_csv: Path) -> List[str]:
    if not master_csv.exists():
        raise FileNotFoundError(master_csv)
    df = pd.read_csv(master_csv, dtype=str)
    col = next((c for c in df.columns if c.lower() == "symbol"), df.columns[0])
    syms = df[col].dropna().astype(str).str.strip().str.upper().tolist()
    return syms

def fetch_all_from_master(master_csv: Path = MASTER_CSV, start: str = "2000-01-01", full: bool = False, limit: Optional[int] = None):
    syms = _read_master_symbols(master_csv)
    if limit:
        syms = syms[:limit]
    LOGGER.info("Fetching %d symbols from master %s", len(syms), master_csv)
    for s in syms:
        try:
            if full:
                (RAW_DIR / f"{s}.csv").unlink(missing_ok=True)
            fetch_and_store_symbol(s, start=start)
            time.sleep(0.2)
        except Exception:
            LOGGER.exception("Failed fetch %s", s)
            continue
