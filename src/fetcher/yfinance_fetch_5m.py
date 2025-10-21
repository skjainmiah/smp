# src/fetcher/yfinance_fetch_5m.py
"""
5-minute intraday fetcher (CSV-only)
- Stores monthly files under data/intraday_5m/<SYMBOL>/<YYYY-MM>.csv
- Maintains data/meta/last_fetch_5m.json
- Threaded fetch for speed
- Configurable via arguments and env vars
"""
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import logging
import concurrent.futures
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
import os
import tempfile
import shutil

# --- robust project root detection (avoid src/src) ---
THIS = Path(__file__).resolve()
def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / "src").exists() or (p / "data").exists() or (p / ".git").exists() or (p / "README.md").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    # fallback: go up two levels
    if len(start.parents) >= 2:
        return start.parents[2]
    return start.parents[1]

PROJECT_ROOT = find_repo_root(THIS.parent)

# --- paths & defaults (env-overridable) ---
INTRADAY_DIR = PROJECT_ROOT / "data" / "intraday_5m"
META_FILE = PROJECT_ROOT / "data" / "meta" / "last_fetch_5m.json"
FAILED_LOG = PROJECT_ROOT / "data" / "meta" / "failed_5m_symbols.csv"
DEFAULT_MASTER_CSV = PROJECT_ROOT / "data" / "EQUITY_L.csv"

INTERVAL = os.getenv("YF_5M_INTERVAL", "5m")
MAX_DAYS_DEFAULT = int(os.getenv("YF_5M_MAX_DAYS", "59"))
MAX_WORKERS = int(os.getenv("YF_5M_MAX_WORKERS", "8"))
RETRY = int(os.getenv("YF_5M_RETRY", "3"))
KEEP_MONTHS_DEFAULT = int(os.getenv("YF_5M_KEEP_MONTHS", "3"))

LOGGER = logging.getLogger("yfinance_5m")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")

# --- helpers for meta / failures ---
def _load_meta() -> Dict:
    try:
        if META_FILE.exists():
            return json.loads(META_FILE.read_text())
    except Exception:
        LOGGER.exception("Failed loading 5m meta")
    return {}

def _save_meta(meta: Dict):
    try:
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        META_FILE.write_text(json.dumps(meta, indent=2))
    except Exception:
        LOGGER.exception("Failed saving 5m meta")

def _log_failed(symbol: str):
    try:
        FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
        if FAILED_LOG.exists():
            df = pd.read_csv(FAILED_LOG)
        else:
            df = pd.DataFrame(columns=["symbol","ts"])
    except Exception:
        df = pd.DataFrame(columns=["symbol","ts"])
    df = pd.concat([df, pd.DataFrame([[symbol, datetime.utcnow().isoformat()]], columns=["symbol","ts"])], ignore_index=True)
    try:
        df.to_csv(FAILED_LOG, index=False)
    except Exception:
        LOGGER.exception("Failed writing failed log for %s", symbol)

# --- master list reading ---
def _read_master(master_csv: Path) -> List[str]:
    if not master_csv or not Path(master_csv).exists():
        LOGGER.warning("Master CSV not found: %s", master_csv)
        return []
    df = pd.read_csv(master_csv, dtype=str)
    # try to find symbol-like column
    col = next((c for c in df.columns if c.lower() in ("symbol","ticker")), df.columns[0])
    return df[col].dropna().astype(str).str.strip().str.upper().tolist()

# --- yfinance symbol transform ---
def _yf_symbol(s: str) -> str:
    s = s.strip().upper()
    return s if "." in s else f"{s}.NS"

# --- fetch for single symbol ---
def _fetch_5m_for(symbol: str, days: int = MAX_DAYS_DEFAULT) -> pd.DataFrame:
    """
    Fetch intraday 5m history for the last `days` days for `symbol`.
    Returns DataFrame with datetime (tz-aware UTC), open, high, low, close, volume, symbol.
    """
    yf_sym = _yf_symbol(symbol)
    end = datetime.utcnow()
    start = end - timedelta(days=max(1, days))
    LOGGER.info("Fetching 5m for %s from %s to %s (interval=%s)", symbol, start.date(), end.date(), INTERVAL)
    for attempt in range(1, RETRY+1):
        try:
            df = yf.download(yf_sym, start=start, end=end, interval=INTERVAL, progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                LOGGER.warning("No 5m data returned for %s (attempt %d)", symbol, attempt)
                return pd.DataFrame()
            # reset index and flatten
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            # map possible date column names to "datetime"
            if "datetime" not in df.columns:
                if "date" in df.columns:
                    df = df.rename(columns={"date":"datetime"})
                elif "index" in df.columns:
                    df = df.rename(columns={"index":"datetime"})
            if not all(c in df.columns for c in ("datetime","open","high","low","close")):
                LOGGER.warning("5m fetch for %s missing required columns, got: %s", symbol, list(df.columns))
                return pd.DataFrame()
            # ensure datetime tz-aware (UTC)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.dropna(subset=["datetime"])
            df["symbol"] = symbol
            # re-order expected columns
            cols = ["datetime","open","high","low","close"]
            if "volume" in df.columns:
                cols.append("volume")
            cols += [c for c in df.columns if c not in cols and c != "symbol"]
            cols.append("symbol")
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception:
            LOGGER.exception("Attempt %d failed fetching 5m for %s", attempt, symbol)
            time.sleep(1 + attempt)
    # failed after retries
    _log_failed(symbol)
    return pd.DataFrame()

# --- atomic save to monthly files ---
def _save_monthly_atomic(df: pd.DataFrame, symbol: str):
    if df.empty:
        return
    # ensure month column
    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    out_dir = INTRADAY_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    for month, g in df.groupby("month"):
        out_file = out_dir / f"{month}.csv"
        # read existing
        if out_file.exists():
            try:
                old = pd.read_csv(out_file, parse_dates=["datetime"])
                old["datetime"] = pd.to_datetime(old["datetime"], utc=True, errors="coerce")
            except Exception:
                LOGGER.exception("Failed reading existing month file for %s %s", symbol, month)
                old = pd.DataFrame()
            g["datetime"] = pd.to_datetime(g["datetime"], utc=True, errors="coerce")
            combined = pd.concat([old, g], ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
        else:
            combined = g.sort_values("datetime")
        # atomic write
        tmpf = out_file.with_suffix(".csv.tmp")
        try:
            combined.to_csv(tmpf, index=False)
            tmpf.replace(out_file)
        except Exception:
            # fallback: write directly
            try:
                combined.to_csv(out_file, index=False)
            except Exception:
                LOGGER.exception("Failed writing monthly file for %s %s", symbol, month)
    LOGGER.info("Saved %s (%d rows) into monthly files", symbol, len(df))

def _cleanup_old(symbol: str, keep_months: int = KEEP_MONTHS_DEFAULT):
    out_dir = INTRADAY_DIR / symbol
    if not out_dir.exists():
        return
    months = sorted([p.stem for p in out_dir.glob("*.csv")])
    if len(months) > keep_months:
        to_delete = months[:-keep_months]
        for old in to_delete:
            try:
                fp = out_dir / f"{old}.csv"
                if fp.exists():
                    fp.unlink()
                    LOGGER.info("Deleted old month %s for %s", old, symbol)
            except Exception:
                LOGGER.exception("Could not delete old month %s for %s", old, symbol)

# --- per-symbol workflow ---
def _process_symbol(symbol: str, days: int = MAX_DAYS_DEFAULT, keep_months: int = KEEP_MONTHS_DEFAULT):
    try:
        df = _fetch_5m_for(symbol, days)
        if df.empty:
            LOGGER.info("No new 5m df for %s", symbol)
            return {"symbol": symbol, "rows": 0, "ok": False}
        _save_monthly_atomic(df, symbol)
        _cleanup_old(symbol, keep_months=keep_months)
        meta = _load_meta()
        meta[symbol] = {"last_fetch": datetime.utcnow().isoformat()}
        _save_meta(meta)
        return {"symbol": symbol, "rows": len(df), "ok": True}
    except Exception:
        LOGGER.exception("Processing failed for %s", symbol)
        _log_failed(symbol)
        return {"symbol": symbol, "rows": 0, "ok": False}

# --- public API: fetch_all_5m ---
def fetch_all_5m(master_csv: Optional[Path] = None, max_days: Optional[int] = None, limit: Optional[int] = None, max_workers: Optional[int] = None, keep_months: Optional[int] = None) -> Dict:
    """
    Fetch 5m intraday for all symbols in master_csv.
    Parameters:
      - master_csv: Path to master CSV listing symbols (defaults data/EQUITY_L.csv)
      - max_days: how many recent days to fetch (defaults env/default)
      - limit: limit number of symbols (for testing)
      - max_workers: threadpool workers (defaults env/default)
      - keep_months: how many monthly files to keep per symbol
    Returns dict summary: {total, succeeded, failed, rows_fetched, details}
    """
    master_csv = Path(master_csv) if master_csv else DEFAULT_MASTER_CSV
    days = int(max_days) if max_days is not None else MAX_DAYS_DEFAULT
    workers = int(max_workers) if max_workers is not None else MAX_WORKERS
    keep_months = int(keep_months) if keep_months is not None else KEEP_MONTHS_DEFAULT

    symbols = _read_master(master_csv)
    if limit:
        symbols = symbols[:limit]
    if not symbols:
        LOGGER.warning("No symbols to fetch (master %s empty)", master_csv)
        return {"total": 0, "succeeded": 0, "failed": 0, "rows": 0, "details": []}

    LOGGER.info("Starting 5m fetch for %d symbols (days=%s, workers=%s)", len(symbols), days, workers)
    results = []
    rows_total = 0
    succeeded = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_symbol, s, days, keep_months): s for s in symbols}
        for fut in concurrent.futures.as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.get("ok"):
                    succeeded += 1
                    rows_total += int(res.get("rows", 0))
                else:
                    failed += 1
            except Exception:
                LOGGER.exception("Symbol %s failed in executor", s)
                failed += 1
                results.append({"symbol": s, "rows": 0, "ok": False})
    summary = {"total": len(symbols), "succeeded": succeeded, "failed": failed, "rows": rows_total, "details": results}
    LOGGER.info("5m fetch completed: %d total, %d succeeded, %d failed, %d rows", summary["total"], summary["succeeded"], summary["failed"], summary["rows"])
    return summary

# --- allow running as script for testing ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch 5m intraday data for symbols in master CSV")
    parser.add_argument("--master", type=str, default=str(DEFAULT_MASTER_CSV), help="master CSV path")
    parser.add_argument("--days", type=int, default=1, help="recent days to fetch (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="limit number of symbols for testing")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="threadpool workers")
    parser.add_argument("--keep-months", type=int, default=KEEP_MONTHS_DEFAULT, help="how many monthly files to keep")
    args = parser.parse_args()

    out = fetch_all_5m(master_csv=Path(args.master), max_days=args.days, limit=args.limit, max_workers=args.workers, keep_months=args.keep_months)
    print(json.dumps(out, indent=2, default=str))
