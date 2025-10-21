# src/preprocess/standardize.py
"""
Standardize raw OHLC CSVs into the canonical schema:
    date,open,high,low,close,adj_close,volume,symbol

Features:
- Robust column mapping for common raw CSV variants.
- Optionally fill missing trading days using pandas_market_calendars (preferred).
  If pandas_market_calendars is not available, falls back to business days (Mon-Fri).
- Optionally forward-fill missing OHLC data for non-trading days or holidays.
- Handles cases where Adj Close is missing (keeps as-is) or present.
- Ensures sorted ascending by date, removes duplicates.
"""

from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Iterable, List

LOGGER = logging.getLogger("preprocess.standardize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

CANONICAL_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]

# Common column name variants mapping -> canonical
COLUMN_MAP = {
    "date": ["date", "timestamp", "dt", "trade_date"],
    "open": ["open", "o", "price_open"],
    "high": ["high", "h", "price_high"],
    "low": ["low", "l", "price_low"],
    "close": ["close", "c", "last", "price_close"],
    "adj_close": ["adj_close", "adjclose", "adj close", "adjusted_close", "adjusted close"],
    "volume": ["volume", "vol", "v"],
    "symbol": ["symbol", "ticker", "code"]
}


def _infer_column_map(columns: Iterable[str]) -> dict:
    """Infer mapping from input column names to canonical names."""
    columns_lower = [c.lower().strip() for c in columns]
    mapping = {}
    for canon, variants in COLUMN_MAP.items():
        for col in columns_lower:
            if col == canon:
                mapping[col] = canon
            else:
                for v in variants:
                    if col == v:
                        mapping[col] = canon
                        break
                if mapping.get(col) == canon:
                    break
    # build reverse mapping (original -> canonical)
    rev = {}
    for orig, orig_lower in zip(columns, columns_lower):
        if orig_lower in mapping:
            rev[orig] = mapping[orig_lower]
    return rev


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns based on heuristics; returns df with renamed columns (lowercase)."""
    orig_cols = list(df.columns)
    lower_map = {c: c.lower().strip().replace(" ", "_") for c in orig_cols}
    df = df.rename(columns=lower_map)
    inferred = _infer_column_map(df.columns)
    # rename using inferred map
    rename_dict = {}
    for col in df.columns:
        # prefer direct mapping if canonical already present
        if col in CANONICAL_COLS:
            rename_dict[col] = col
        elif col in inferred:
            rename_dict[col] = inferred[col]
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def _ensure_columns(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Make sure canonical columns exist; create with defaults if missing."""
    for c in CANONICAL_COLS:
        if c not in df.columns:
            if c == "volume":
                df[c] = 0
            elif c == "symbol":
                df[c] = symbol if symbol else pd.NA
            else:
                df[c] = pd.NA
    # Keep only canonical columns in canonical order
    df = df[CANONICAL_COLS]
    return df


def _to_date_str(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convert date column to YYYY-MM-DD strings (no timezone)."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        LOGGER.warning("Some dates could not be parsed and will be dropped.")
        df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    return df


def _get_trading_calendar(start: str, end: str, market: str = "NSE") -> pd.DatetimeIndex:
    """
    Return trading days between start and end inclusive.
    Tries to use pandas_market_calendars (best) and falls back to business days (Mon-Fri).
    """
    try:
        import pandas_market_calendars as mcal  # optional dependency
        if market.upper() in ("NSE", "NSEI", "NIFTY"):
            cal = mcal.get_calendar("NSE")
        else:
            # default to NYSE if unknown
            cal = mcal.get_calendar("NYSE")
        schedule = cal.schedule(start_date=start, end_date=end)
        trading_days = schedule.index.normalize()
        return pd.DatetimeIndex(trading_days)
    except Exception:
        LOGGER.info("pandas_market_calendars not available or failed; falling back to business days.")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return pd.bdate_range(start=start_dt, end=end_dt)  # Mon-Fri


def standardize_raw_csv(
    raw_csv_path: Path,
    out_csv_path: Optional[Path] = None,
    symbol: Optional[str] = None,
    fill_missing_days: bool = True,
    fill_method: str = "ffill",
    market: str = "NSE",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    drop_partial_days: bool = False
) -> pd.DataFrame:
    """
    Read a raw CSV, standardize columns and optionally fill missing trading days.

    Parameters
    ----------
    raw_csv_path : Path
        Path to raw CSV file to standardize.
    out_csv_path : Optional[Path]
        If provided, writes standardized CSV to this path.
    symbol : Optional[str]
        If symbol column is missing, set this symbol for all rows.
    fill_missing_days : bool
        If True, reindex to trading calendar days and fill missing rows using fill_method.
    fill_method : str
        One of 'ffill', 'bfill', or 'none'. If 'none' missing days will contain NaN for OHLC.
    market : str
        Market identifier for trading calendar (default 'NSE').
    start_date, end_date : Optional[str]
        Bounds for trading calendar if fill_missing_days==True; if None, uses min/max dates in data.
    drop_partial_days : bool
        If True, drop weekdays where volume==0 and OHLC are NaN (useful to remove non-trading placeholders).

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame (and writes CSV if out_csv_path provided).
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"{raw_csv_path} not found")

    LOGGER.info("Standardizing %s", raw_csv_path)
    # read with flexible parsing
    df = pd.read_csv(raw_csv_path, low_memory=False)
    df = _rename_columns(df)
    df = _ensure_columns(df, symbol)
    df = _to_date_str(df, "date")
    # Cast numeric columns to floats/ints where possible
    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("Int64")
    if symbol:
        df["symbol"] = symbol
    else:
        # upper-case symbol column if present
        try:
            df["symbol"] = df["symbol"].astype(str).str.upper()
        except Exception:
            df["symbol"] = df["symbol"]

    # remove duplicate dates keeping last (assuming later rows are more recent)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    # Optionally reindex to trading days
    if fill_missing_days:
        start = start_date if start_date else df["date"].min()
        end = end_date if end_date else df["date"].max()
        trading_days = _get_trading_calendar(start, end, market=market)
        trading_days_str = trading_days.strftime("%Y-%m-%d")
        df = df.set_index("date")
        # reindex - missing days will have NaN
        df = df.reindex(trading_days_str)
        # ensure symbol column present after reindex
        df["symbol"] = symbol if symbol else df["symbol"].ffill().bfill()
        if fill_method and fill_method.lower() != "none":
            if fill_method.lower() == "ffill":
                df[["open", "high", "low", "close", "adj_close"]] = df[["open", "high", "low", "close", "adj_close"]].ffill()
                df["volume"] = df["volume"].fillna(0).astype("Int64")
            elif fill_method.lower() == "bfill":
                df[["open", "high", "low", "close", "adj_close"]] = df[["open", "high", "low", "close", "adj_close"]].bfill()
                df["volume"] = df["volume"].fillna(0).astype("Int64")
            else:
                LOGGER.warning("Unknown fill_method '%s' - not filling", fill_method)

        df = df.reset_index().rename(columns={"index": "date"})
    else:
        # keep only rows from start_date/end_date if provided
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

    # If requested, drop rows that appear to be partial placeholders (e.g., volume==0 and NaNs)
    if drop_partial_days:
        condition_partial = (df[["open", "high", "low", "close", "adj_close"]].isna().all(axis=1)) & (df["volume"] == 0)
        removed = condition_partial.sum()
        if removed:
            LOGGER.info("Dropping %d partial/empty rows", int(removed))
            df = df[~condition_partial]

    # final ordering and types
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]]
    # volumes to integer where possible
    try:
        df["volume"] = df["volume"].astype("Int64")
    except Exception:
        # leave as-is if conversion fails
        pass

    if out_csv_path:
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        LOGGER.info("Wrote standardized CSV: %s (%d rows)", out_csv_path, len(df))

    return df


def standardize_folder(
    raw_dir: Path,
    out_dir: Path,
    pattern: str = "*.csv",
    symbol_from_filename: bool = True,
    fill_missing_days: bool = True,
    fill_method: str = "ffill",
    market: str = "NSE"
) -> List[Path]:
    """
    Standardize all CSVs in raw_dir and write outputs to out_dir.

    Returns list of output paths written.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in sorted(raw_dir.glob(pattern)):
        try:
            # derive symbol from filename if desired (e.g. RELIANCE.NS.csv -> RELIANCE.NS)
            symbol = None
            if symbol_from_filename:
                symbol = p.stem.upper()
            out_path = out_dir / f"{p.stem}.csv"
            standardize_raw_csv(
                raw_csv_path=p,
                out_csv_path=out_path,
                symbol=symbol,
                fill_missing_days=fill_missing_days,
                fill_method=fill_method,
                market=market
            )
            written.append(out_path)
        except Exception as e:
            LOGGER.exception("Failed to standardize %s: %s", p, e)
    return written


# quick CLI for convenience
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standardize raw OHLC CSV(s) to canonical schema.")
    parser.add_argument("--raw", type=str, required=True, help="raw csv file or directory")
    parser.add_argument("--out", type=str, default="data/standard/auto", help="output file or directory")
    parser.add_argument("--pattern", type=str, default="*.csv", help="glob pattern when raw is a directory")
    parser.add_argument("--fill_missing_days", action="store_true", help="reindex to trading days and fill missing")
    parser.add_argument("--fill_method", type=str, default="ffill", choices=["ffill", "bfill", "none"], help="fill method for missing days")
    parser.add_argument("--market", type=str, default="NSE", help="market calendar to use")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    if raw.is_dir():
        standardize_folder(raw, out, pattern=args.pattern, fill_missing_days=args.fill_missing_days, fill_method=args.fill_method, market=args.market)
    else:
        out_path = out if not out.is_dir() else out / raw.name
        standardize_raw_csv(raw, out_path, fill_missing_days=args.fill_missing_days, fill_method=args.fill_method, market=args.market)
