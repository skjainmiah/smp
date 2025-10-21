# src/analysis/recommendation_tracker_csv.py
"""
CSV-backed recommendation tracker and utilities.

Provides:
- store_recommendations_csv(df) -> out_path
- daily_check_outcomes_csv()
- fetch_global_indices_csv(indices: list)

This module is intentionally defensive: creates data dirs, logs actions,
and uses a robust repo-root discovery so it works regardless of how Python is invoked.

Notes:
- Fundamentals are fetched via yfinance.info (best-effort) and cached to
  data/meta/fundamentals_cache.json to avoid repeated network calls.
- fundamentals_details is stored as a compact JSON string in the CSV (so you
  can later parse it).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
import pandas as pd
import pytz
import yfinance as yf
import numpy as np
import tempfile
import os

# minimal logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")
LOG = logging.getLogger("recommendation_tracker_csv")

# --- repo root detection (robust) ---
THIS = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / "src").exists() or (p / "data").exists() or (p / ".git").exists() or (p / "README.md").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    if len(start.parents) >= 2:
        return start.parents[2]
    return start.parents[1]

REPO_ROOT = find_repo_root(THIS.parent)

# --- paths ---
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
EOD_DIR = DATA_DIR / "standard" / "auto"
META_DIR = DATA_DIR / "meta"
RECOMMENDATIONS_CSV = DATA_DIR / "recommendations.csv"       # cumulative tracker
OUTCOMES_CSV = DATA_DIR / "outcomes.csv"
GLOBAL_MARKETS_CSV = DATA_DIR / "global_markets.csv"
FUND_CACHE_FILE = META_DIR / "fundamentals_cache.json"

# ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EOD_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# timezone
TZ = pytz.timezone("Asia/Kolkata")

# -------------------------
# Fundamentals caching helpers
# -------------------------
def _load_fund_cache() -> Dict[str, dict]:
    try:
        if FUND_CACHE_FILE.exists():
            return json.loads(FUND_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        LOG.exception("Failed loading fundamentals cache; starting fresh.")
    return {}

def _save_fund_cache(cache: Dict[str, dict]):
    try:
        FUND_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        FUND_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        LOG.exception("Failed saving fundamentals cache.")

# -------------------------
# Fundamentals fetcher + scorer
# -------------------------
def fetch_fundamentals_for_symbol(symbol: str, use_cache: bool = True, force_refresh: bool = False) -> dict:
    """
    Use yfinance.Ticker.info to fetch basic fundamentals.
    Returns a dict with numeric fields when available.
    Uses a local cache (data/meta/fundamentals_cache.json) to avoid repeated requests.
    """
    sym_key = symbol.strip().upper()
    if "." not in sym_key:
        yf_sym = f"{sym_key}.NS"
    else:
        yf_sym = sym_key

    cache = _load_fund_cache() if use_cache else {}
    if use_cache and not force_refresh and sym_key in cache:
        LOG.debug("Using cached fundamentals for %s", sym_key)
        return cache[sym_key]

    info = {}
    try:
        t = yf.Ticker(yf_sym)
        info_raw = {}
        try:
            # note: .info may be rate-limited; best-effort
            info_raw = t.info or {}
        except Exception:
            # fallback: try history/meta
            LOG.debug("yfinance.info failed for %s; attempting minimal fields", yf_sym)
            info_raw = {}
        # safe extraction
        def _get(k):
            try:
                return info_raw.get(k)
            except Exception:
                return None
        info["yf_symbol"] = yf_sym
        info["trailingPE"] = _get("trailingPE")
        info["forwardPE"] = _get("forwardPE")
        info["priceToBook"] = _get("priceToBook")
        info["marketCap"] = _get("marketCap")
        info["epsTrailingTwelveMonths"] = _get("epsTrailingTwelveMonths")
        info["totalRevenue"] = _get("totalRevenue") or _get("revenue")
        # timestamp
        info["fetched_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:
        LOG.exception("Failed fetching fundamentals for %s", symbol)
        info = {"yf_symbol": yf_sym, "fetched_at": datetime.now(timezone.utc).isoformat()}

    # store in cache
    try:
        cache[sym_key] = info
        if use_cache:
            _save_fund_cache(cache)
    except Exception:
        LOG.exception("Failed updating fundamentals cache for %s", sym_key)

    return info

def fundamentals_score_and_label(fund: dict) -> Tuple[float, str]:
    """
    Simple scoring approach (0..1) using PE and PB normalized heuristics.
    Returns: (score (0..1), label string)
    Labels: Worst (0-0.25), Bad (0.25-0.5), Better (0.5-0.75), Best (0.75-1)
    """
    # safe numeric extraction
    pe = fund.get("trailingPE")
    pb = fund.get("priceToBook")
    # normalize PE: prefer lower; treat PE None as neutral ~ median
    def score_pe(pe_val):
        try:
            if pe_val is None or pe_val <= 0 or np.isnan(pe_val):
                return 0.5
            pe_val = float(pe_val)
            if pe_val < 10: return 1.0
            if pe_val < 20: return 0.8
            if pe_val < 30: return 0.6
            if pe_val < 40: return 0.4
            return 0.2
        except Exception:
            return 0.5
    def score_pb(pb_val):
        try:
            if pb_val is None or np.isnan(pb_val):
                return 0.5
            pb_val = float(pb_val)
            if pb_val < 1: return 1.0
            if pb_val < 2: return 0.8
            if pb_val < 3: return 0.6
            if pb_val < 5: return 0.4
            return 0.2
        except Exception:
            return 0.5

    pe_s = score_pe(pe)
    pb_s = score_pb(pb)
    eps = fund.get("epsTrailingTwelveMonths")
    eps_s = 1.0 if (eps is not None and isinstance(eps, (int,float)) and eps > 0) else 0.5
    score = 0.5*pe_s + 0.3*pb_s + 0.2*eps_s
    score = float(max(0.0, min(1.0, score)))
    if score < 0.25:
        label = "Worst"
    elif score < 0.5:
        label = "Bad"
    elif score < 0.75:
        label = "Better"
    else:
        label = "Best"
    return score, label

# -------------------------
# Helper utilities
# -------------------------
def _ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def _today_str():
    return datetime.now(TZ).strftime("%Y%m%d")

def _atomic_write_csv(df: pd.DataFrame, out_path: Path):
    """
    Write CSV to a temporary file then rename to out_path for atomicity.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp_", dir=str(out_path.parent))
    os.close(fd)
    try:
        # write via pandas (ensure tmp endswith .csv)
        tmp_path = Path(tmp)
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
    finally:
        if Path(tmp).exists():
            try:
                Path(tmp).unlink()
            except Exception:
                pass

# -------------------------
# store_recommendations_csv
# -------------------------
def store_recommendations_csv(df: pd.DataFrame, enrich_fundamentals: bool = True, force_refresh_fund: bool = False) -> Optional[Path]:
    """
    Store recommendations:
     - enrich with fundamentals (best-effort)
     - append to data/recommendations.csv (cumulative)
     - write results/live_recommendations_YYYYMMDD.csv

    Returns path to the daily results file, or None if nothing written.
    """
    try:
        if df is None or df.empty:
            LOG.info("store_recommendations_csv called with empty df.")
            return None

        # copy and ensure expected columns
        expected_cols = [
            "symbol","name","action","category","strategy","confidence",
            "latest_close","support","resistance","target","stoploss","buy_zone","date"
        ]
        df = df.copy()
        df = _ensure_columns(df, expected_cols)

        # normalize date field to YYYY-MM-DD
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["date"] = datetime.now(TZ).strftime("%Y-%m-%d")

        # Enrich with fundamentals (best-effort) for unique symbols
        if enrich_fundamentals:
            try:
                unique_syms = sorted(df["symbol"].dropna().astype(str).unique().tolist())
                LOG.info("Fetching fundamentals for %d unique symbols (best-effort)...", len(unique_syms))
                fund_rows = []
                for sym in unique_syms:
                    try:
                        fund = fetch_fundamentals_for_symbol(sym, use_cache=True, force_refresh=force_refresh_fund)
                        score, label = fundamentals_score_and_label(fund)
                        fund_rows.append({
                            "symbol": sym,
                            "fundamentals_score": score,
                            "fundamentals_label": label,
                            "fundamentals_details": json.dumps(fund, separators=(",", ":"), ensure_ascii=False)
                        })
                    except Exception:
                        LOG.exception("Fundamentals fetch/score failed for %s", sym)
                        fund_rows.append({
                            "symbol": sym,
                            "fundamentals_score": None,
                            "fundamentals_label": None,
                            "fundamentals_details": None
                        })
                fund_df = pd.DataFrame(fund_rows)
                df = df.merge(fund_df, on="symbol", how="left")
            except Exception:
                LOG.exception("Fundamentals enrichment failed; continuing without fundamentals.")

        # write daily results file (timestamped) atomically
        out_fname = RESULTS_DIR / f"live_recommendations_{_today_str()}.csv"
        try:
            _atomic_write_csv(df, out_fname)
            LOG.info("Wrote daily results -> %s (%d rows)", out_fname, len(df))
        except Exception:
            LOG.exception("Failed atomic write for daily results; falling back to plain to_csv.")
            df.to_csv(out_fname, index=False)
            LOG.info("Wrote daily results (non-atomic) -> %s (%d rows)", out_fname, len(df))

        # append to cumulative recommendations tracker (upsert by symbol+date+action)
        if RECOMMENDATIONS_CSV.exists():
            try:
                existing = pd.read_csv(RECOMMENDATIONS_CSV, low_memory=False)
            except Exception:
                LOG.exception("Failed reading existing recommendations csv; will overwrite.")
                existing = pd.DataFrame()
            if not existing.empty:
                combined = pd.concat([existing, df], ignore_index=True, sort=False)
            else:
                combined = df
            try:
                combined = combined.drop_duplicates(subset=["symbol","date","action"], keep="last")
            except Exception:
                LOG.exception("Dropping duplicates failed; keeping full combined set.")
            # atomic write updated cumulative
            try:
                _atomic_write_csv(combined, RECOMMENDATIONS_CSV)
            except Exception:
                combined.to_csv(RECOMMENDATIONS_CSV, index=False)
        else:
            try:
                _atomic_write_csv(df, RECOMMENDATIONS_CSV)
            except Exception:
                df.to_csv(RECOMMENDATIONS_CSV, index=False)

        LOG.info("Updated cumulative recommendations -> %s (total %d rows)", RECOMMENDATIONS_CSV, len(pd.read_csv(RECOMMENDATIONS_CSV)))
        return out_fname
    except Exception:
        LOG.exception("Failed storing recommendations CSV.")
        raise

# -------------------------
# daily_check_outcomes_csv
# -------------------------
def daily_check_outcomes_csv():
    """
    Check open recommendations in data/recommendations.csv against latest EOD prices
    and append outcome rows when target or stoploss is hit.
    """
    try:
        if not RECOMMENDATIONS_CSV.exists():
            LOG.info("No recommendations CSV found; skipping outcome check.")
            return

        recs = pd.read_csv(RECOMMENDATIONS_CSV, low_memory=False)
        if recs.empty:
            LOG.info("Recommendations CSV empty; skipping outcome check.")
            return

        # ensure columns
        recs = _ensure_columns(recs, ["symbol","date","target","stoploss","status","closed_at","days_to_target"])
        # focus on open recommendations
        open_mask = recs["status"].isnull() | (recs["status"] == "") | (recs["status"].astype(str).str.upper() == "OPEN")
        open_recs = recs[open_mask].copy()
        LOG.info("Checking outcomes for %d open recommendations.", len(open_recs))

        outcomes = []
        for _, r in open_recs.iterrows():
            sym = str(r.get("symbol")).strip()
            target = r.get("target")
            stop = r.get("stoploss")
            rec_date = r.get("date")
            # read EOD file for symbol
            eod_file = EOD_DIR / f"{sym}.csv"
            if not eod_file.exists():
                LOG.debug("EOD file not found for %s; skipping", sym)
                continue
            try:
                df = pd.read_csv(eod_file, parse_dates=["date"], low_memory=False)
            except Exception:
                LOG.exception("Failed reading EOD for %s", sym)
                continue
            if df.empty:
                continue
            # find rows after recommendation date (inclusive)
            try:
                if pd.isna(rec_date):
                    rec_dt = None
                else:
                    rec_dt = pd.to_datetime(rec_date, errors="coerce")
            except Exception:
                rec_dt = None
            # if rec_dt present, consider rows from that date onward
            if rec_dt is not None and not pd.isna(rec_dt):
                consider = df[df["date"] >= rec_dt.strftime("%Y-%m-%d")]
            else:
                consider = df
            # iterate looking for first hit
            hit_type = None
            hit_date = None
            hit_price = None
            for _, day in consider.iterrows():
                close = float(day.get("close", day.get("Close", 0) or 0))
                if pd.notna(target) and float(target) > 0 and close >= float(target):
                    hit_type = "TARGET_HIT"
                    hit_date = pd.to_datetime(day["date"]).strftime("%Y-%m-%d")
                    hit_price = close
                    break
                if pd.notna(stop) and float(stop) > 0 and close <= float(stop):
                    hit_type = "STOP_HIT"
                    hit_date = pd.to_datetime(day["date"]).strftime("%Y-%m-%d")
                    hit_price = close
                    break
            if hit_type:
                outcomes.append({
                    "symbol": sym,
                    "rec_date": rec_date,
                    "event_type": hit_type,
                    "event_date": hit_date,
                    "price": hit_price,
                    "note": ""
                })
                # update the recs DataFrame
                idx = recs[(recs["symbol"]==sym) & (recs["date"]==r.get("date"))].index
                if len(idx):
                    i = idx[0]
                    recs.at[i, "status"] = hit_type
                    recs.at[i, "closed_at"] = hit_date
                    # days to target
                    try:
                        days_delta = (pd.to_datetime(hit_date) - pd.to_datetime(r.get("date"))).days
                    except Exception:
                        days_delta = None
                    recs.at[i, "days_to_target"] = days_delta

        # append outcomes to OUTCOMES_CSV
        if outcomes:
            out_df = pd.DataFrame(outcomes)
            if OUTCOMES_CSV.exists():
                try:
                    old = pd.read_csv(OUTCOMES_CSV, low_memory=False)
                    combined_out = pd.concat([old, out_df], ignore_index=True, sort=False)
                except Exception:
                    LOG.exception("Failed reading old outcomes; overwriting with new outcomes.")
                    combined_out = out_df
            else:
                combined_out = out_df
            # atomic write
            try:
                _atomic_write_csv(combined_out, OUTCOMES_CSV)
            except Exception:
                combined_out.to_csv(OUTCOMES_CSV, index=False)
            LOG.info("Appended %d outcome rows to %s", len(out_df), OUTCOMES_CSV)
            # write back updated recs
            try:
                _atomic_write_csv(recs, RECOMMENDATIONS_CSV)
            except Exception:
                recs.to_csv(RECOMMENDATIONS_CSV, index=False)
            LOG.info("Updated recommendations statuses in %s", RECOMMENDATIONS_CSV)
        else:
            LOG.info("No outcomes detected today.")
    except Exception:
        LOG.exception("daily_check_outcomes_csv failed.")
        raise

# -------------------------
# fetch_global_indices_csv
# -------------------------
def fetch_global_indices_csv(indices: list = None):
    """
    Fetch snapshot of global indices (best-effort via yfinance) and append to data/global_markets.csv.
    If yfinance fails, write a timestamped empty row (so system knows a fetch attempt occurred).
    """
    try:
        if indices is None:
            indices = ['^GSPC','^IXIC','^DJI','^FTSE','^N225','^HSI','^GDAXI','^FCHI','^NSEI','^BSESN']
        rows = []
        ts = datetime.now(timezone.utc).isoformat()
        for idx in indices:
            try:
                t = yf.Ticker(idx)
                hist = t.history(period="1d")
                if hist is not None and not hist.empty:
                    last = hist.iloc[-1]
                    price = float(last["Close"])
                    openp = float(last.get("Open", price))
                    change_pct = ((price - openp) / openp) * 100 if openp and openp != 0 else 0.0
                    rows.append({
                        "idx_symbol": idx,
                        "price": price,
                        "change_pct": round(change_pct, 3),
                        "fetched_at": ts
                    })
                else:
                    rows.append({"idx_symbol": idx, "price": None, "change_pct": None, "fetched_at": ts})
            except Exception:
                LOG.exception("Failed fetching index %s; writing empty row", idx)
                rows.append({"idx_symbol": idx, "price": None, "change_pct": None, "fetched_at": ts})
        df = pd.DataFrame(rows)
        # append to GLOBAL_MARKETS_CSV
        if GLOBAL_MARKETS_CSV.exists():
            try:
                old = pd.read_csv(GLOBAL_MARKETS_CSV, low_memory=False)
                combined = pd.concat([old, df], ignore_index=True, sort=False)
            except Exception:
                LOG.exception("Failed reading existing global_markets.csv; overwriting.")
                combined = df
        else:
            combined = df
        # atomic write
        try:
            _atomic_write_csv(combined, GLOBAL_MARKETS_CSV)
        except Exception:
            combined.to_csv(GLOBAL_MARKETS_CSV, index=False)
        LOG.info("Appended global indices snapshot (%d) to %s", len(df), GLOBAL_MARKETS_CSV)
    except Exception:
        LOG.exception("fetch_global_indices_csv failed.")
        raise

# -------------------------
# End of module
# -------------------------
