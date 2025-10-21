# src/orchestrator.py
"""
Orchestrator (CSV-only) — DAILY EOD + daily intraday 5-minute fetch at 05:00 IST.

Behavior:
- Runs once daily at 05:00 IST (skips holidays)
  - Fetch incremental EOD daily OHLC data
  - Fetch intraday 5-minute data for recent N days (default 1) — enough to capture the previous trading day's 5m bars
  - Consolidate intraday monthly files into data/intraday_5m_daily/<SYMBOL>.csv (UTC-normalized)
  - Run recommender and store CSV recommendations
  - Run outcome checks and global snapshot (CSV)
- No continuous intraday scheduler; intraday data is refreshed once daily.
- Use env var INTRADAY_MAX_DAYS_FETCH to change how many recent days' 5m bars to fetch (default=1).
Usage:
    python -m src.orchestrator --once                # run everything once (fetches intraday for last N days)
    python -m src.orchestrator                       # start scheduler (daily 05:00 IST)
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime, time as dtime, timedelta
import pytz
import time as _time
import os
import pandas as pd
import importlib
from typing import Optional

# -------------------------
# Robust repo root detection & Logging (LOG available early)
# -------------------------
THIS = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    """
    Climb up directories looking for repo root heuristics:
    presence of 'src' dir, 'data' dir, '.git' or 'README.md'.
    Fallback: two levels up from start.
    """
    p = start
    for _ in range(10):
        if (p / "src").exists() or (p / "data").exists() or (p / ".git").exists() or (p / "README.md").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    # fallback: two levels up
    if len(start.parents) >= 2:
        return start.parents[2]
    return start.parents[1]

# compute PROJECT_ROOT once
PROJECT_ROOT = find_repo_root(THIS.parent)

# ensure PROJECT_ROOT on sys.path first (so imports like 'src.fetcher...' work)
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# initialize logging early so we can log during imports/setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")
LOG = logging.getLogger("orchestrator")
LOG.debug("Computed PROJECT_ROOT: %s", PROJECT_ROOT)
LOG.debug("sys.path[0:3]: %s", sys.path[:3])

# -------------------------
# Config / Paths
# -------------------------
TZ = pytz.timezone("Asia/Kolkata")
MASTER_CSV = PROJECT_ROOT / "data" / "EQUITY_L.csv"
EOD_STANDARD_DIR = PROJECT_ROOT / "data" / "standard" / "auto"
INTRADAY_DIR = PROJECT_ROOT / "data" / "intraday_5m"
INTRADAY_DAILY_DIR = PROJECT_ROOT / "data" / "intraday_5m_daily"
RESULTS_DIR = PROJECT_ROOT / "results"
META_DIR = PROJECT_ROOT / "data" / "meta"

INTRADAY_RETENTION_DAYS = int(os.getenv("INTRADAY_RETENTION_DAYS", "90"))
INTRADAY_MAX_DAYS_FETCH = int(os.getenv("INTRADAY_MAX_DAYS_FETCH", "1"))  # default: fetch 1 day of 5m bars
INTRADAY_CONSOLIDATE_KEEP_MONTHS = int(os.getenv("INTRADAY_CONSOLIDATE_KEEP_MONTHS", "3"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTRADAY_DAILY_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
EOD_STANDARD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Defensive imports helper
# -------------------------
def try_import(module_names):
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            LOG.info("Imported module '%s' successfully.", name)
            return mod
        except Exception as e:
            LOG.debug("Import failed for %s: %s", name, e)
    return None

# fetchers
eod_mod = try_import(["src.fetcher.yfinance_fetch", "src.fetcher.yfinance_fetcher", "fetcher.yfinance_fetch", "yfinance_fetch"])
intraday_mod = try_import(["src.fetcher.yfinance_fetch_5m", "fetcher.yfinance_fetch_5m", "yfinance_fetch_5m", "src.fetcher.yfinance_fetch_5m"])

# recommender
recommender_mod = try_import(["src.analysis.live_recommender", "src.live_recommender", "analysis.live_recommender"])

# CSV tracker
csv_tracker_mod = try_import(["src.analysis.recommendation_tracker_csv", "analysis.recommendation_tracker_csv", "recommendation_tracker_csv"])

# holidays module (optional)
holidays_mod = try_import(["src.fetcher.holidays", "src.holidays", "holidays", "src.fetcher.scheduler"])

# functions safely resolved
fetch_all_from_master = getattr(eod_mod, "fetch_all_from_master", None) if eod_mod else None
fetch_all_5m = getattr(intraday_mod, "fetch_all_5m", None) if intraday_mod else None
generate_live_recommendations = getattr(recommender_mod, "generate_live_recommendations", None) if recommender_mod else None

# CSV tracker functions
store_recommendations_csv = getattr(csv_tracker_mod, "store_recommendations_csv", None) if csv_tracker_mod else None
daily_check_outcomes_csv = getattr(csv_tracker_mod, "daily_check_outcomes_csv", None) if csv_tracker_mod else None
fetch_global_indices_csv = getattr(csv_tracker_mod, "fetch_global_indices_csv", None) if csv_tracker_mod else None

# holidays
is_trading_day = getattr(holidays_mod, "is_trading_day", lambda d: True) if holidays_mod else (lambda d: True)

# -------------------------
# Utilities
# -------------------------
def ensure_dirs():
    INTRADAY_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    EOD_STANDARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

def consolidate_intraday_for_symbol(symbol: str, keep_months: int = INTRADAY_CONSOLIDATE_KEEP_MONTHS):
    """
    Read monthly files from data/intraday_5m/<symbol>/*.csv, concat, dedupe by datetime,
    normalize to UTC tz-aware datetimes, and write to data/intraday_5m_daily/<symbol>.csv.
    """
    src_dir = INTRADAY_DIR / symbol
    out_file = INTRADAY_DAILY_DIR / f"{symbol}.csv"
    if not src_dir.exists():
        LOG.debug("No intraday folder for %s", symbol)
        return
    files = sorted(src_dir.glob("*.csv"))
    if not files:
        LOG.debug("No monthly intraday files for %s", symbol)
        return
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["datetime"], low_memory=False)
            parts.append(df)
        except Exception as e:
            LOG.warning("Skipping intraday file %s for %s: %s", f, symbol, e)
            continue
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True)
    if "datetime" not in combined.columns:
        LOG.warning("Combined intraday for %s has no datetime column; skipping", symbol)
        return
    # normalize datetimes to UTC tz-aware
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    # keep last N days
    cutoff = datetime.now(pytz.UTC) - timedelta(days=INTRADAY_RETENTION_DAYS)
    combined = combined[combined["datetime"] >= cutoff]
    # write consolidated file with ISO timestamps
    combined.to_csv(out_file, index=False)
    LOG.info("Consolidated intraday for %s → %s rows", symbol, len(combined))

def consolidate_all_intraday(limit: int = None):
    if not INTRADAY_DIR.exists():
        LOG.info("Intraday dir does not exist; skipping consolidation.")
        return
    symbols = [p.name for p in INTRADAY_DIR.iterdir() if p.is_dir()]
    if limit:
        symbols = symbols[:limit]
    LOG.info("Consolidating intraday for %d symbols...", len(symbols))
    for s in symbols:
        try:
            consolidate_intraday_for_symbol(s)
        except Exception:
            LOG.exception("Failed consolidating intraday for %s", s)

# -------------------------
# Core pipeline steps
# -------------------------
def run_5m_fetch(max_days: Optional[int] = None, limit: Optional[int] = None):
    """
    Run 5m fetcher once for recent days. Pass max_days (int) to the fetcher when supported.
    """
    if fetch_all_5m is None:
        LOG.warning("5m fetcher not available; skipping.")
        return
    try:
        LOG.info("Running 5m fetch (max_days=%s, limit=%s)...", max_days, limit)
        # try to call with max_days if fetcher supports it
        try:
            if max_days is not None:
                fetch_all_5m(max_days=max_days, limit=limit)
            else:
                fetch_all_5m(limit=limit)
        except TypeError:
            # older signature may accept "days" or "limit" only
            try:
                fetch_all_5m(limit=limit)
            except Exception:
                fetch_all_5m()
    except Exception:
        LOG.exception("5m fetch failed.")

def run_eod_fetch(limit: Optional[int] = None):
    if fetch_all_from_master is None:
        LOG.warning("EOD fetcher not available; skipping.")
        return
    try:
        LOG.info("Running EOD fetch (incremental)...")
        try:
            fetch_all_from_master(master_csv=MASTER_CSV, start="2000-01-01", full=False, limit=limit)
        except TypeError:
            fetch_all_from_master()
    except Exception:
        LOG.exception("EOD fetch failed.")

def run_recommender_and_store(storage: str = "csv", min_rows: int = 40):
    if generate_live_recommendations is None:
        LOG.error("Recommender function not available; skipping.")
        return None
    LOG.info("Running live recommender...")
    try:
        df_recs = generate_live_recommendations(min_rows=min_rows)
    except TypeError:
        df_recs = generate_live_recommendations()
    except Exception:
        LOG.exception("Recommender failed.")
        return None
    if df_recs is None or df_recs.empty:
        LOG.info("No recommendations returned by recommender.")
        return None
    if storage == "csv":
        if store_recommendations_csv is None:
            LOG.error("CSV tracker not available. Cannot store recommendations.")
            return None
        try:
            out_path = store_recommendations_csv(df_recs)
            LOG.info("Stored recommendations (CSV) → %s", out_path)
            return out_path
        except Exception:
            LOG.exception("Failed storing recommendations (CSV).")
            return None
    else:
        LOG.error("Unknown storage mode: %s", storage)
        return None

def run_daily_outcome_check(storage: str = "csv"):
    if storage == "csv":
        if daily_check_outcomes_csv is None:
            LOG.warning("Daily outcomes function not available; skipping.")
            return
        try:
            daily_check_outcomes_csv()
        except Exception:
            LOG.exception("CSV daily outcome check failed.")
    else:
        LOG.error("Unknown storage mode: %s", storage)

def run_global_snapshot(storage: str = "csv"):
    if storage == "csv":
        if fetch_global_indices_csv is None:
            LOG.warning("CSV global indices function not available; skipping.")
            return
        try:
            fetch_global_indices_csv()
        except Exception:
            LOG.exception("CSV global snapshot failed.")
    else:
        LOG.error("Unknown storage mode: %s", storage)

# -------------------------
# Pipeline compositions
# -------------------------
def daily_pipeline(max_days_intraday: Optional[int] = None, limit: Optional[int] = None):
    """
    Daily pipeline (to be run at 05:00 IST):
      1) Fetch intraday 5m for recent N days (default 1)
      2) Consolidate intraday monthly files into per-symbol file
      3) Fetch EOD incremental
      4) Run recommender and store to CSV
      5) Run outcome check
      6) Global indices snapshot
    """
    LOG.info("Starting daily pipeline (intraday_days=%s, limit=%s)", max_days_intraday, limit)
    try:
        run_5m_fetch(max_days=max_days_intraday, limit=limit)
    except Exception:
        LOG.exception("5m fetch failed in daily pipeline.")
    try:
        consolidate_all_intraday(limit=limit)
    except Exception:
        LOG.exception("Intraday consolidation failed in daily pipeline.")
    try:
        run_eod_fetch(limit=limit)
    except Exception:
        LOG.exception("EOD fetch failed in daily pipeline.")
    try:
        run_recommender_and_store(storage="csv", min_rows=40)
    except Exception:
        LOG.exception("Recommender/store failed in daily pipeline.")
    try:
        run_daily_outcome_check(storage="csv")
    except Exception:
        LOG.exception("Daily outcome check failed in daily pipeline.")
    try:
        run_global_snapshot(storage="csv")
    except Exception:
        LOG.exception("Global snapshot failed in daily pipeline.")
    LOG.info("Daily pipeline finished.")

# -------------------------
# Scheduler: daily 05:00 IST only (no frequent intraday jobs)
# -------------------------
def start_scheduler(max_days_intraday: int = INTRADAY_MAX_DAYS_FETCH, limit: Optional[int] = None):
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception:
        LOG.exception("APScheduler not installed. Please: pip install apscheduler")
        raise

    sched = BackgroundScheduler(timezone=TZ)

    def job_daily():
        today = datetime.now(TZ).date()
        LOG.info("Scheduled daily job triggered for %s", today.isoformat())
        if not is_trading_day(today):
            LOG.info("Today %s is a holiday or weekend - skipping daily pipeline.", today.isoformat())
            return
        LOG.info("Scheduled daily pipeline running at %s", datetime.now(TZ).isoformat())
        try:
            daily_pipeline(max_days_intraday=max_days_intraday, limit=limit)
        except Exception:
            LOG.exception("Scheduled daily job failed.")

    # schedule daily at 05:00 IST
    sched.add_job(job_daily, trigger=CronTrigger(hour=5, minute=0), id="daily_full_pipeline")

    LOG.info("Starting scheduler: daily pipeline at 05:00 IST (intraday fetch once per day).")
    sched.start()
    try:
        while True:
            _time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        LOG.info("Shutting down scheduler...")
        sched.shutdown()

# -------------------------
# CLI
# -------------------------
def run_once(max_days_intraday: int = INTRADAY_MAX_DAYS_FETCH, limit: Optional[int] = None):
    ensure_dirs()
    LOG.info("Running pipeline once (daily_pipeline).")
    daily_pipeline(max_days_intraday=max_days_intraday, limit=limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator: daily fetch EOD + 5m and generate/store recommendations")
    parser.add_argument("--once", action="store_true", help="Run the pipeline once and exit")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols (for testing)")
    parser.add_argument("--max-days", type=int, default=INTRADAY_MAX_DAYS_FETCH, help="Recent days of intraday 5m to fetch (default from env/1)")
    args = parser.parse_args()

    if args.once:
        run_once(max_days_intraday=args.max_days, limit=args.limit)
    else:
        start_scheduler(max_days_intraday=args.max_days, limit=args.limit)
