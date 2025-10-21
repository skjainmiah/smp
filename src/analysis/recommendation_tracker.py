# src/analysis/recommendation_tracker_csv.py
"""
CSV-only recommendation tracker (drop-in)
- data/recommendations.csv  (append)
- data/outcomes.csv
- data/global_markets.csv
- data/fundamentals/*.json + fundamentals_summary.csv  (created by fundamentals_fetcher_csv)
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
import yfinance as yf
from typing import Optional, List

LOG = logging.getLogger("recommendation_tracker")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FUND_DIR = DATA_DIR / "fundamentals"
RECS_CSV = DATA_DIR / "recommendations.csv"
OUTCOMES_CSV = DATA_DIR / "outcomes.csv"
GLOBAL_CSV = DATA_DIR / "global_markets.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# import fundamentals fetcher (CSV) if available
try:
    from src.fundamentals.fundamentals_fetcher_csv import fetch_and_store_fundamentals_csv, evaluate_fundamentals_snapshot_csv
except Exception:
    fetch_and_store_fundamentals_csv = None
    evaluate_fundamentals_snapshot_csv = None
    LOG.warning("fundamentals_fetcher_csv not available")

def _next_id(df):
    return int(df["rec_id"].max()) + 1 if (df is not None and not df.empty) else 1

def store_recommendations_csv(df: pd.DataFrame, export_enriched: bool=True) -> Optional[Path]:
    """
    Append recommendations DataFrame to data/recommendations.csv.
    Ensures required columns, adds rec_id and status fields.
    Fetches fundamentals for tickers (if fundamentals_fetcher_csv available)
    and writes an enriched CSV under results/.
    """
    if df is None or df.empty:
        LOG.info("No recs to store")
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RECS_CSV.exists():
        existing = pd.read_csv(RECS_CSV)
    else:
        existing = pd.DataFrame(columns=[
            "rec_id","symbol","name","action","category","strategies_triggered","latest_close",
            "support","resistance","target","stoploss","buy_zone","date","status","closed_at","days_to_target",
            "fundamentals_label","fundamentals_score"
        ])
    next_id = _next_id(existing)
    rows = []
    tickers = []
    for _, r in df.iterrows():
        category = r.get("category") if r.get("category") else "Stocks"
        rec = {
            "rec_id": next_id,
            "symbol": r.get("symbol"),
            "name": r.get("name", ""),
            "action": r.get("action"),
            "category": category,
            "strategies_triggered": r.get("strategies_triggered"),
            "latest_close": r.get("latest_close"),
            "support": r.get("support"),
            "resistance": r.get("resistance"),
            "target": r.get("target"),
            "stoploss": r.get("stoploss"),
            "buy_zone": r.get("buy_zone"),
            "date": r.get("date", datetime.now().strftime("%Y-%m-%d")),
            "status": "OPEN",
            "closed_at": None,
            "days_to_target": None,
            "fundamentals_label": None,
            "fundamentals_score": None
        }
        rows.append(rec)
        tickers.append(rec["symbol"])
        next_id += 1

    df_new = pd.DataFrame(rows)
    out = pd.concat([existing, df_new], ignore_index=True, sort=False)
    out.to_csv(RECS_CSV, index=False)
    LOG.info("Appended %d recommendations to %s", len(rows), RECS_CSV)

    # Fetch fundamentals for unique tickers and evaluate
    ticks = list(set([t for t in tickers if t]))
    if ticks and fetch_and_store_fundamentals_csv:
        try:
            fetch_and_store_fundamentals_csv(ticks)
            # update labels/scores
            for t in ticks:
                ev = evaluate_fundamentals_snapshot_csv(t)
                out.loc[out["symbol"]==t, "fundamentals_label"] = ev.get("label")
                out.loc[out["symbol"]==t, "fundamentals_score"] = ev.get("score")
            out.to_csv(RECS_CSV, index=False)
        except Exception:
            LOG.exception("Failed fetch/evaluate fundamentals")

    # export enriched for today's date
    if export_enriched and not df_new.empty:
        day = df_new.iloc[0]["date"]
        path = RESULTS_DIR / f"live_recommendations_enriched_{day.replace('-','')}.csv"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_day = out[out["date"]==day].copy()
        out_day.to_csv(path, index=False)
        LOG.info("Exported enriched recommendations: %s", path)
        return path
    return None

def daily_check_outcomes_csv(eod_dir: str = str(PROJECT_ROOT / "data" / "standard" / "auto")):
    """
    Check open recommendations against the latest EOD prices and update statuses (TARGET_HIT / STOP_HIT)
    Append outcome events to data/outcomes.csv
    """
    if not RECS_CSV.exists():
        LOG.info("No recommendations file found")
        return
    df = pd.read_csv(RECS_CSV)
    open_recs = df[df["status"]=="OPEN"].copy()
    if open_recs.empty:
        LOG.info("No open recs to check")
        return
    outcomes = []
    for _, rec in open_recs.iterrows():
        sym = rec["symbol"]
        eod_file = Path(eod_dir) / f"{sym}.csv"
        if not eod_file.exists():
            LOG.debug("EOD file not found for %s", sym)
            continue
        try:
            dfe = pd.read_csv(eod_file, parse_dates=["date"]).sort_values("date")
            last = dfe.iloc[-1]
            price = float(last["close"])
            date_str = pd.to_datetime(last["date"]).strftime("%Y-%m-%d")
            outcomes.append({"rec_id": rec["rec_id"], "date": date_str, "price": price, "note": ""})
            status = None
            if pd.notna(rec.get("target")) and price >= float(rec["target"]):
                status = "TARGET_HIT"
            elif pd.notna(rec.get("stoploss")) and price <= float(rec["stoploss"]):
                status = "STOP_HIT"
            if status:
                df.loc[df["rec_id"]==rec["rec_id"], "status"] = status
                df.loc[df["rec_id"]==rec["rec_id"], "closed_at"] = date_str
                days = (pd.to_datetime(date_str) - pd.to_datetime(rec["date"])).days
                df.loc[df["rec_id"]==rec["rec_id"], "days_to_target"] = int(days)
        except Exception:
            LOG.exception("Failed checking outcomes for %s", sym)
    if outcomes:
        outdf = pd.DataFrame(outcomes)
        if OUTCOMES_CSV.exists():
            prev = pd.read_csv(OUTCOMES_CSV)
            combined = pd.concat([prev, outdf], ignore_index=True)
        else:
            combined = outdf
        combined.to_csv(OUTCOMES_CSV, index=False)
        LOG.info("Appended %d outcome rows to %s", len(outcomes), OUTCOMES_CSV)
    df.to_csv(RECS_CSV, index=False)
    LOG.info("Updated recommendations statuses")

def fetch_global_indices_csv(indices: Optional[List[str]] = None):
    """
    Fetch basic snapshot for major global indices and append to data/global_markets.csv
    """
    if indices is None:
        indices = ['^GSPC','^IXIC','^DJI','^FTSE','^N225','^HSI','^GDAXI','^FCHI','^NSEI','^BSESN']
    rows = []
    for sym in indices:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="1d")
            if hist is None or hist.empty:
                continue
            last = hist.reset_index().iloc[-1]
            price = float(last["Close"])
            openp = float(last.get("Open", price))
            change = price - openp
            rows.append({"symbol": sym, "price": price, "change": change, "fetched_at": datetime.utcnow().isoformat()})
        except Exception:
            LOG.exception("Failed fetching index %s", sym)
    if rows:
        df = pd.DataFrame(rows)
        if GLOBAL_CSV.exists():
            prev = pd.read_csv(GLOBAL_CSV)
            comb = pd.concat([prev, df], ignore_index=True)
        else:
            comb = df
        comb.to_csv(GLOBAL_CSV, index=False)
        LOG.info("Saved global indices snapshot to %s", GLOBAL_CSV)
