# src/fundamentals/fundamentals_fetcher_csv.py
"""
CSV/JSON fundamentals fetcher
- Saves JSON per ticker to data/fundamentals/<TICKER>.json
- Saves summary CSV to data/fundamentals/fundamentals_summary.csv
- Evaluates a simple score -> WORST/BAD/BETTER/BEST
"""
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import yfinance as yf

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
FUND_DIR = PROJECT_ROOT / "data" / "fundamentals"
SUMMARY_CSV = FUND_DIR / "fundamentals_summary.csv"

LOG = logging.getLogger("fundamentals_fetcher")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")

FUND_DIR.mkdir(parents=True, exist_ok=True)

def _ns_ticker(t: str):
    return t if "." in t else f"{t}.NS"

def _safe_to_json_frame(obj):
    try:
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json(orient="split", date_format="iso"))
        return obj if obj is not None else {}
    except Exception:
        return {}

def fetch_fundamentals_single_csv(ticker: str) -> Dict[str, Any]:
    yf_t = _ns_ticker(ticker)
    tk = yf.Ticker(yf_t)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        LOG.exception("Failed tk.info for %s", ticker)
        info = {}
    payload = {
        "ticker": ticker,
        "fetched_at": datetime.utcnow().isoformat(),
        "info": info,
        "financials": _safe_to_json_frame(getattr(tk, "financials", {})),
        "quarterly_financials": _safe_to_json_frame(getattr(tk, "quarterly_financials", {})),
        "balance_sheet": _safe_to_json_frame(getattr(tk, "balance_sheet", {})),
        "quarterly_balance_sheet": _safe_to_json_frame(getattr(tk, "quarterly_balance_sheet", {})),
        "earnings": _safe_to_json_frame(getattr(tk, "earnings", {})),
        "quarterly_earnings": _safe_to_json_frame(getattr(tk, "quarterly_earnings", {})),
    }
    return payload

def fetch_and_store_fundamentals_csv(tickers: List[str]):
    rows = []
    for t in tickers:
        try:
            p = fetch_fundamentals_single_csv(t)
            out = FUND_DIR / f"{t}.json"
            with out.open("w", encoding="utf-8") as fh:
                json.dump(p, fh, default=str)
            info = p.get("info", {}) or {}
            def getf(k):
                v = info.get(k)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None
            row = {"ticker": t, "fetched_at": p["fetched_at"], "trailingPE": getf("trailingPE"), "forwardPE": getf("forwardPE"), "priceToBook": getf("priceToBook"), "bookValue": getf("bookValue"), "marketCap": getf("marketCap"), "returnOnEquity": getf("returnOnEquity"), "profitMargins": getf("profitMargins")}
            rows.append(row)
            LOG.info("Saved fundamentals for %s", t)
        except Exception:
            LOG.exception("Failed fundamentals for %s", t)
    if rows:
        df_new = pd.DataFrame(rows)
        if SUMMARY_CSV.exists():
            df_old = pd.read_csv(SUMMARY_CSV)
            df_old = df_old[~df_old["ticker"].isin(df_new["ticker"].tolist())]
            combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            combined = df_new
        combined.to_csv(SUMMARY_CSV, index=False)
        LOG.info("Updated fundamentals summary: %s", SUMMARY_CSV)

def load_fundamentals_summary(ticker: str):
    if not SUMMARY_CSV.exists():
        return {}
    df = pd.read_csv(SUMMARY_CSV)
    row = df[df["ticker"]==ticker]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def evaluate_fundamentals_snapshot_csv(ticker: str):
    jfile = FUND_DIR / f"{ticker}.json"
    info = {}
    qfin = {}
    if jfile.exists():
        try:
            payload = json.loads(jfile.read_text())
            info = payload.get("info", {}) or {}
            qfin = payload.get("quarterly_financials", {}) or {}
        except Exception:
            LOG.exception("Failed reading fundamentals json for %s", ticker)
    else:
        info = load_fundamentals_summary(ticker)
    def _extract(v):
        try:
            return None if pd.isna(v) else float(v)
        except Exception:
            return None
    pe = _extract(info.get("trailingPE") or info.get("forwardPE"))
    pb = _extract(info.get("priceToBook"))
    roe = _extract(info.get("returnOnEquity"))
    profit = _extract(info.get("profitMargins"))
    score = 0
    if pe is not None:
        if pe <= 15: score += 2
        elif pe <= 25: score += 1
        else: score -= 1
    if pb is not None:
        if pb <= 2: score += 2
        elif pb <= 3.5: score += 1
        else: score -= 1
    if roe is not None:
        if roe >= 0.18: score += 2
        elif roe >= 0.12: score += 1
        else: score -= 1
    if profit is not None:
        if profit >= 0.15: score += 2
        elif profit >= 0.08: score += 1
        else: score -= 1
    label = "WORST" if score < 0 else ("BAD" if score < 2 else ("BETTER" if score < 5 else "BEST"))
    return {"ticker": ticker, "label": label, "score": int(score), "metrics": {"pe": pe, "pb": pb, "roe": roe, "profit_margin": profit}}
