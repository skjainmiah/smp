# src/api/app.py
from __future__ import annotations

import os
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------
# Repo root + paths (robust)
# ---------------------------
THIS = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / "src").exists() or (p / "data").exists() or (p / ".git").exists() or (p / "README.md").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.parents[2] if len(start.parents) >= 3 else start.parents[1]

REPO_ROOT = find_repo_root(THIS.parent)
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
EOD_DIR = DATA_DIR / "standard" / "auto"
RECOMMENDATIONS_CSV = DATA_DIR / "recommendations.csv"
OUTCOMES_CSV = DATA_DIR / "outcomes.csv"
GLOBAL_MARKETS_CSV = DATA_DIR / "global_markets.csv"
OPTIONABLE_CSV = DATA_DIR / "meta" / "optionable_symbols.csv"  # optional (symbol column)

# ---------------------------
# App + CORS
# ---------------------------
app = FastAPI(title="Stock Recommendations API", version="1.0.0")

# CORS: set FRONTEND_ORIGINS="https://yourboltapp.bolt.new,https://yourdomain.com"
origins_env = os.getenv("FRONTEND_ORIGINS", "*")
allowed_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE","OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------
def _today_str():
    return datetime.now().strftime("%Y%m%d")

def _list_daily_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("live_recommendations_*.csv"))

def _latest_daily_file_for(date: Optional[str] = None) -> Optional[Path]:
    """
    If date is provided (YYYYMMDD), try that; else pick latest by name.
    """
    if not RESULTS_DIR.exists():
        return None
    if date:
        p = RESULTS_DIR / f"live_recommendations_{date}.csv"
        if p.exists():
            return p
    files = _list_daily_files()
    return files[-1] if files else None

def _read_df_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _action_order_key(action: str) -> int:
    # for sorting by action
    mapping = {
        "STRONG BUY": 4,
        "BUY": 3,
        "HOLD": 2,
        "SELL": 1,
        "STRONG SELL": 0
    }
    return mapping.get(str(action).upper(), -1)

def _load_optionable_symbols() -> Optional[pd.Series]:
    if OPTIONABLE_CSV.exists():
        try:
            df = pd.read_csv(OPTIONABLE_CSV, dtype=str)
            col = next((c for c in df.columns if c.lower() == "symbol"), df.columns[0])
            return df[col].dropna().str.upper().str.strip()
        except Exception:
            return None
    return None

def _filter_category(df: pd.DataFrame, category: Optional[str]) -> pd.DataFrame:
    """
    Category rules:
      - "options": show only symbols present in optionable list (if OPTIONABLE_CSV exists),
                   otherwise return empty.
      - "stocks" or None: return those NOT in optionable OR all if no file exists.
    """
    if category is None:
        return df
    cat = category.strip().lower()
    optionable = _load_optionable_symbols()
    if cat == "options":
        if optionable is None:
            # if we don't know optionables, be conservative
            return df.iloc[0:0]
        return df[df["symbol"].astype(str).str.upper().isin(optionable)]
    if cat == "stocks":
        if optionable is None:
            return df
        return df[~df["symbol"].astype(str).str.upper().isin(optionable)]
    return df

def _apply_filters(
    df: pd.DataFrame,
    category: Optional[str],
    action: Optional[str],
    min_conf: Optional[float],
    q: Optional[str]
) -> pd.DataFrame:
    if category:
        df = _filter_category(df, category)
    if action:
        acts = [a.strip().upper() for a in action.split(",")]
        df = df[df["action"].astype(str).str.upper().isin(acts)]
    if min_conf is not None:
        # confidence may be 0..1 or 0..100 depending on generation; normalize if >1
        c = df["confidence"]
        if c.max(skipna=True) and c.max(skipna=True) > 1.5:
            df = df[c >= min_conf * 100.0]
        else:
            df = df[c >= min_conf]
    if q:
        ql = q.strip().lower()
        df = df[df.apply(lambda r: ql in str(r.get("symbol","")).lower() or ql in str(r.get("name","")).lower(), axis=1)]
    return df

def _apply_sort(df: pd.DataFrame, sort: Optional[str]) -> pd.DataFrame:
    if sort is None:
        return df.sort_values("date", ascending=False)
    s = sort.lower()
    if s == "confidence":
        return df.sort_values(["confidence","date"], ascending=[False, False], na_position="last")
    if s == "action":
        return df.assign(_ord=df["action"].apply(_action_order_key)).sort_values(["_ord","date"], ascending=[False, False]).drop(columns=["_ord"])
    if s == "latest":
        return df.sort_values("date", ascending=False)
    return df

def _paginate(df: pd.DataFrame, page: int, limit: int) -> pd.DataFrame:
    start = max(0, (page - 1) * limit)
    end = start + limit
    return df.iloc[start:end]

def _read_latest_recommendation_for_symbol(sym: str) -> Optional[dict]:
    if RECOMMENDATIONS_CSV.exists():
        all_df = pd.read_csv(RECOMMENDATIONS_CSV, low_memory=False)
        sub = all_df[all_df["symbol"].astype(str).str.upper() == sym]
        if not sub.empty:
            sub = sub.sort_values("date", ascending=False)
            return json.loads(sub.head(1).to_json(orient="records"))[0]
    # fallback to latest daily file
    latest_daily = _latest_daily_file_for()
    if latest_daily:
        df = pd.read_csv(latest_daily, low_memory=False)
        sub = df[df["symbol"].astype(str).str.upper() == sym]
        if not sub.empty:
            sub = sub.sort_values("date", ascending=False)
            return json.loads(sub.head(1).to_json(orient="records"))[0]
    return None

# ---------------------------
# Security helpers (admin)
# ---------------------------
def require_admin(x_admin_token: Optional[str] = Header(None)):
    admin_token = os.getenv("ADMIN_TOKEN", None)
    if not admin_token:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN not configured on server")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return True

# ---------------------------
# API: health & version
# ---------------------------
@app.get("/api/health")
def health():
    latest_file = _latest_daily_file_for()
    return {"status": "ok", "latest_file": str(latest_file) if latest_file else None}

@app.get("/api/version")
def version():
    return {"app": "stock-recs-api", "version": app.version}

# ---------------------------
# API: today's (or latest) recommendations
# ---------------------------
@app.get("/api/recommendations/today")
def get_today_recommendations(date: Optional[str] = Query(None, description="YYYYMMDD; if omitted uses latest")):
    p = _latest_daily_file_for(date or _today_str()) or _latest_daily_file_for()
    if not p:
        return JSONResponse(status_code=404, content={"error": "no daily results available"})
    df = _read_df_csv(p)
    return JSONResponse(content=json.loads(df.to_json(orient="records", date_format="iso")))

# ---------------------------
# API: recommendations (paginated + filters)
# ---------------------------
@app.get("/api/recommendations")
def get_recommendations(
    category: Optional[str] = Query(None, description="stocks|options"),
    sort: Optional[str] = Query("latest", description="confidence|action|latest"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    action: Optional[str] = Query(None, description="comma-separated: BUY,SELL,STRONG BUY,STRONG SELL"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    q: Optional[str] = Query(None, description="search in symbol/name"),
    date: Optional[str] = Query(None, description="YYYYMMDD: if set, use that daily file; else latest")
):
    # Use specific date file or latest daily file
    daily = _latest_daily_file_for(date) or _latest_daily_file_for()
    if not daily:
        return JSONResponse(content={"total": 0, "page": page, "limit": limit, "results": []})
    df = _read_df_csv(daily)

    # Normalize essential columns
    for c in ["symbol", "action", "confidence", "date"]:
        if c not in df.columns:
            df[c] = None

    df = _apply_filters(df, category, action, min_confidence, q)
    total = len(df)
    if total == 0:
        return {"total": 0, "page": page, "limit": limit, "results": []}

    df = _apply_sort(df, sort)
    page_df = _paginate(df, page, limit)
    results = json.loads(page_df.to_json(orient="records", date_format="iso"))
    return {"total": total, "page": page, "pages": math.ceil(total/limit), "limit": limit, "results": results}

# ---------------------------
# API: single stock detail
# ---------------------------
@app.get("/api/stocks/{symbol}")
def get_stock(symbol: str):
    sym = symbol.strip().upper()

    # latest rec (from cumulative or latest daily)
    latest_rec = _read_latest_recommendation_for_symbol(sym)

    # history (up to 365 days)
    eod_file = EOD_DIR / f"{sym}.csv"
    history: List[dict] = []
    if eod_file.exists():
        try:
            hist_df = pd.read_csv(eod_file, parse_dates=["date"], low_memory=False)
            hist_df = hist_df.tail(365)
            # ensure keys exist with standard names
            ren = {}
            if "Open" in hist_df.columns: ren["Open"] = "open"
            if "High" in hist_df.columns: ren["High"] = "high"
            if "Low" in hist_df.columns: ren["Low"] = "low"
            if "Close" in hist_df.columns: ren["Close"] = "close"
            if "Adj Close" in hist_df.columns: ren["Adj Close"] = "adj_close"
            hist_df = hist_df.rename(columns=ren)
            history = json.loads(hist_df.to_json(orient="records", date_format="iso"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read EOD for {sym}: {e}")

    # outcomes for this symbol
    outcomes: List[dict] = []
    if OUTCOMES_CSV.exists():
        try:
            out_df = pd.read_csv(OUTCOMES_CSV, low_memory=False)
            o = out_df[out_df["symbol"].astype(str).str.upper() == sym]
            o = o.sort_values("event_date", ascending=False)
            outcomes = json.loads(o.to_json(orient="records", date_format="iso"))
        except Exception:
            outcomes = []

    # fundamentals (from latest_rec if present)
    fundamentals = None
    if latest_rec and "fundamentals_details" in latest_rec and isinstance(latest_rec["fundamentals_details"], str):
        try:
            fundamentals = json.loads(latest_rec["fundamentals_details"])
        except Exception:
            fundamentals = None

    return {
        "symbol": sym,
        "recommendation": latest_rec,
        "history": history,
        "outcomes": outcomes,
        "fundamentals": fundamentals,
    }

# ---------------------------
# API: global indices latest snapshot (one per index)
# ---------------------------
@app.get("/api/indices")
def indices_latest():
    if not GLOBAL_MARKETS_CSV.exists():
        return []
    df = pd.read_csv(GLOBAL_MARKETS_CSV, low_memory=False)
    if df.empty:
        return []
    # keep last row per idx_symbol
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df = df.sort_values("fetched_at").drop_duplicates(subset=["idx_symbol"], keep="last")
    return json.loads(df.to_json(orient="records", date_format="iso"))

# ---------------------------
# Admin: upload CSV (replaces/merges today's)
# ---------------------------
def _require_admin(x_admin_token: Optional[str] = Header(None)):
    admin_token = os.getenv("ADMIN_TOKEN", None)
    if not admin_token:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN not configured on server")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return True

@app.post("/api/admin/upload_recs")
async def admin_upload_recs(file: UploadFile = File(...), _: bool = Depends(_require_admin)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    today = datetime.now().strftime("%Y%m%d")
    out = RESULTS_DIR / f"live_recommendations_{today}.csv"
    df.to_csv(out, index=False)

    # append/upsert into cumulative
    if RECOMMENDATIONS_CSV.exists():
        try:
            existing = pd.read_csv(RECOMMENDATIONS_CSV, low_memory=False)
            combined = pd.concat([existing, df], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["symbol","date","action"], keep="last")
        except Exception:
            combined = df
    else:
        combined = df
    combined.to_csv(RECOMMENDATIONS_CSV, index=False)
    return {"status": "ok", "rows": len(df)}

# ---------------------------
# Ingest JSON (orchestrator → API)
# ---------------------------
@app.post("/api/ingest")
async def ingest_recommendations(
    payload: List[Dict],
    x_admin_token: Optional[str] = Header(None)
):
    # Protect with same ADMIN_TOKEN
    admin_token = os.getenv("ADMIN_TOKEN", None)
    if not admin_token or x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not isinstance(payload, list) or len(payload) == 0:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty list of recommendation objects.")

    df = pd.DataFrame(payload)
    # write to today's file
    today = datetime.now().strftime("%Y%m%d")
    out = RESULTS_DIR / f"live_recommendations_{today}.csv"
    df.to_csv(out, index=False)
    # merge into cumulative
    if RECOMMENDATIONS_CSV.exists():
        try:
            existing = pd.read_csv(RECOMMENDATIONS_CSV, low_memory=False)
            combined = pd.concat([existing, df], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["symbol","date","action"], keep="last")
        except Exception:
            combined = df
    else:
        combined = df
    combined.to_csv(RECOMMENDATIONS_CSV, index=False)
    return {"status": "ok", "rows": len(df)}

# ---------------------------
# local run
# ---------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=port, reload=False)
