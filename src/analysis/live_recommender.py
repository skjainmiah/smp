# src/analysis/live_recommender.py
"""
Live recommender loader (Bollinger + RSI strategy)
- Robust repo-root / strategies dir detection
- Loads strategies from src/strategies (or fallbacks)
- Exposes generate_live_recommendations(min_rows=40) -> pandas.DataFrame
- Accepts strategy.signal(df) returns as: dict, pd.Series, list, int/float, str, or None
"""

from __future__ import annotations
import sys
from pathlib import Path
import importlib.util
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s - %(message)s")
LOG = logging.getLogger("live_recommender")

# ---------- repo/strategies discovery ----------
THIS = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "src").exists() or (p / "data").exists() or (p / ".git").exists() or (p / "README.md").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    # fallback
    if len(start.parents) >= 2:
        return start.parents[2]
    return start.parents[1]

REPO_ROOT = find_repo_root(THIS.parent)

# candidate strategy folders
CANDIDATE_STRATEGIES = [
    REPO_ROOT / "src" / "strategies",
    REPO_ROOT / "strategies",
    THIS.parent.parent / "strategies",
    REPO_ROOT / "src" / "analysis" / "strategies",
]

# ensure repo root in sys.path so imports like src.fetcher... work
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

STRATEGIES_DIR: Optional[Path] = None
for cand in CANDIDATE_STRATEGIES:
    if cand.exists() and cand.is_dir():
        STRATEGIES_DIR = cand
        break

if STRATEGIES_DIR is None:
    LOG.warning("Strategies dir not found. Tried: %s", [str(p) for p in CANDIDATE_STRATEGIES])
else:
    LOG.info("Using strategies dir: %s", STRATEGIES_DIR)

# ---------- dynamic loader ----------
def load_strategy_module(path: Path):
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            LOG.warning("Spec not created for %s", path)
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        LOG.exception("Failed loading strategy module %s", path)
        return None

def discover_strategies() -> List[Dict[str, Any]]:
    strategies = []
    if STRATEGIES_DIR is None:
        return strategies
    for f in sorted(STRATEGIES_DIR.glob("*.py")):
        if f.name.startswith("__"):
            continue
        mod = load_strategy_module(f)
        if mod is None:
            continue
        name = getattr(mod, "NAME", f.stem)
        if not hasattr(mod, "signal"):
            LOG.warning("Strategy %s does not expose 'signal(df)'; skipping", name)
            continue
        strategies.append({"name": name, "module": mod, "path": f})
        LOG.info("Loaded strategy %s from %s", name, f.name)
    if not strategies:
        LOG.warning("No strategies loaded. Ensure src/strategies has strategy files (e.g. bollinger_rsi.py)")
    return strategies

# cache
_STRATEGIES = discover_strategies()

# ---------- helpers ----------
def _read_eod_for_symbol(symbol: str, repo_root: Path = REPO_ROOT) -> Optional[pd.DataFrame]:
    path = repo_root / "data" / "standard" / "auto" / f"{symbol}.csv"
    if not path.exists():
        LOG.debug("EOD file not found for %s (%s)", symbol, path)
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
        return df
    except Exception:
        LOG.exception("Failed reading EOD for %s", symbol)
        return None

def _normalize_action(val: Any) -> Optional[str]:
    """
    Normalize various possible returns to canonical action string or None.
    Accepts:
      - dict with 'action' (string or numeric)
      - pandas Series (last value)
      - int/float (1,2,-1,-2)
      - str like "BUY","STRONG BUY"
    Returns one of: "STRONG BUY","BUY",None,"SELL","STRONG SELL"
    """
    if val is None:
        return None
    # dict case
    if isinstance(val, dict):
        a = val.get("action")
        return _normalize_action(a)
    # pandas Series / list / tuple -> take last element
    if isinstance(val, (pd.Series, list, tuple)):
        try:
            last = val.iloc[-1] if isinstance(val, pd.Series) else (val[-1] if len(val) else None)
            return _normalize_action(last)
        except Exception:
            return None
    # numeric
    if isinstance(val, (int, float)):
        try:
            v = int(val)
            if v >= 2:
                return "STRONG BUY"
            if v == 1:
                return "BUY"
            if v == -1:
                return "SELL"
            if v <= -2:
                return "STRONG SELL"
            return None
        except Exception:
            return None
    # string
    if isinstance(val, str):
        s = val.strip().upper()
        if s in ("STRONG BUY","STRONGBUY","STRONG_BUY","2"):
            return "STRONG BUY"
        if s in ("BUY","1"):
            return "BUY"
        if s in ("SELL","-1"):
            return "SELL"
        if s in ("STRONG SELL","STRONGSELL","STRONG_SELL","-2"):
            return "STRONG SELL"
        if s in ("HOLD","NONE","", "NO_ACTION"):
            return None
        # Try mapping common words
        if "BUY" in s and "STRONG" in s:
            return "STRONG BUY"
        if "BUY" in s:
            return "BUY"
        if "SELL" in s and "STRONG" in s:
            return "STRONG SELL"
        if "SELL" in s:
            return "SELL"
        return None
    # fallback
    return None

def _get_from_sig(sig: Any, key: str, default=None):
    """If sig is dict return value for key, else None"""
    if isinstance(sig, dict):
        return sig.get(key, default)
    return default

# ---------- main generator ----------
def generate_live_recommendations(min_rows: int = 40) -> pd.DataFrame:
    """
    Scan EOD standardized files (data/standard/auto/*.csv), run strategies on each,
    and return a DataFrame of actionable recommendations.
    """
    if not _STRATEGIES:
        LOG.warning("No strategies available to run.")
        return pd.DataFrame()

    eod_dir = REPO_ROOT / "data" / "standard" / "auto"
    if not eod_dir.exists():
        LOG.warning("EOD dir not found: %s", eod_dir)
        return pd.DataFrame()

    files = sorted([p for p in eod_dir.glob("*.csv")])
    LOG.info("Found %d EOD files to evaluate.", len(files))
    recs = []

    for f in files:
        symbol = f.stem
        try:
            df = _read_eod_for_symbol(symbol)
            if df is None or len(df) < min_rows:
                LOG.debug("Skipping %s — insufficient rows (%s)", symbol, len(df) if df is not None else 0)
                continue
            latest_row = df.iloc[-1]
            latest_close = float(latest_row.get("close", latest_row.get("Close", latest_row.get("adj_close", 0)) or 0))
            date_str = latest_row.get("date", datetime.utcnow().date().isoformat())
            for s in _STRATEGIES:
                mod = s["module"]
                try:
                    sig = mod.signal(df.copy())
                    # normalize action
                    action = _normalize_action(sig)
                    if action is None:
                        # No actionable signal from this strategy
                        continue

                    # Extract fields if present in dict
                    confidence = _get_from_sig(sig, "confidence") or _get_from_sig(sig, "conf") or None
                    support = _get_from_sig(sig, "support", None)
                    resistance = _get_from_sig(sig, "resistance", None)
                    target = _get_from_sig(sig, "target", None)
                    stoploss = _get_from_sig(sig, "stoploss", None)
                    rsi = _get_from_sig(sig, "rsi", None)
                    bb_upper = _get_from_sig(sig, "bb_upper", None)
                    bb_lower = _get_from_sig(sig, "bb_lower", None)
                    rationale = _get_from_sig(sig, "rationale", None)

                    # ensure numeric fields cast safely
                    def _safe_float(x):
                        try:
                            return float(x) if x is not None else None
                        except Exception:
                            return None

                    rec = {
                        "symbol": symbol,
                        "name": symbol,
                        "action": action,
                        "strategy": s["name"],
                        "confidence": _safe_float(confidence) if confidence is not None else None,
                        "latest_close": latest_close,
                        "support": _safe_float(support),
                        "resistance": _safe_float(resistance),
                        "target": _safe_float(target),
                        "stoploss": _safe_float(stoploss),
                        "rsi": _safe_float(rsi),
                        "bb_upper": _safe_float(bb_upper),
                        "bb_lower": _safe_float(bb_lower),
                        "rationale": rationale,
                        "date": pd.to_datetime(date_str, errors="coerce").strftime("%Y-%m-%d")
                    }

                    # category: simple heuristic for now - Options if strategy set indicates options (placeholder)
                    # In future, use instrument metadata to determine optionable symbols.
                    rec["category"] = "Stocks"

                    recs.append(rec)
                    LOG.info("Strategy %s signaled %s for %s", s["name"], action, symbol)
                except Exception:
                    LOG.exception("Strategy %s failed on %s", s["name"], symbol)
                    continue
        except Exception:
            LOG.exception("Failed evaluating symbol %s", symbol)
            continue

    if not recs:
        LOG.info("No actionable signals.")
        return pd.DataFrame()

    df_recs = pd.DataFrame(recs)

    # add buy_zone placeholder if missing
    if "buy_zone" not in df_recs.columns:
        df_recs["buy_zone"] = ""

    # canonical column order
    cols = [
        "symbol","name","action","category","strategy","confidence","latest_close",
        "support","resistance","target","stoploss","buy_zone","date",
        "rsi","bb_upper","bb_lower","rationale"
    ]
    for c in cols:
        if c not in df_recs.columns:
            df_recs[c] = None
    df_recs = df_recs[cols]
    return df_recs

# ---------- self-test ----------
if __name__ == "__main__":
    LOG.info("live_recommender self-test.")
    LOG.info("REPO_ROOT: %s", REPO_ROOT)
    LOG.info("STRATEGIES_DIR: %s", STRATEGIES_DIR if STRATEGIES_DIR is not None else "None")
    LOG.info("Discovered strategies: %s", [s["name"] for s in _STRATEGIES])
    df = generate_live_recommendations(min_rows=20)
    LOG.info("Recommendations returned: %d", 0 if df is None else len(df))
    if df is not None and not df.empty:
        print(df.head().to_string(index=False))
