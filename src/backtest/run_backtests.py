# src/backtest/run_backtests.py
"""
Run backtests for all standardized symbol CSVs using strategies in chartink_strategies.ALL_STRATEGIES.

Outputs:
 - CSV summary: results/backtest_summary.csv
 - Per-backtest trades JSON: results/trades/<strategy>/<SYMBOL>.json
 - (Optional) Persist results to Postgres when --to-db is passed and DB is reachable.

Usage examples:
    python run_backtests.py                    # run all strategies on all CSVs (data/standard/auto)
    python run_backtests.py --input data/standard/multibagger --strategies short_term_breakout,intraday_best_buy --limit 50
    python run_backtests.py --to-db
"""

import argparse
from pathlib import Path
import pandas as pd
import json
import logging
from tqdm import tqdm
import traceback
from datetime import datetime

# import engine and strategies
from src.backtest.engine import run_backtest, summarize_backtest
from src.backtest.strategies.chartink_strategies import ALL_STRATEGIES

# optional DB persistence (uses your scaffold's persistence.pg)
try:
    from src.persistence.pg import get_engine, init_schema
    HAS_DB = True
except Exception:
    HAS_DB = False

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
LOGGER = logging.getLogger("run_backtests")

RESULTS_DIR = Path("results")
TRADES_DIR = RESULTS_DIR / "trades"
SUMMARY_FILE = RESULTS_DIR / "backtest_summary.csv"

# ensure folders
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

def _write_trades(trades: list, strategy_name: str, symbol: str):
    out_dir = TRADES_DIR / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trades, f, default=str, indent=2)

def _persist_to_db(engine, strategy_name: str, symbol: str, start_date: str, end_date: str, trades: list, metrics: dict):
    """
    Simple persistence: insert into strategies and backtests table (if they exist).
    This uses a very small safe approach:
      - Insert strategy into strategies table (if not exists) and get id
      - Insert a row into backtests with trades JSON and metrics JSON
    """
    try:
        with engine.connect() as conn:
            # upsert strategy
            res = conn.execute(
                "SELECT id FROM strategies WHERE name = :name",
                {"name": strategy_name}
            )
            row = res.fetchone()
            if row:
                strategy_id = row[0]
            else:
                r2 = conn.execute(
                    "INSERT INTO strategies (name, source, description) VALUES (:n,:s,:d) RETURNING id",
                    {"n": strategy_name, "s": "chartink_converted", "d": ""}
                )
                strategy_id = r2.fetchone()[0]

            # insert backtest
            conn.execute(
                """
                INSERT INTO backtests (strategy_id, symbol, start_date, end_date, trades, metrics)
                VALUES (:strategy_id, :symbol, :start_date, :end_date, :trades::jsonb, :metrics::jsonb)
                """,
                {
                    "strategy_id": int(strategy_id),
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "trades": json.dumps(trades, default=str),
                    "metrics": json.dumps(metrics, default=str)
                }
            )
            conn.commit()
    except Exception:
        LOGGER.exception("DB persist failed for %s - %s", strategy_name, symbol)

def run_all(
    input_dir: Path,
    strategies: dict,
    limit: int = None,
    to_db: bool = False,
    initial_capital: float = 100000.0,
    position_size_pct: float = 0.1,
    commission: float = 0.0005,
    slippage: float = 0.0005
):
    # get CSV list
    csvs = sorted(Path(input_dir).glob("*.csv"))
    if limit:
        csvs = csvs[:limit]

    LOGGER.info("Found %d csv files in %s", len(csvs), input_dir)
    strategy_items = list(strategies.items())

    # load DB engine if requested
    engine = None
    if to_db:
        if not HAS_DB:
            LOGGER.warning("DB persistence requested but persistence.pg not available. Skipping DB writes.")
            to_db = False
        else:
            try:
                engine = get_engine()
                init_schema(engine)
                LOGGER.info("DB engine ready")
            except Exception:
                LOGGER.exception("Failed to init DB engine. Disabling DB writes.")
                to_db = False

    rows = []
    # loop symbols
    for csv_path in tqdm(csvs, desc="Symbols"):
        try:
            df = pd.read_csv(csv_path)
            symbol = Path(csv_path).stem.upper()
            # ensure it has date column and sorted
            if "date" not in df.columns:
                LOGGER.warning("Skipping %s: no date column", csv_path)
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            start_date = df['date'].min().strftime("%Y-%m-%d")
            end_date = df['date'].max().strftime("%Y-%m-%d")
        except Exception:
            LOGGER.exception("Failed to read %s, skipping", csv_path)
            continue

        # loop strategies
        for strategy_name, strategy_fn in strategy_items:
            try:
                LOGGER.info("Running %s on %s", strategy_name, symbol)
                # run backtest
                res = run_backtest(
                    df=df.copy(),
                    strategy_fn=strategy_fn,
                    initial_capital=initial_capital,
                    position_size_pct=position_size_pct,
                    commission=commission,
                    slippage=slippage,
                    verbose=False
                )
                metrics = res.get("metrics", {})
                trades = res.get("trades", [])
                # write trades json
                _write_trades(trades, strategy_name, symbol)

                # prepare summary row
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_trades": metrics.get("num_trades"),
                    "cagr": metrics.get("cagr"),
                    "sharpe": metrics.get("sharpe"),
                    "annual_vol": metrics.get("annualized_vol"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "max_dd_peak": metrics.get("max_drawdown_peak_date"),
                    "max_dd_trough": metrics.get("max_drawdown_trough_date"),
                    "win_rate": metrics.get("win_rate"),
                    "profit_factor": metrics.get("profit_factor"),
                    "expectancy": metrics.get("expectancy"),
                    "total_return": metrics.get("total_return"),
                }
                rows.append(row)

                # persist to DB if requested
                if to_db and engine:
                    _persist_to_db(engine, strategy_name, symbol, start_date, end_date, trades, metrics)
            except Exception:
                LOGGER.exception("Backtest failed for %s on %s", strategy_name, symbol)
                # write an error row so we can inspect later
                rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "error": traceback.format_exc()
                })

    # write summary CSV (append if exists)
    summary_df = pd.DataFrame(rows)
    if SUMMARY_FILE.exists():
        try:
            prev = pd.read_csv(SUMMARY_FILE)
            combined = pd.concat([prev, summary_df], ignore_index=True)
            combined.to_csv(SUMMARY_FILE, index=False)
            LOGGER.info("Appended %d rows to %s", len(summary_df), SUMMARY_FILE)
        except Exception:
            LOGGER.exception("Failed to append to summary file; overwriting.")
            summary_df.to_csv(SUMMARY_FILE, index=False)
    else:
        summary_df.to_csv(SUMMARY_FILE, index=False)
        LOGGER.info("Wrote summary file %s (%d rows)", SUMMARY_FILE, len(summary_df))

    LOGGER.info("Backtests completed. Summary rows: %d", len(rows))
    return SUMMARY_FILE

def parse_args():
    parser = argparse.ArgumentParser(description="Run backtests for all standardized CSVs using Chartink strategies")
    parser.add_argument("--input", type=str, default="data/standard/auto", help="folder with standardized CSVs")
    parser.add_argument("--strategies", type=str, default="ALL", help="comma-separated strategy names or ALL")
    parser.add_argument("--limit", type=int, default=None, help="limit number of symbols for quick tests")
    parser.add_argument("--to-db", action="store_true", help="persist results to Postgres (requires src.persistence.pg)")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--position-pct", type=float, default=0.1)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0005)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # select strategies
    if args.strategies.strip().upper() == "ALL":
        strategies = ALL_STRATEGIES
    else:
        wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
        strategies = {k: v for k, v in ALL_STRATEGIES.items() if k in wanted}
        if not strategies:
            LOGGER.error("No valid strategies selected. Available: %s", ", ".join(list(ALL_STRATEGIES.keys())))
            raise SystemExit(1)

    run_all(
        input_dir=Path(args.input),
        strategies=strategies,
        limit=args.limit,
        to_db=args.to_db,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_pct,
        commission=args.commission,
        slippage=args.slippage
    )
