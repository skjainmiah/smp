# src/backtest/engine.py
"""
Backtest engine for signal-based strategies on OHLC EOD data.

Assumptions:
- Input df must be standardized with columns:
    date, open, high, low, close, adj_close, volume, symbol
  and date sorted ascending.

- strategy_fn(df) -> pd.Series of signals aligned with df index:
    1  -> enter/hold long
    0  -> flat / hold no position
   -1  -> optional exit/sell signal (engine treats -1 as close/short-exit;
          by default we only support long entries/exits; shorting can be added)

Main functions:
- run_backtest(...)  : run backtest over dataframe and return results dictionary
- summarize_backtest(...) : compute performance metrics from results

Usage:
    from src.backtest.engine import run_backtest
    res = run_backtest(df, my_strategy_fn, initial_capital=100000, commission=0.001, slippage=0.001)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import math
from datetime import datetime

# Performance metric helpers --------------------------------------------------

def _annualize_return(total_return: float, days: int) -> float:
    """Compute CAGR given total_return (final/initial) and days elapsed."""
    if days <= 0:
        return 0.0
    years = days / 252.0  # trading days approximation
    if years <= 0:
        return 0.0
    try:
        return (total_return ** (1.0 / years)) - 1.0
    except Exception:
        return 0.0

def _annualized_vol(returns: pd.Series) -> float:
    """Annualized volatility of daily returns."""
    if returns.dropna().empty:
        return 0.0
    return returns.std(ddof=1) * math.sqrt(252.0)

def _sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio (excess returns). risk_free is annual rate (e.g., 0.04)."""
    if returns.dropna().empty:
        return 0.0
    # convert risk_free to daily
    rf_daily = (1.0 + risk_free) ** (1.0 / 252.0) - 1.0 if risk_free else 0.0
    excess = returns - rf_daily
    ann_excess = excess.mean() * 252.0
    ann_vol = _annualized_vol(returns)
    if ann_vol == 0:
        return float('inf') if ann_excess > 0 else 0.0
    return ann_excess / ann_vol

def _max_drawdown(equity: pd.Series) -> Tuple[float, Optional[str], Optional[str]]:
    """
    Max drawdown and its start/end dates.
    Returns (max_dd, peak_date_str, trough_date_str) where max_dd is positive percentage.
    """
    if equity.empty:
        return 0.0, None, None
    roll_max = equity.cummax()
    drawdowns = (roll_max - equity) / roll_max
    if drawdowns.max() <= 0:
        return 0.0, None, None
    idx_trough = drawdowns.idxmax()
    trough_date = equity.index[idx_trough]
    # find peak date prior to trough
    peak_idx = equity[:idx_trough + 1].idxmax()
    peak_date = equity.index[peak_idx]
    return float(drawdowns.max()), str(peak_date), str(trough_date)

# Trade dataclass -------------------------------------------------------------

@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    qty: Optional[int] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: Optional[float] = None

# Engine ---------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 100000.0,
    position_size_pct: Optional[float] = None,
    fixed_size: Optional[int] = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    risk_per_trade: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized OHLC DataFrame sorted ascending by date.
    strategy_fn : callable
        Function that accepts df and returns a pd.Series (same length) of signals:
        1=buy/long, 0=flat, -1=exit/short-signal (defaults to close).
    initial_capital : float
        Starting capital in currency units.
    position_size_pct : float, optional
        Fraction of equity to allocate per trade (e.g., 0.1 = 10%).
        If provided, fixed_size is ignored.
    fixed_size : int, optional
        Fixed number of shares to buy at each entry.
    commission : float
        Commission charged as fraction of trade value (e.g., 0.001 = 0.1%).
    slippage : float
        Slippage as fraction of price (e.g., 0.001 = 0.1%).
    risk_per_trade : float
        If provided, attempts money management by sizing position so that
        risk_per_trade (absolute currency) is the max loss to stop_loss.
        NOTE: requires stop loss logic inside strategy or an external stop price.
    verbose : bool
        If True prints basic progress.

    Returns
    -------
    dict
        {
            'trades': List[Trade as dict],
            'equity_curve': pd.Series (indexed by date),
            'daily_returns': pd.Series,
            'metrics': {...}
        }
    """
    # validation
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    required_cols = {"date", "open", "high", "low", "close"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # ensure date index and sorted
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df_index = df['date']
    df.set_index('date', inplace=True)

    # obtain signals
    signals = strategy_fn(df.reset_index())  # pass reset index so strategy can expect 'date' column
    if not isinstance(signals, (pd.Series, np.ndarray, list)):
        raise ValueError("strategy_fn must return a pandas Series/array/list of signals.")
    sig = pd.Series(signals, index=df.index).fillna(0).astype(int)

    equity = []
    cash = initial_capital
    position = 0
    avg_entry_price = 0.0
    trades: List[Trade] = []
    equity_index = []
    last_equity = initial_capital

    # track daily portfolio value
    for current_date, row in df.iterrows():
        price = float(row['close'])
        signal = int(sig.loc[current_date])

        # Entry logic: when signal==1 and no position -> buy
        if signal == 1 and position == 0:
            # determine qty
            if fixed_size is not None:
                qty = int(fixed_size)
            elif position_size_pct is not None:
                alloc = cash * float(position_size_pct)
                qty = int(alloc // price) if price > 0 else 0
            else:
                # default: full allocation
                qty = int(cash // price) if price > 0 else 0

            if qty > 0:
                # account for slippage: assume buy price = price * (1 + slippage)
                buy_price = price * (1.0 + float(slippage))
                trade_value = qty * buy_price
                fee = trade_value * float(commission)
                total_cost = trade_value + fee
                if total_cost > cash:
                    # reduce qty to fit cash
                    qty = int(cash // (buy_price * (1.0 + commission)))
                    trade_value = qty * buy_price
                    fee = trade_value * float(commission)
                    total_cost = trade_value + fee

                if qty <= 0:
                    # cannot open a position
                    if verbose:
                        print(f"{current_date.date()}: insufficient cash to open position at price {buy_price:.2f}")
                else:
                    cash -= total_cost
                    position = qty
                    avg_entry_price = buy_price
                    trades.append(Trade(entry_date=str(current_date.date()), entry_price=buy_price, qty=qty, fees=fee))
                    if verbose:
                        print(f"{current_date.date()}: BUY {qty} @ {buy_price:.2f}, fees {fee:.2f}")

        # Exit logic: when signal == -1 and position > 0 -> sell
        elif signal == -1 and position > 0:
            sell_price = price * (1.0 - float(slippage))
            proceeds = position * sell_price
            fee = proceeds * float(commission)
            cash += (proceeds - fee)
            last_trade = trades[-1] if trades else None
            pnl = (sell_price - (last_trade.entry_price if last_trade else avg_entry_price)) * position - (last_trade.fees if last_trade and last_trade.fees else 0) - fee
            pnl_pct = pnl / (last_trade.entry_price * position) if last_trade and last_trade.entry_price and position else None
            # finalize trade
            if last_trade:
                last_trade.exit_date = str(current_date.date())
                last_trade.exit_price = sell_price
                last_trade.pnl = pnl
                last_trade.pnl_pct = pnl_pct
                last_trade.fees = (last_trade.fees or 0.0) + fee
            position = 0
            avg_entry_price = 0.0
            if verbose:
                print(f"{current_date.date()}: SELL all @ {sell_price:.2f}, fees {fee:.2f}, pnl {pnl:.2f}")

        # otherwise hold existing position -- update mark-to-market equity
        mtm = cash + (position * price if position > 0 else 0.0)
        equity.append(mtm)
        equity_index.append(current_date)

    equity_series = pd.Series(data=equity, index=equity_index, name="equity")
    # daily returns from equity curve
    daily_returns = equity_series.pct_change().fillna(0.0)

    # finalize unfinished open positions by marking to last close but not auto-closing trades.
    # If position still open, compute unrealized pnl but trades list contains only closed trades as updated above.

    # compute metrics
    total_return = (equity_series.iloc[-1] / initial_capital) if initial_capital else 1.0
    days = (equity_series.index[-1] - equity_series.index[0]).days if len(equity_series.index) >= 2 else 0
    cagr = _annualize_return(total_return, days) if initial_capital else 0.0
    ann_vol = _annualized_vol(daily_returns)
    sharpe = _sharpe_ratio(daily_returns)
    max_dd, dd_start, dd_trough = _max_drawdown(equity_series)
    num_trades = sum(1 for t in trades if t.exit_date is not None)
    wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
    losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
    win_rate = (len(wins) / num_trades) if num_trades > 0 else None
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = -sum(t.pnl for t in losses) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss and gross_profit else None
    expectancy = None
    if num_trades > 0:
        avg_win = (sum(t.pnl for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
        expectancy = ((len(wins)/num_trades) * avg_win) - ((len(losses)/num_trades) * abs(avg_loss))

    metrics = {
        "initial_capital": initial_capital,
        "ending_equity": float(equity_series.iloc[-1]),
        "total_return": float(total_return - 1.0),
        "cagr": float(cagr),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_peak_date": dd_start,
        "max_drawdown_trough_date": dd_trough,
        "num_trades": int(num_trades),
        "win_rate": float(win_rate) if win_rate is not None else None,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "expectancy": float(expectancy) if expectancy is not None else None
    }

    # convert trades dataclass objects to dicts for JSON-friendly storage
    trades_out = [asdict(t) for t in trades]

    return {
        "trades": trades_out,
        "equity_curve": equity_series,
        "daily_returns": daily_returns,
        "metrics": metrics
    }

# Utility: nice summary ------------------------------------------------------

def summarize_backtest(res: Dict[str, Any]) -> pd.DataFrame:
    """
    Return a small DataFrame summarizing key metrics and top-level numbers.
    """
    m = res.get("metrics", {})
    summary = {
        "initial_capital": m.get("initial_capital"),
        "ending_equity": m.get("ending_equity"),
        "total_return_pct": m.get("total_return") * 100 if m.get("total_return") is not None else None,
        "cagr_pct": m.get("cagr") * 100 if m.get("cagr") is not None else None,
        "sharpe": m.get("sharpe"),
        "annual_vol_pct": m.get("annualized_vol") * 100 if m.get("annualized_vol") is not None else None,
        "max_drawdown_pct": m.get("max_drawdown") * 100 if m.get("max_drawdown") is not None else None,
        "num_trades": m.get("num_trades"),
        "win_rate_pct": m.get("win_rate") * 100 if m.get("win_rate") is not None else None,
        "profit_factor": m.get("profit_factor"),
        "expectancy": m.get("expectancy")
    }
    return pd.DataFrame([summary])

# Quick demo runner if executed directly -------------------------------------

if __name__ == "__main__":
    # demo usage with example strategy (moving average crossover)
    import sys
    from math import isnan

    def demo_strategy(df: pd.DataFrame):
        # expects df with date index or 'date' column
        df = df.copy()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        sig = pd.Series(0, index=df.index)
        sig[(df['ma20'] > df['ma50']) & (df['ma20'].shift(1) <= df['ma50'].shift(1))] = 1
        sig[(df['ma20'] < df['ma50']) & (df['ma20'].shift(1) >= df['ma50'].shift(1))] = -1
        return sig.fillna(0).astype(int)

    if len(sys.argv) < 2:
        print("Usage: python engine.py /path/to/standardized_symbol.csv")
        sys.exit(0)

    path = sys.argv[1]
    df = pd.read_csv(path)
    out = run_backtest(df, demo_strategy, initial_capital=100000, position_size_pct=0.2, commission=0.0005, slippage=0.0005, verbose=True)
    print("METRICS:")
    for k, v in out['metrics'].items():
        print(f"  {k}: {v}")
    print("\nTRADES:")
    for t in out['trades']:
        print(t)
