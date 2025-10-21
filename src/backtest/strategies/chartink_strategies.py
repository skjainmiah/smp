# src/backtest/strategies/chartink_strategies.py
"""
Chartink strategies (OHLC-only) — one file with multiple strategy functions.

Each function:
    - accepts a standardized DataFrame `df` with columns:
        date, open, high, low, close, adj_close, volume, symbol
      and index aligned as desired (functions operate on df sorted by date).
    - returns a pandas Series of signals aligned to df.index with values in {1, 0, -1}.

Removed intraday-only strategies. Remaining OHLC-only strategies:
 - short_term_breakout
 - potential_breakout
 - perfect_sell_short
 - boss_scanner
 - strong_stocks_consecutive_up
 - daily_rsi_oversold_overbought
 - nr7_current_day
 - ichimoku_uptrend_cloud_crossover
"""

from typing import Optional
import pandas as pd
import numpy as np


# -------------------------
# Helper indicator functions
# -------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted moving average — linear weights (most recent biggest)."""
    weights = np.arange(1, period + 1)
    def _wma(x):
        if np.isnan(x).all():
            return np.nan
        return np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()
    return series.rolling(period, min_periods=1).apply(_wma, raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="ffill")


def bollinger_bands(series: pd.Series, period: int = 20, stds: float = 2.0):
    ma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0)
    upper = ma + stds * std
    lower = ma - stds * std
    return upper, ma, lower


def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52):
    """
    Returns dict with keys:
      tenkan, kijun, span_a, span_b, chikou
    span_a/span_b are shifted forward by kijun (26) bars (same convention as TradingView).
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)  # leading span A
    span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)  # lagging

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "span_a": span_a,
        "span_b": span_b,
        "chikou": chikou
    }


def range_series(df: pd.DataFrame) -> pd.Series:
    return (df['high'] - df['low'])


# -------------------------
# Strategy functions (OHLC-only)
# -------------------------
def short_term_breakout(df: pd.DataFrame) -> pd.Series:
    """
    Price breaks above short-term highs with volume confirmation (daily).
    - 5-day rolling max of close > (120-day rolling max shifted by 6) * 1.05
    - volume > sma(volume,5)
    - close > previous close
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)

    max5 = close.rolling(5, min_periods=1).max()
    max120 = close.rolling(120, min_periods=1).max().shift(6)

    cond1 = max5 > (max120 * 1.05)
    cond2 = volume > sma(volume, 5)
    cond3 = close > close.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[(cond1 & cond2 & cond3).fillna(False)] = 1
    return sig.astype(int)


def potential_breakout(df: pd.DataFrame) -> pd.Series:
    """
    Potential breakout (daily):
    - close * 1.05 > rolling_max(high, 200)
    - volume > sma(volume,50)
    - close > 90 (price filter)
    - sma20 > sma50 and sma50 > sma200
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    volume = df['volume'].astype(float)

    rolling_max_200 = high.rolling(200, min_periods=1).max()
    cond1 = (close * 1.05) > rolling_max_200
    cond3 = volume > sma(volume, 50)
    cond4 = close > 90
    cond5 = sma(close, 20) > sma(close, 50)
    cond6 = sma(close, 50) > sma(close, 200)

    cond_final = cond1 & cond3 & cond4 & cond5 & cond6
    sig = pd.Series(0, index=df.index)
    sig[cond_final.fillna(False)] = 1
    return sig.astype(int)


def perfect_sell_short(df: pd.DataFrame) -> pd.Series:
    """
    Short/sell signal based on recent highs and volume (daily).
    Emits -1 when conditions match.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    high = df['high'].astype(float)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)

    cond1 = high.shift(1) > high.shift(2)
    cond2 = close < high.shift(1)
    cond3 = volume > volume.shift(3)
    cond4 = high < high.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[(cond1 & cond2 & cond3 & cond4).fillna(False)] = -1
    return sig.astype(int)


def boss_scanner(df: pd.DataFrame) -> pd.Series:
    """
    Boss scanner (daily + weekly + monthly):
      - daily volume > sma(volume,20)
      - daily close > daily upper Bollinger band (20,2)
      - weekly close > weekly upper Bollinger
      - monthly close > monthly upper Bollinger
      - RSI daily/weekly/monthly > 60
      - monthly WMA30 > monthly WMA50
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    cond_vol = volume > sma(volume, 20)
    upper_daily, ma_daily, lower_daily = bollinger_bands(close, period=20, stds=2)
    cond_boll = close > upper_daily

    # weekly, monthly resamples
    dfi = df.copy()
    dfi['date'] = pd.to_datetime(dfi['date'])
    dfi = dfi.set_index('date')
    weekly = dfi.resample('W').agg({'high':'max','low':'min','close':'last','volume':'sum'})
    monthly = dfi.resample('M').agg({'high':'max','low':'min','close':'last','volume':'sum'})

    weekly_close = weekly['close']
    w_upper, w_ma, w_lower = bollinger_bands(weekly_close, period=20, stds=2)
    weekly_cond = weekly_close > w_upper

    monthly_close = monthly['close']
    m_upper, m_ma, m_lower = bollinger_bands(monthly_close, period=20, stds=2)
    monthly_cond = monthly_close > m_upper

    weekly_cond_fwd = weekly_cond.reindex(dfi.index, method='ffill').fillna(False)
    monthly_cond_fwd = monthly_cond.reindex(dfi.index, method='ffill').fillna(False)

    # RSI checks (daily, weekly, monthly)
    rsi_daily = rsi(close, 14)
    weekly_rsi = rsi(weekly_close, 14).reindex(dfi.index, method='ffill').fillna(0)
    monthly_rsi = rsi(monthly_close, 14).reindex(dfi.index, method='ffill').fillna(0)
    cond_rsi = (rsi_daily > 60) & (weekly_rsi > 60) & (monthly_rsi > 60)

    # monthly WMA (using wma helper)
    m_wma30 = wma(monthly_close, 30)
    m_wma50 = wma(monthly_close, 50)
    m_wma30_f = m_wma30.reindex(dfi.index, method='ffill')
    m_wma50_f = m_wma50.reindex(dfi.index, method='ffill')
    cond_wma = (m_wma30_f > m_wma50_f)

    sig = pd.Series(0, index=df.index)
    cond_all = cond_vol & cond_boll & weekly_cond_fwd & monthly_cond_fwd & cond_rsi & cond_wma
    sig[cond_all.fillna(False)] = 1
    return sig.astype(int)


def strong_stocks_consecutive_up(df: pd.DataFrame, days: int = 5) -> pd.Series:
    """
    Stocks closing higher than previous trading day continuously for the past `days` days.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)

    cond = pd.Series(True, index=df.index)
    for lag in range(1, days):
        cond = cond & (close > close.shift(lag))
    sig = pd.Series(0, index=df.index)
    sig[cond.fillna(False)] = 1
    return sig.astype(int)


def daily_rsi_oversold_overbought(df: pd.DataFrame) -> pd.Series:
    """
    Daily RSI oversold/overbought:
      - RSI(14) crosses above 30 -> +1
      - RSI(14) crosses below 70 -> -1
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)
    r = rsi(close, 14)

    cross_above_30 = (r > 30) & (r.shift(1) <= 30)
    cross_below_70 = (r < 70) & (r.shift(1) >= 70)

    sig = pd.Series(0, index=df.index)
    sig[cross_above_30.fillna(False)] = 1
    sig[cross_below_70.fillna(False)] = -1
    return sig.astype(int)


def nr7_current_day(df: pd.DataFrame) -> pd.Series:
    """
    NR7: Narrowest range day within last 7 days (i.e., today's range is the minimum over last 7).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    rng = range_series(df)
    rolling_min_7 = rng.rolling(7, min_periods=1).min()
    cond = rng == rolling_min_7
    sig = pd.Series(0, index=df.index)
    sig[cond.fillna(False)] = 1
    return sig.astype(int)


def ichimoku_uptrend_cloud_crossover(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku cloud crossover (daily):
      - Close crosses above cloud top and span_a > span_b
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['close'].astype(float)

    ichi = ichimoku(df, tenkan=9, kijun=26, senkou_b=52)
    span_a = ichi['span_a']
    span_b = ichi['span_b']

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)

    cond_cross = (close > cloud_top) & (close.shift(1) <= cloud_top.shift(1))
    cond_span = span_a > span_b

    sig = pd.Series(0, index=df.index)
    sig[(cond_cross & cond_span).fillna(False)] = 1
    return sig.astype(int)


# -------------------------
# Expose OHLC-only strategies in a dict for convenience
# -------------------------
ALL_STRATEGIES = {
    "short_term_breakout": short_term_breakout,
    "potential_breakout": potential_breakout,
    "perfect_sell_short": perfect_sell_short,
    "boss_scanner": boss_scanner,
    "strong_stocks_consecutive_up": strong_stocks_consecutive_up,
    "daily_rsi_oversold_overbought": daily_rsi_oversold_overbought,
    "nr7_current_day": nr7_current_day,
    "ichimoku_uptrend_cloud_crossover": ichimoku_uptrend_cloud_crossover
}
