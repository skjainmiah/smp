# src/strategies/bollinger_rsi.py
"""
Bollinger Bands + RSI strategy.
Produces signal for the latest EOD candle:
- STRONG BUY: RSI <= rsi_oversold AND (close <= lower_band OR touched lower_band)
- BUY: RSI oversold OR close below lower_band (weaker)
- STRONG SELL / SELL symmetrical
Also returns support/resistance estimates and a confidence score (0..1).
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

NAME = "bollinger_rsi"

# params (tuneable)
BB_WINDOW = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SR_LOOKBACK = 50   # days to search for recent pivot support/resistance

def _rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_down = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def _recent_support_resistance(df: pd.DataFrame, lookback: int = 50):
    """
    Simple support/resistance estimates:
    - support: recent minimum low over lookback days
    - resistance: recent maximum high over lookback days
    """
    if len(df) < 2:
        return None, None
    last = df.tail(lookback)
    support = float(last["low"].min())
    resistance = float(last["high"].max())
    return support, resistance

def signal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Accepts standardized EOD DataFrame with columns: date, open, high, low, close (and maybe volume).
    Returns: dict with keys: action, confidence (0..1), support, resistance, target, stoploss, rationale
    """
    if df is None or df.empty:
        return {"action": None}

    df = df.copy().reset_index(drop=True)
    # ensure numeric columns
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    close = df["close"].astype(float)

    if len(close) < max(BB_WINDOW, RSI_PERIOD) + 1:
        return {"action": None}

    ma, upper, lower = _bollinger_bands(close, window=BB_WINDOW, n_std=BB_STD)
    rsi = _rsi(close, period=RSI_PERIOD)

    last_idx = len(df) - 1
    last_close = float(close.iloc[last_idx])
    last_high = float(df["high"].iloc[last_idx])
    last_low = float(df["low"].iloc[last_idx])
    last_upper = float(upper.iloc[last_idx])
    last_lower = float(lower.iloc[last_idx])
    last_ma = float(ma.iloc[last_idx])
    last_rsi = float(rsi.iloc[last_idx])

    # check touches: did any candle in last 1-2 days touch the band?
    touched_lower = False
    touched_upper = False
    for i in range(max(0, last_idx-2), last_idx+1):
        if pd.notna(lower.iloc[i]) and df["low"].iloc[i] <= lower.iloc[i] + 1e-8:
            touched_lower = True
        if pd.notna(upper.iloc[i]) and df["high"].iloc[i] >= upper.iloc[i] - 1e-8:
            touched_upper = True

    # support / resistance
    support, resistance = _recent_support_resistance(df, lookback=SR_LOOKBACK)

    # compute distance metrics used for confidence
    # Relative distance to band (0..1) where smaller distance -> stronger signal
    def rel_dist_to_lower(close_price, lower_band, ma_band):
        try:
            if lower_band is None or ma_band is None or ma_band==0:
                return 1.0
            return min(1.0, abs(close_price - lower_band) / (ma_band))
        except Exception:
            return 1.0

    def rel_dist_to_upper(close_price, upper_band, ma_band):
        try:
            if upper_band is None or ma_band is None or ma_band==0:
                return 1.0
            return min(1.0, abs(upper_band - close_price) / (ma_band))
        except Exception:
            return 1.0

    dist_lower = rel_dist_to_lower(last_close, last_lower, max(abs(last_ma), 1e-6))
    dist_upper = rel_dist_to_upper(last_close, last_upper, max(abs(last_ma), 1e-6))

    # Confidence heuristic (0..1) for buy: lower distance closer -> higher confidence, RSI more extreme -> higher confidence
    buy_conf = max(0.0, 1.0 - dist_lower) * (1.0 if last_rsi <= 50 else max(0.2, (100-last_rsi)/100))
    sell_conf = max(0.0, 1.0 - dist_upper) * (1.0 if last_rsi >= 50 else max(0.2, last_rsi/100))

    # Convert to action with strong / normal thresholds
    action = None
    confidence = 0.0
    rationale = []

    # Strong Buy: RSI oversold AND (close <= lower OR touched lower)
    if last_rsi <= RSI_OVERSOLD and (last_close <= last_lower or touched_lower):
        action = "STRONG BUY"
        confidence = min(1.0, 0.6 + 0.4 * buy_conf)
        rationale.append("RSI oversold and price at/lower Bollinger band")
    # Buy: either RSI oversold OR close below lower band OR touched lower recently
    elif (last_rsi <= RSI_OVERSOLD) or (last_close <= last_lower) or touched_lower:
        action = "BUY"
        confidence = min(1.0, 0.3 + 0.5 * buy_conf)
        rationale.append("RSI oversold or price at/lower Bollinger band")
    # Strong Sell: RSI overbought AND (close >= upper OR touched upper)
    elif last_rsi >= RSI_OVERBOUGHT and (last_close >= last_upper or touched_upper):
        action = "STRONG SELL"
        confidence = min(1.0, 0.6 + 0.4 * sell_conf)
        rationale.append("RSI overbought and price at/above Bollinger upper band")
    elif (last_rsi >= RSI_OVERBOUGHT) or (last_close >= last_upper) or touched_upper:
        action = "SELL"
        confidence = min(1.0, 0.3 + 0.5 * sell_conf)
        rationale.append("RSI overbought or price at/above Bollinger upper band")
    else:
        action = None

    # Compute simple target/stoploss: target = close * (1 + 0.03 * confidence) ; stoploss = close*(1 - 0.02)
    if action in ("BUY", "STRONG BUY"):
        target = round(last_close * (1 + 0.03 * (0.5 + confidence)), 2)
        stoploss = round(last_close * 0.97, 2)
    elif action in ("SELL", "STRONG SELL"):
        target = round(last_close * (1 - 0.03 * (0.5 + confidence)), 2)
        stoploss = round(last_close * 1.03, 2)
    else:
        target = None
        stoploss = None

    return {
        "action": action,
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "support": round(float(support), 2) if support is not None else None,
        "resistance": round(float(resistance), 2) if resistance is not None else None,
        "target": target,
        "stoploss": stoploss,
        "rsi": round(last_rsi, 2),
        "bb_upper": round(last_upper, 2) if last_upper is not None else None,
        "bb_lower": round(last_lower, 2) if last_lower is not None else None,
        "ma": round(last_ma, 2) if last_ma is not None else None,
        "rationale": "; ".join(rationale)
    }
