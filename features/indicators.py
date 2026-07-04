import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd":        macd_line,
        "macd_signal": signal_line,
        "macd_hist":   histogram,
    })


def compute_bollinger(close: pd.Series, period=20, std=2) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    pct = (close - lower) / (upper - lower).replace(0, float("nan"))
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_pct":   pct,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: open, high, low, close, volume.
    Returns same df with all indicator columns appended.
    """
    df = df.copy().sort_index()

    df["rsi_14"] = compute_rsi(df["close"])

    macd = compute_macd(df["close"])
    df["macd"]        = macd["macd"]
    df["macd_signal"] = macd["macd_signal"]
    df["macd_hist"]   = macd["macd_hist"]

    bb = compute_bollinger(df["close"])
    df["bb_upper"] = bb["bb_upper"]
    df["bb_mid"]   = bb["bb_mid"]
    df["bb_lower"] = bb["bb_lower"]
    df["bb_pct"]   = bb["bb_pct"]

    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"])

    df["returns_1d"] = df["close"].pct_change(1)
    df["returns_5d"] = df["close"].pct_change(5)

    df["next_day_up"] = df["close"].shift(-1) > df["close"]

    return df