from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_NAMES, TARGET_NAMES


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create causal features using information available at the close of each candle."""
    data = raw.copy().sort_index()
    close = data["Close"].astype(float)
    open_ = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).clip(lower=0)

    previous_close = close.shift(1)
    daily_return = close.pct_change()
    log_volume = np.log1p(volume)

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr_14 = true_range.rolling(14, min_periods=14).mean()

    volume_mean = log_volume.rolling(20, min_periods=20).mean()
    volume_std = log_volume.rolling(20, min_periods=20).std().replace(0, np.nan)
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()

    features = pd.DataFrame(index=data.index)
    features["open_gap"] = open_ / previous_close - 1
    features["high_from_prev_close"] = high / previous_close - 1
    features["low_from_prev_close"] = low / previous_close - 1
    features["close_from_prev_close"] = close / previous_close - 1
    features["return_1d"] = daily_return
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    features["range_pct"] = (high - low) / previous_close
    features["body_pct"] = (close - open_) / previous_close
    features["volume_z20"] = (log_volume - volume_mean) / volume_std
    features["rsi_14"] = rsi / 100.0
    features["atr_pct14"] = atr_14 / close
    features["realized_vol20"] = daily_return.rolling(20, min_periods=20).std()
    features["close_vs_ema20"] = close / ema_20 - 1
    features["close_vs_ema50"] = close / ema_50 - 1

    features = features[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return features


def build_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Next candle OHLC returns relative to the current candle close."""
    data = raw.copy().sort_index()
    current_close = data["Close"].astype(float)
    targets = pd.DataFrame(index=data.index)
    targets["next_open_return"] = data["Open"].shift(-1) / current_close - 1
    targets["next_high_return"] = data["High"].shift(-1) / current_close - 1
    targets["next_low_return"] = data["Low"].shift(-1) / current_close - 1
    targets["next_close_return"] = data["Close"].shift(-1) / current_close - 1
    return targets[TARGET_NAMES].replace([np.inf, -np.inf], np.nan)


def prepare_ticker(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return aligned features, targets, current dates, next-candle dates and closes."""
    features = build_features(raw)
    targets = build_targets(raw)
    next_dates = pd.Series(raw.index, index=raw.index).shift(-1)
    aligned = features.join(targets, how="inner")
    aligned["target_date"] = next_dates
    aligned["current_close"] = raw["Close"].astype(float)
    aligned = aligned.dropna(subset=FEATURE_NAMES + TARGET_NAMES + ["target_date", "current_close"])

    if aligned.empty:
        return {
            "features": np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            "targets": np.empty((0, len(TARGET_NAMES)), dtype=np.float32),
            "dates": np.array([], dtype="datetime64[ns]"),
            "target_dates": np.array([], dtype="datetime64[ns]"),
            "current_close": np.array([], dtype=np.float32),
        }

    return {
        "features": aligned[FEATURE_NAMES].to_numpy(dtype=np.float32),
        "targets": aligned[TARGET_NAMES].to_numpy(dtype=np.float32),
        "dates": aligned.index.to_numpy(dtype="datetime64[ns]"),
        "target_dates": pd.to_datetime(aligned["target_date"]).to_numpy(dtype="datetime64[ns]"),
        "current_close": aligned["current_close"].to_numpy(dtype=np.float32),
    }
