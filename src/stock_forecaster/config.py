from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "generalized_transformer.pt"

# Starter universe only. Increase this carefully after the end-to-end pipeline works.
INDIAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "ONGC.NS",
]
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "TSLA",
    "XOM", "AMD", "COST", "KO", "AVGO", "NFLX", "WMT",
]
DEFAULT_TICKERS = INDIAN_TICKERS + US_TICKERS

START_DATE = "2010-01-01"
INTERVAL = "1d"
SEQUENCE_LENGTH = 64
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2024-12-31"
TEST_START = "2025-01-01"
UNSEEN_TEST_START = "2023-01-01"
RANDOM_SEED = 42

# Feature order is part of the saved model contract. Do not reorder without retraining.
FEATURE_NAMES = [
    "open_gap", "high_from_prev_close", "low_from_prev_close",
    "close_from_prev_close", "return_1d", "return_5d", "return_20d",
    "range_pct", "body_pct", "volume_z20", "rsi_14", "atr_pct14",
    "realized_vol20", "close_vs_ema20", "close_vs_ema50",
]
TARGET_NAMES = ["next_open_return", "next_high_return", "next_low_return", "next_close_return"]

MODEL_CONFIG = {
    "d_model": 96,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.10,
}
