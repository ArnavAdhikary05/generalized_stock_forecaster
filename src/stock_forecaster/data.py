from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_TICKERS, INTERVAL, RAW_DATA_DIR, START_DATE


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance frame to a clean OHLCV table."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    data = frame.copy()
    # yfinance can return a MultiIndex in some download modes; history() normally does not.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[(data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    data["Volume"] = data["Volume"].fillna(0).clip(lower=0)
    return data


def download_one(ticker: str, start: str = START_DATE, interval: str = INTERVAL) -> pd.DataFrame:
    """Download one ticker. Yahoo Finance data is for personal/research use; check its terms."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for downloading data. Install it with: pip install yfinance") from exc

    history = yf.Ticker(ticker).history(
        start=start,
        interval=interval,
        auto_adjust=True,
        actions=False,
        repair=True,
    )
    return normalize_ohlcv(history)


def download_universe(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = START_DATE,
    interval: str = INTERVAL,
    output_dir: Path = RAW_DATA_DIR,
    pause_seconds: float = 0.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures: list[str] = []

    for ticker in tickers:
        safe_name = ticker.replace("^", "INDEX_").replace("=", "_").replace("/", "_")
        destination = output_dir / f"{safe_name}.csv"
        try:
            print(f"Downloading {ticker} ...", flush=True)
            frame = download_one(ticker, start=start, interval=interval)
            if frame.empty:
                print(f"  No rows returned for {ticker}; skipping.")
                failures.append(ticker)
                continue
            frame.to_csv(destination, index_label="Date")
            print(f"  Saved {len(frame):,} rows -> {destination.name}")
            successes += 1
        except Exception as exc:  # keep going if one symbol fails
            print(f"  Failed {ticker}: {exc}")
            failures.append(ticker)
        time.sleep(max(0.0, pause_seconds))

    print(f"\nDownload complete: {successes} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed tickers:", ", ".join(failures))


def load_raw_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return normalize_ohlcv(frame)


def load_all_csvs(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all valid CSVs from data/raw, using each filename as its ticker name."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    result: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            frame = load_raw_csv(path)
            if len(frame) >= 100:
                result[path.stem] = frame
            else:
                print(f"Skipping {path.name}: only {len(frame)} usable rows.")
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if len(result) < 5:
        raise RuntimeError(
            f"Only {len(result)} usable ticker files found in {data_dir}. "
            "Download at least five tickers before training."
        )
    return result
