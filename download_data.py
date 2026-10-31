from __future__ import annotations

import argparse

from stock_forecaster.config import DEFAULT_TICKERS, INTERVAL, START_DATE
from stock_forecaster.data import download_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the starter multi-stock OHLCV dataset.")
    parser.add_argument("--start", default=START_DATE, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--interval", default=INTERVAL, help="Yahoo Finance interval; default is 1d")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between ticker downloads")
    args = parser.parse_args()
    download_universe(args.tickers, start=args.start, interval=args.interval, pause_seconds=args.pause)


if __name__ == "__main__":
    main()
