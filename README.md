# Generalized Stock Forecaster — v0.1

A research starter for training **one shared Transformer** on daily OHLCV histories from many tickers. It predicts the next candle's Open, High, Low and Close as returns relative to the latest close, plus an auxiliary bullish/bearish score.

> This is an educational/research model, not a reliable trading system or investment advice. It does not guarantee profitable forecasts. Backtest with realistic fees, slippage, market impact and risk controls before considering any live use.

## What this first version does

- Downloads daily adjusted OHLCV history for a starter universe of Indian and US tickers.
- Builds causal features: normalized OHLC, returns, range/body, volume z-score, RSI, ATR, realized volatility and EMA distances.
- Trains one PyTorch Transformer across multiple tickers.
- Does **not** use a ticker-ID embedding, so it cannot simply memorize a ticker label.
- Holds out a group of ticker symbols from training and reports an unseen-stock test.
- Also reports a future-period test on tickers that were present during training.
- Saves a checkpoint and includes a command-line next-candle inference script.

The default ticker list is only a starter sample (30 symbols). It is not a sufficiently broad market dataset to claim universal generalization. Expand it only after the full pipeline runs successfully.

## 1. Environment setup

Python 3.10+ is recommended. Create a virtual environment:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended by the official PyTorch selector for your CUDA/Windows setup: https://pytorch.org/get-started/locally/ . Then install the project:

```bash
pip install -e .
pip install -r requirements.txt
```

If you already have a working CUDA-enabled PyTorch environment, do not reinstall PyTorch unnecessarily. Verify it first:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Download the starter dataset

From the project root:

```bash
python download_data.py
```

This downloads daily candles from 2010 onward into `data/raw/`. To download a smaller test universe first:

```bash
python download_data.py --tickers RELIANCE.NS TCS.NS INFY.NS AAPL MSFT NVDA
```

For a small universe, the model's unseen-stock evaluation is weak. Use the full list (or a much larger clean dataset) for a meaningful experiment.

**Data note:** this starter uses `yfinance`, an open-source interface to Yahoo Finance's publicly available APIs. Its maintainers state that it is intended for personal use and ask users to review Yahoo's terms. It is convenient for prototyping, not a substitute for a licensed market-data feed for commercial or large-scale redistribution. See https://ranaroussi.github.io/yfinance/ .

## 3. Train

```bash
python -m stock_forecaster.train
```

Optional settings:

```bash
python -m stock_forecaster.train --epochs 50 --batch-size 128 --sequence-length 64
```

The default chronological evaluation windows are:

- Training: target candle dates through 2022-12-31, using only the training ticker group.
- Validation: 2023-01-01 through 2024-12-31, using the training ticker group.
- Seen-stock future test: 2025 onward, using training-group tickers.
- Unseen-stock test: 2023 onward, using tickers excluded entirely from training.

The unseen-stock test is the key transfer test. Do not tune repeatedly against the test results; use a new untouched holdout set for final reporting once the model design is settled.

Artifacts are saved under `checkpoints/`:

- `generalized_transformer.pt` — trained weights, model config and normalization statistics.
- `generalized_transformer.history.csv` — training/validation loss by epoch.

## 4. Predict the next candle

After training:

```bash
python -m stock_forecaster.predict RELIANCE.NS
python -m stock_forecaster.predict AAPL
```

The script fetches fresh daily data for the requested ticker and prints the next-candle OHLC estimate. A bullish score is also printed, but it is **uncalibrated** until a separate calibration process is implemented. A 70% model score must not be interpreted as a proven 70% real-world win probability.

## 5. Add more tickers

Edit `INDIAN_TICKERS` and `US_TICKERS` in `src/stock_forecaster/config.py`, or pass a custom list to `download_data.py`. Prefer liquid instruments with long, clean histories. Keep an untouched group of symbols for final unseen-stock testing.

For larger datasets, move from this prototype data source to a licensed provider with a documented API, adjustment policy, corporate-action handling and redistribution rights. Store raw data by ticker and date, version the dataset, and record the download timestamp/source.

## 6. Important modeling limitations and next improvements

1. **No claim of market predictability:** a model may learn little or no signal beyond a naïve baseline. Compare it to zero-return, previous-close and simple statistical baselines.
2. **Data leakage:** all features are computed using data available at or before the input candle's close. Splits are by target date. Do not randomly split overlapping windows.
3. **Survivorship bias:** the starter ticker universe contains current names, not a historical point-in-time universe of delisted companies. This can inflate backtest performance.
4. **Corporate actions:** OHLC is downloaded with `auto_adjust=True`; verify provider adjustment behavior before comparing predictions to broker candles.
5. **Market calendars:** each ticker is modeled on its own trading sessions. This version does not align candles across time zones or add index/macro context.
6. **Prediction uncertainty:** this version predicts point estimates. Later versions should produce quantiles or distributions and evaluate interval coverage.
7. **Execution realism:** directional accuracy does not equal profitability. Add transaction costs, slippage, liquidity filters and walk-forward portfolio backtests before evaluating trading value.

## Suggested next research iterations

- Increase the dataset from 30 to several hundred or thousands of liquid symbols, with a properly licensed data source.
- Add sector/index context and market-regime features without leaking future information.
- Compare the Transformer with XGBoost, LightGBM, LSTM and a zero-return baseline.
- Add quantile outputs for prediction intervals, calibration and uncertainty-aware abstention.
- Add separate models/evaluation for daily, hourly and intraday bars. Do not mix timeframes as if they were identical observations.
- Integrate the inference function into the existing Streamlit app only after out-of-sample behavior is validated.

## Suggested Streamlit integration

Once the checkpoint exists, the page can call the inference function:

```python
from stock_forecaster.predict import predict_next_candle

forecast = predict_next_candle(selected_ticker)
st.metric("Predicted next close", f"{forecast['predicted_close']:,.2f}")
st.caption(
    f"Model direction: {forecast['model_direction']} | "
    f"Uncalibrated bullish score: {forecast['bullish_probability_uncalibrated']:.1%}"
)
```

Keep the forecast visibly labeled as a model estimate, not a guaranteed price target.
