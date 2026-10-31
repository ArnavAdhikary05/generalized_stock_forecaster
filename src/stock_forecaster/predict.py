from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CHECKPOINT, FEATURE_NAMES, START_DATE
from .data import download_one
from .features import build_features
from .model import CandleTransformer


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model before requesting predictions."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Feature list in checkpoint does not match this code version. Retrain the model.")

    model = CandleTransformer(
        input_features=len(FEATURE_NAMES),
        sequence_length=int(checkpoint["sequence_length"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_next_candle(ticker: str, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)

    # Use fresh historical data for the requested ticker. 10 years gives rolling indicators
    # enough warm-up history while keeping the request to one instrument.
    raw = download_one(ticker, start=START_DATE)
    if raw.empty:
        raise RuntimeError(f"No historical data returned for {ticker}.")

    feature_frame = build_features(raw).dropna()
    sequence_length = int(checkpoint["sequence_length"])
    if len(feature_frame) < sequence_length:
        raise RuntimeError(
            f"Only {len(feature_frame)} usable feature rows for {ticker}; need at least {sequence_length}."
        )

    latest_features = feature_frame.iloc[-sequence_length:].to_numpy(dtype=np.float32)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    scaled_features = (latest_features - feature_mean) / feature_std
    tensor = torch.from_numpy(scaled_features).unsqueeze(0).to(device)

    with torch.inference_mode():
        scaled_prediction, direction_logit = model(tensor)
        predicted_returns = scaled_prediction.squeeze(0).cpu().numpy() * target_std + target_mean
        bullish_probability = float(torch.sigmoid(direction_logit).item())

    last_date = feature_frame.index[-1]
    previous_close = float(raw.loc[last_date, "Close"])
    predicted_open = previous_close * (1.0 + float(predicted_returns[0]))
    predicted_high = previous_close * (1.0 + float(predicted_returns[1]))
    predicted_low = previous_close * (1.0 + float(predicted_returns[2]))
    predicted_close = previous_close * (1.0 + float(predicted_returns[3]))

    # Enforce the candle geometry: high cannot be below open/close and low cannot be above them.
    predicted_high = max(predicted_high, predicted_open, predicted_close)
    predicted_low = min(predicted_low, predicted_open, predicted_close)

    return {
        "ticker": ticker,
        "last_observed_date": str(last_date.date()),
        "last_close": previous_close,
        "predicted_open": predicted_open,
        "predicted_high": predicted_high,
        "predicted_low": predicted_low,
        "predicted_close": predicted_close,
        "predicted_close_return_pct": float(predicted_returns[3]) * 100.0,
        "bullish_probability_uncalibrated": bullish_probability,
        "model_direction": "Bullish" if bullish_probability >= 0.5 else "Bearish",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast the next daily candle for one ticker.")
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. RELIANCE.NS or AAPL")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    result = predict_next_candle(args.ticker.upper(), args.checkpoint)
    print("\nNext-candle model output (research estimate, not a trading signal)")
    print("-" * 62)
    print(f"Ticker:              {result['ticker']}")
    print(f"Last observed date:  {result['last_observed_date']}")
    print(f"Last close:          {result['last_close']:.4f}")
    print(f"Predicted open:      {result['predicted_open']:.4f}")
    print(f"Predicted high:      {result['predicted_high']:.4f}")
    print(f"Predicted low:       {result['predicted_low']:.4f}")
    print(f"Predicted close:     {result['predicted_close']:.4f}")
    print(f"Predicted close ret: {result['predicted_close_return_pct']:+.3f}%")
    print(f"Model direction:     {result['model_direction']}")
    print(
        "Bullish score:      "
        f"{result['bullish_probability_uncalibrated']:.1%} (uncalibrated; do not treat as true probability)"
    )


if __name__ == "__main__":
    main()
