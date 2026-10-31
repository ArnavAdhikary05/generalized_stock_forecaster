from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    CHECKPOINT_DIR, DEFAULT_CHECKPOINT, FEATURE_NAMES, MODEL_CONFIG,
    RANDOM_SEED, SEQUENCE_LENGTH, TEST_START, TRAIN_END,
    UNSEEN_TEST_START, VALIDATION_END, VALIDATION_START,
)
from .data import load_all_csvs
from .dataset import WindowDataset, fit_scalers, make_sample_lists
from .features import prepare_ticker
from .model import CandleTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_bundle(sequence_length: int):
    raw_data = load_all_csvs()
    prepared = {}
    for ticker, frame in raw_data.items():
        item = prepare_ticker(frame)
        if len(item["features"]) > sequence_length + 20:
            prepared[ticker] = item
        else:
            print(f"Skipping {ticker}: not enough feature rows after preprocessing.")

    splits = make_sample_lists(
        prepared,
        sequence_length=sequence_length,
        train_end=TRAIN_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        unseen_test_start=UNSEEN_TEST_START,
        seed=RANDOM_SEED,
    )
    train_samples, val_samples, seen_samples, unseen_samples, train_tickers, holdout_tickers = splits
    if not train_samples or not val_samples:
        raise RuntimeError(
            "Training or validation split is empty. Check that downloaded data spans "
            f"the required periods (training through {TRAIN_END}; validation {VALIDATION_START} to {VALIDATION_END})."
        )

    means = fit_scalers(prepared, train_tickers, train_samples, TRAIN_END)
    return prepared, train_samples, val_samples, seen_samples, unseen_samples, train_tickers, holdout_tickers, means


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    regression_loss = nn.HuberLoss(delta=1.0)
    direction_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_items = 0

    for features, targets, directions in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        directions = directions.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            predictions, direction_logits = model(features)
            loss_reg = regression_loss(predictions, targets)
            loss_dir = direction_loss(direction_logits, directions)
            loss = loss_reg + 0.20 * loss_dir
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def evaluate(model, loader, device, target_mean, target_std) -> dict[str, float]:
    model.eval()
    predicted_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []
    direction_probability_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for features, targets, _directions in loader:
            features = features.to(device)
            pred_scaled, direction_logits = model(features)
            pred = pred_scaled.cpu().numpy() * target_std + target_mean
            actual = targets.numpy() * target_std + target_mean
            predicted_chunks.append(pred)
            actual_chunks.append(actual)
            direction_probability_chunks.append(torch.sigmoid(direction_logits).cpu().numpy())

    if not predicted_chunks:
        return {"samples": 0.0}

    predicted = np.concatenate(predicted_chunks, axis=0)
    actual = np.concatenate(actual_chunks, axis=0)
    direction_prob = np.concatenate(direction_probability_chunks, axis=0)
    direction_actual = actual[:, 3] > 0
    direction_from_close = predicted[:, 3] > 0

    return {
        "samples": float(len(actual)),
        "ohlc_mae_bps": float(np.mean(np.abs(predicted - actual)) * 10000.0),
        "close_mae_bps": float(np.mean(np.abs(predicted[:, 3] - actual[:, 3])) * 10000.0),
        "close_direction_accuracy": float(np.mean(direction_from_close == direction_actual)),
        "direction_head_accuracy": float(np.mean((direction_prob >= 0.5) == direction_actual)),
        "direction_head_mean_probability": float(np.mean(direction_prob)),
        # Naive baseline: predict zero return for the next close (same as previous close).
        "zero_return_close_mae_bps": float(np.mean(np.abs(actual[:, 3])) * 10000.0),
        "zero_return_direction_accuracy": float(np.mean((np.zeros_like(actual[:, 3]) > 0) == direction_actual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-stock daily candle Transformer.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (
        data, train_samples, val_samples, seen_samples, unseen_samples,
        train_tickers, holdout_tickers, scalers,
    ) = prepare_bundle(args.sequence_length)
    feature_mean, feature_std, target_mean, target_std = scalers

    print(f"Tickers loaded: {len(data)}")
    print(f"Training tickers: {len(train_tickers)} | held-out tickers: {len(holdout_tickers)}")
    print(
        f"Samples — train: {len(train_samples):,}, validation: {len(val_samples):,}, "
        f"seen-stock test: {len(seen_samples):,}, unseen-stock test: {len(unseen_samples):,}"
    )
    print("Held-out tickers:", ", ".join(holdout_tickers))

    train_ds = WindowDataset(data, train_samples, *scalers, args.sequence_length)
    val_ds = WindowDataset(data, val_samples, *scalers, args.sequence_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model_config = copy.deepcopy(MODEL_CONFIG)
    model = CandleTransformer(
        input_features=len(FEATURE_NAMES),
        sequence_length=args.sequence_length,
        **model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train={train_loss:.5f} "
            f"val={val_loss:.5f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "model_config": model_config,
        "sequence_length": args.sequence_length,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_tickers": train_tickers,
        "holdout_tickers": holdout_tickers,
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "seed": RANDOM_SEED,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"\nSaved best model: {args.checkpoint}")

    history_path = args.checkpoint.with_suffix(".history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    # Report out-of-sample metrics on both previously seen and unseen symbols.
    if seen_samples:
        seen_ds = WindowDataset(data, seen_samples, *scalers, args.sequence_length)
        seen_loader = DataLoader(seen_ds, batch_size=args.batch_size, shuffle=False)
        print("\nSeen-stock future test:")
        print(json.dumps(evaluate(model, seen_loader, device, target_mean, target_std), indent=2))
    else:
        print("\nNo seen-stock test samples found for the configured test period.")

    if unseen_samples:
        unseen_ds = WindowDataset(data, unseen_samples, *scalers, args.sequence_length)
        unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, shuffle=False)
        print("\nUnseen-stock test (tickers excluded from training):")
        print(json.dumps(evaluate(model, unseen_loader, device, target_mean, target_std), indent=2))
    else:
        print("\nNo unseen-stock test samples found for the configured test period.")


if __name__ == "__main__":
    main()
