from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    data: dict[str, dict[str, np.ndarray]]
    train_samples: list[tuple[str, int]]
    validation_samples: list[tuple[str, int]]
    seen_test_samples: list[tuple[str, int]]
    unseen_test_samples: list[tuple[str, int]]
    train_tickers: list[str]
    holdout_tickers: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


class WindowDataset(Dataset):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        samples: list[tuple[str, int]],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.data = data
        self.samples = samples
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ticker, end_index = self.samples[index]
        item = self.data[ticker]
        start_index = end_index - self.sequence_length + 1
        x = item["features"][start_index : end_index + 1]
        y = item["targets"][end_index]
        x = (x - self.feature_mean) / self.feature_std
        y_scaled = (y - self.target_mean) / self.target_std
        direction = np.float32(y[3] > 0.0)
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_scaled, dtype=np.float32)),
            torch.tensor(direction, dtype=torch.float32),
        )


def make_sample_lists(
    data: dict[str, dict[str, np.ndarray]],
    sequence_length: int,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    unseen_test_start: str,
    seed: int,
    holdout_fraction: float = 0.20,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], list[str], list[str]]:
    import random
    import pandas as pd

    tickers = sorted(data)
    if len(tickers) < 5:
        raise RuntimeError("Use at least five tickers so unseen-stock evaluation is meaningful.")

    shuffled = tickers.copy()
    random.Random(seed).shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout_tickers = sorted(shuffled[:holdout_count])
    train_tickers = sorted(shuffled[holdout_count:])

    train_samples: list[tuple[str, int]] = []
    validation_samples: list[tuple[str, int]] = []
    seen_test_samples: list[tuple[str, int]] = []
    unseen_test_samples: list[tuple[str, int]] = []

    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(validation_start)
    val_end_ts = pd.Timestamp(validation_end)
    test_start_ts = pd.Timestamp(test_start)
    unseen_start_ts = pd.Timestamp(unseen_test_start)

    for ticker, item in data.items():
        if len(item["features"]) <= sequence_length:
            continue
        is_holdout = ticker in holdout_tickers
        for end_index in range(sequence_length - 1, len(item["features"])):
            target_date = pd.Timestamp(item["target_dates"][end_index])
            sample = (ticker, end_index)
            if not is_holdout:
                if target_date <= train_end_ts:
                    train_samples.append(sample)
                elif val_start_ts <= target_date <= val_end_ts:
                    validation_samples.append(sample)
                elif target_date >= test_start_ts:
                    seen_test_samples.append(sample)
            elif target_date >= unseen_start_ts:
                unseen_test_samples.append(sample)

    return (
        train_samples, validation_samples, seen_test_samples, unseen_test_samples,
        train_tickers, holdout_tickers,
    )


def fit_scalers(
    data: dict[str, dict[str, np.ndarray]],
    train_tickers: list[str],
    train_samples: list[tuple[str, int]],
    train_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    cutoff = pd.Timestamp(train_end)
    feature_chunks: list[np.ndarray] = []
    for ticker in train_tickers:
        item = data[ticker]
        mask = pd.to_datetime(item["dates"]) <= cutoff
        if mask.any():
            feature_chunks.append(item["features"][mask])
    if not feature_chunks:
        raise RuntimeError("No training feature rows found. Check the date range and downloaded data.")

    train_features = np.concatenate(feature_chunks, axis=0).astype(np.float64)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    train_targets = np.stack([data[ticker]["targets"][index] for ticker, index in train_samples]).astype(np.float64)
    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0)
    target_std[target_std < 1e-6] = 1.0

    return tuple(
        array.astype(np.float32)
        for array in (feature_mean, feature_std, target_mean, target_std)
    )  # type: ignore[return-value]
