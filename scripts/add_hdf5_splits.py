"""Add deterministic train/val/test split datasets to an existing mock HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_indices(n_samples: int, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[str, np.ndarray]:
    fractions = np.asarray([train_frac, val_frac, test_frac], dtype=float)
    if np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError("Split fractions must be non-negative and sum to a positive value.")
    fractions = fractions / fractions.sum()

    n_train = int(round(n_samples * fractions[0]))
    n_val = int(round(n_samples * fractions[1]))
    n_train = min(max(n_train, 0), n_samples)
    n_val = min(max(n_val, 0), n_samples - n_train)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    return {
        "train": np.sort(perm[:n_train]).astype(np.int64),
        "val": np.sort(perm[n_train : n_train + n_val]).astype(np.int64),
        "test": np.sort(perm[n_train + n_val :]).astype(np.int64),
    }


def main() -> None:
    args = parse_args()
    path = Path(args.h5)
    with h5py.File(path, "a") as hf:
        existing = ["split", "train_indices", "val_indices", "test_indices"]
        if any(key in hf for key in existing) and not args.overwrite:
            raise RuntimeError(f"{path} already contains split datasets. Pass --overwrite to replace them.")

        for key in existing:
            if key in hf:
                del hf[key]

        n_samples = len(hf["spectra"])
        indices = build_indices(n_samples, args.train_frac, args.val_frac, args.test_frac, args.seed)
        split = np.full(n_samples, SPLIT_CODES["test"], dtype=np.uint8)
        for name, idx in indices.items():
            hf.create_dataset(f"{name}_indices", data=idx)
            split[idx] = SPLIT_CODES[name]

        hf.create_dataset("split", data=split)
        hf.attrs["split_code_json"] = json.dumps(SPLIT_CODES)
        hf.attrs["split_counts_json"] = json.dumps({name: int(len(idx)) for name, idx in indices.items()})

    print(json.dumps({name: int(len(idx)) for name, idx in indices.items()}, indent=2))


if __name__ == "__main__":
    main()
