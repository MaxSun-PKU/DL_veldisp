"""
Evaluate a trained HybridAGNNet on a held-out HDF5 split.

This is intended for the explicit test split written by
generate_sdss_quasar_mocks.py, so test metrics are not mixed with the training
or validation subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader_v2 import MockAGNDataset
from physmodel import HybridAGNNet
from train_hybrid_sdss import load_h5_assets, load_h5_physical_config, velocity_pixel_scale


AUTO_CHANNELS = {
    1: ("flux",),
    3: ("flux", "ivar", "mask"),
    4: ("flux", "ivar", "mask", "sigma_mask"),
    5: ("flux", "ivar", "mask", "sigma_flux", "sigma_mask"),
    6: ("flux", "ivar", "mask", "sigma_flux", "inst_sigma", "sigma_mask"),
    9: ("flux", "ivar", "mask", "sigma_flux", "inst_sigma", "region_cat", "region_fe", "region_cahk", "sigma_mask"),
    10: ("flux", "ivar", "mask", "sigma_flux", "inst_sigma", "region_cat", "region_fe", "region_cahk", "agn_log_l5100", "sigma_mask"),
}
SIGMA_SCALE_KMS = 120.0
DEFAULT_SCIENCE_SIGMA_BIN_EDGES = (30.0, 60.0, 80.0, 120.0, 180.0, 250.0, 400.0, 700.0)



def parse_float_sequence(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None or str(text).strip() == "":
        return default
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or not np.isfinite(values).all():
        raise ValueError(f"Invalid float sequence: {text!r}")
    return values


def summarize_fixed_bins(true_sigma: np.ndarray, pred_sigma: np.ndarray, bin_edges: tuple[float, ...], pred_sigma_err=None):
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (true_sigma >= float(lo)) & (true_sigma < float(hi))
        if not np.any(mask):
            continue
        err = pred_sigma_err[mask] if pred_sigma_err is not None else None
        row = summarize(true_sigma[mask], pred_sigma[mask], pred_sigma_err=err)
        row.update({"bin_lo": float(lo), "bin_hi": float(hi)})
        rows.append(row)
    return rows



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="figures/sdss_test_evaluation.pdf")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--channels", default="auto", help="Comma-separated channels, or auto to infer from checkpoint.")
    parser.add_argument("--calibrate-split", default="val", help="Split used for affine sigma calibration; pass none to disable.")
    parser.add_argument("--calibration-json-out", default=None, help="Optional JSON file containing affine/error calibration for real-SDSS inference.")
    parser.add_argument("--low-uncertainty-percentile", type=float, default=35.0)
    parser.add_argument("--quality-min-host", type=float, default=0.45)
    parser.add_argument("--quality-min-sigma-mask-fraction", type=float, default=0.015)
    parser.add_argument("--quality-min-valid-fraction", type=float, default=0.20)
    parser.add_argument("--quality-min-resolution-ratio", type=float, default=0.90)
    parser.add_argument("--quality-high-conf-min-resolution-ratio", type=float, default=1.25)
    parser.add_argument("--quality-max-fit-recon-mae", type=float, default=0.18)
    parser.add_argument("--quality-max-sigma-recon-mae", type=float, default=0.16)
    parser.add_argument("--quality-max-sigma-residual-to-feature", type=float, default=0.85)
    parser.add_argument("--quality-min-host-sigma-fraction", type=float, default=0.42)
    parser.add_argument("--quality-min-absorption-depth", type=float, default=0.025)
    parser.add_argument("--real-sdss-systematic-frac", type=float, default=0.12)
    parser.add_argument("--real-sdss-lowres-extra-frac", type=float, default=0.25)
    parser.add_argument("--quality-risk-calibration-max-samples", type=int, default=6000)
    parser.add_argument("--quality-risk-neighbors", type=int, default=96)
    parser.add_argument("--quality-risk-gold-max-expected-error", type=float, default=30.0)
    parser.add_argument("--quality-risk-silver-max-expected-error", type=float, default=45.0)
    parser.add_argument("--quality-risk-seed", type=int, default=20260630)
    parser.add_argument("--science-sigma-bin-edges", default="30,60,80,120,180,250,400,700")
    return parser.parse_args()


def summarize(
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray | None = None,
) -> dict[str, float]:
    residual = pred_sigma - true_sigma
    baseline = np.full_like(true_sigma, np.mean(true_sigma))
    corr = float(np.corrcoef(true_sigma, pred_sigma)[0, 1]) if np.std(pred_sigma) > 0 and np.std(true_sigma) > 0 else 0.0
    metrics = {
        "n": int(true_sigma.size),
        "rmse_kms": float(np.sqrt(np.mean(residual**2))),
        "baseline_mean_rmse_kms": float(np.sqrt(np.mean((baseline - true_sigma) ** 2))),
        "mae_kms": float(np.mean(np.abs(residual))),
        "bias_kms": float(np.mean(residual)),
        "median_abs_error_kms": float(np.median(np.abs(residual))),
        "scatter_kms": float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
        "sigma_corr": corr,
        "true_sigma_std": float(np.std(true_sigma)),
        "pred_sigma_std": float(np.std(pred_sigma)),
        "pred_sigma_min": float(np.min(pred_sigma)),
        "pred_sigma_max": float(np.max(pred_sigma)),
    }
    if pred_sigma_err is not None:
        err = np.asarray(pred_sigma_err, dtype=float)
        valid = np.isfinite(err) & (err > 0)
        if np.any(valid):
            abs_resid = np.abs(residual[valid])
            err_valid = err[valid]
            norm_resid = residual[valid] / err_valid
            metrics.update(
                {
                    "median_pred_sigma_err_kms": float(np.median(err_valid)),
                    "mean_pred_sigma_err_kms": float(np.mean(err_valid)),
                    "coverage_1sigma": float(np.mean(abs_resid <= err_valid)),
                    "coverage_2sigma": float(np.mean(abs_resid <= 2.0 * err_valid)),
                    "normalized_resid_std": float(np.std(norm_resid)),
                    "abs_resid_err_corr": float(np.corrcoef(abs_resid, err_valid)[0, 1])
                    if np.std(abs_resid) > 0 and np.std(err_valid) > 0
                    else 0.0,
                }
            )
    return metrics


def fit_affine_calibration(pred_sigma: np.ndarray, true_sigma: np.ndarray) -> tuple[float, float]:
    if pred_sigma.size < 2 or np.std(pred_sigma) <= 0:
        return 1.0, 0.0
    design = np.vstack([pred_sigma, np.ones_like(pred_sigma)]).T
    slope, intercept = np.linalg.lstsq(design, true_sigma, rcond=None)[0]
    return float(slope), float(intercept)


def apply_error_calibration(pred_sigma_err: np.ndarray, scale: float, floor_kms: float) -> np.ndarray:
    err = np.asarray(pred_sigma_err, dtype=float)
    return np.sqrt((float(scale) * err) ** 2 + float(floor_kms) ** 2)


def fit_error_calibration(
    residual: np.ndarray,
    pred_sigma_err: np.ndarray,
    *,
    target_coverage_1sigma: float = 0.682689492,
    target_coverage_2sigma: float = 0.954499736,
    max_floor_kms: float = 220.0,
) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    pred_sigma_err = np.asarray(pred_sigma_err, dtype=float)
    finite = np.isfinite(residual) & np.isfinite(pred_sigma_err) & (pred_sigma_err > 0)
    if np.count_nonzero(finite) < 20:
        return {"scale": 1.0, "floor_kms": 0.0, "coverage_1sigma": float("nan"), "coverage_2sigma": float("nan")}

    abs_resid = np.abs(residual[finite])
    base_err = pred_sigma_err[finite]
    best: dict[str, float] | None = None
    floors = np.linspace(0.0, max_floor_kms, 221)
    for floor in floors:
        candidate_err = np.sqrt(base_err**2 + floor**2)
        ratios = abs_resid / np.clip(candidate_err, 1e-6, None)
        scale = float(np.nanquantile(ratios, target_coverage_1sigma))
        scale = max(scale, 1e-3)
        calibrated_err = scale * candidate_err
        coverage_1 = float(np.mean(abs_resid <= calibrated_err))
        coverage_2 = float(np.mean(abs_resid <= 2.0 * calibrated_err))
        norm_std = float(np.std(residual[finite] / np.clip(calibrated_err, 1e-6, None)))
        score = (
            (coverage_1 - target_coverage_1sigma) ** 2
            + 0.75 * (coverage_2 - target_coverage_2sigma) ** 2
            + 0.02 * (norm_std - 1.0) ** 2
            + 1e-5 * floor
        )
        row = {
            "scale": scale,
            "floor_kms": float(floor),
            "coverage_1sigma": coverage_1,
            "coverage_2sigma": coverage_2,
            "normalized_resid_std": norm_std,
            "score": float(score),
        }
        if best is None or row["score"] < best["score"]:
            best = row
    return best if best is not None else {"scale": 1.0, "floor_kms": 0.0}


def summarize_subset(
    name: str,
    mask: np.ndarray,
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray | None = None,
) -> dict[str, float | str]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(true_sigma) & np.isfinite(pred_sigma)
    if pred_sigma_err is not None:
        mask &= np.isfinite(pred_sigma_err) & (pred_sigma_err > 0)
    if np.count_nonzero(mask) == 0:
        return {"subset": name, "n": 0}
    row = summarize(true_sigma[mask], pred_sigma[mask], pred_sigma_err[mask] if pred_sigma_err is not None else None)
    row["subset"] = name
    return row


def robust_absorption_depth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return 0.0
    return float(np.nanpercentile(values, 80.0) - np.nanpercentile(values, 20.0))


def batch_reconstruction_diagnostics(
    batch_x: torch.Tensor,
    recon: torch.Tensor,
    agn_part: torch.Tensor,
    stellar_part: torch.Tensor,
) -> dict[str, np.ndarray]:
    x = batch_x.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()
    agn_np = agn_part.detach().cpu().numpy()
    stellar_np = stellar_part.detach().cpu().numpy()
    flux = x[:, 0, :]
    fit_mask = x[:, 2, :] > 0.5 if x.shape[1] > 2 else np.isfinite(flux)
    sigma_mask = x[:, -1, :] > 0.5
    resid = flux - recon_np
    out = {
        "fit_recon_mae": [],
        "sigma_recon_mae": [],
        "sigma_recon_resid_std": [],
        "pred_host_sigma_fraction": [],
        "host_absorption_depth_proxy": [],
        "flux_absorption_depth_proxy": [],
        "sigma_residual_to_feature": [],
    }
    for i in range(x.shape[0]):
        valid_fit = fit_mask[i] & np.isfinite(flux[i]) & np.isfinite(recon_np[i])
        valid_sigma = sigma_mask[i] & np.isfinite(flux[i]) & np.isfinite(recon_np[i])
        fit_mae = float(np.nanmean(np.abs(resid[i, valid_fit]))) if np.any(valid_fit) else np.nan
        sigma_mae = float(np.nanmean(np.abs(resid[i, valid_sigma]))) if np.any(valid_sigma) else np.nan
        sigma_std = float(np.nanstd(resid[i, valid_sigma])) if np.any(valid_sigma) else np.nan
        host_power = float(np.nansum(np.abs(stellar_np[i, valid_sigma]))) if np.any(valid_sigma) else 0.0
        agn_power = float(np.nansum(np.abs(agn_np[i, valid_sigma]))) if np.any(valid_sigma) else 0.0
        host_sigma_fraction = host_power / max(host_power + agn_power, 1e-6)
        host_depth = robust_absorption_depth(stellar_np[i, valid_sigma]) if np.any(valid_sigma) else 0.0
        flux_depth = robust_absorption_depth(flux[i, valid_sigma]) if np.any(valid_sigma) else 0.0
        residual_to_feature = sigma_mae / max(flux_depth, 1e-4) if np.isfinite(sigma_mae) else np.inf
        out["fit_recon_mae"].append(fit_mae)
        out["sigma_recon_mae"].append(sigma_mae)
        out["sigma_recon_resid_std"].append(sigma_std)
        out["pred_host_sigma_fraction"].append(float(np.clip(host_sigma_fraction, 0.0, 1.0)))
        out["host_absorption_depth_proxy"].append(host_depth)
        out["flux_absorption_depth_proxy"].append(flux_depth)
        out["sigma_residual_to_feature"].append(float(residual_to_feature))
    return {key: np.asarray(values, dtype=np.float32) for key, values in out.items()}


QUALITY_RISK_FEATURES = (
    "pred_sigma_kms",
    "pred_sigma_err_kms",
    "pred_host_fraction",
    "median_snr",
    "sigma_mask_fraction",
    "valid_fraction",
    "pred_sigma_resolution_ratio",
    "fit_recon_mae",
    "sigma_recon_mae",
    "sigma_residual_to_feature",
    "pred_host_sigma_fraction",
    "flux_absorption_depth_proxy",
    "training_edge_fraction",
)


def _finite_array(data: dict[str, np.ndarray], key: str, length: int, default: float) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        if arr.size == length:
            return arr
    return np.full(length, float(default), dtype=float)


def quality_risk_feature_matrix(
    data: dict[str, np.ndarray],
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray,
    *,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    n = len(pred_sigma)
    pred = np.asarray(pred_sigma, dtype=float).reshape(-1)
    err = np.asarray(pred_sigma_err, dtype=float).reshape(-1)
    host = _finite_array(data, "pred_host", n, 0.0)
    snr = _finite_array(data, "snr", n, 0.0)
    sigma_mask_fraction = _finite_array(data, "sigma_mask_fraction", n, 0.0)
    valid_fraction = _finite_array(data, "valid_fraction", n, 0.0)
    inst = _finite_array(data, "sigma_inst_kms", n, 70.0)
    pred_resolution_ratio = pred / np.clip(inst, 1e-6, None)
    fit_recon_mae = _finite_array(data, "fit_recon_mae", n, 9.0)
    sigma_recon_mae = _finite_array(data, "sigma_recon_mae", n, 9.0)
    sigma_residual_to_feature = _finite_array(data, "sigma_residual_to_feature", n, 20.0)
    host_sigma_fraction = _finite_array(data, "pred_host_sigma_fraction", n, 0.0)
    flux_depth = _finite_array(data, "flux_absorption_depth_proxy", n, 0.0)
    label_pad = max(10.0, 0.05 * (float(sigma_max) - float(sigma_min)))
    edge_distance = np.minimum(np.abs(pred - float(sigma_min)), np.abs(float(sigma_max) - pred))
    training_edge_fraction = np.clip(1.0 - edge_distance / label_pad, 0.0, 1.0)
    matrix = np.vstack(
        [
            pred / 300.0,
            np.log1p(np.clip(err, 0.0, None)) / np.log(101.0),
            np.clip(host, 0.0, 1.0),
            np.log1p(np.clip(snr, 0.0, None)) / np.log(101.0),
            np.clip(sigma_mask_fraction, 0.0, 1.0),
            np.clip(valid_fraction, 0.0, 1.0),
            np.clip(pred_resolution_ratio, 0.0, 5.0) / 5.0,
            np.log1p(np.clip(fit_recon_mae, 0.0, 20.0)) / np.log(21.0),
            np.log1p(np.clip(sigma_recon_mae, 0.0, 20.0)) / np.log(21.0),
            np.log1p(np.clip(sigma_residual_to_feature, 0.0, 50.0)) / np.log(51.0),
            np.clip(host_sigma_fraction, 0.0, 1.0),
            np.log1p(np.clip(flux_depth, 0.0, 20.0)) / np.log(21.0),
            training_edge_fraction,
        ]
    ).T
    return np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def fit_quality_risk_model(
    calibration_data: dict[str, np.ndarray],
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray,
    residual: np.ndarray,
    *,
    sigma_min: float,
    sigma_max: float,
    max_samples: int,
    n_neighbors: int,
    seed: int,
) -> dict[str, object]:
    features = quality_risk_feature_matrix(
        calibration_data,
        pred_sigma,
        pred_sigma_err,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    abs_residual = np.abs(np.asarray(residual, dtype=float).reshape(-1))
    finite = np.isfinite(features).all(axis=1) & np.isfinite(abs_residual)
    features = features[finite]
    abs_residual = abs_residual[finite]
    if features.shape[0] == 0:
        return {}
    if max_samples > 0 and features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(features.shape[0], size=max_samples, replace=False))
        features = features[take]
        abs_residual = abs_residual[take]
    center = np.nanmedian(features, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(features - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-4), scale, 1.0)
    normalized = (features - center) / scale
    return {
        "feature_names": list(QUALITY_RISK_FEATURES),
        "feature_center": center.astype(float).tolist(),
        "feature_scale": scale.astype(float).tolist(),
        "feature_matrix": normalized.astype(np.float32).tolist(),
        "abs_residual_kms": abs_residual.astype(np.float32).tolist(),
        "n_calibration_samples": int(features.shape[0]),
        "n_neighbors": int(max(8, min(n_neighbors, features.shape[0]))),
        "note": "Nearest-neighbor empirical risk model trained only on the mock validation split.",
    }


def estimate_quality_risk(
    risk_model: dict[str, object],
    data: dict[str, np.ndarray],
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray,
    *,
    sigma_min: float,
    sigma_max: float,
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    if not risk_model:
        n = len(pred_sigma)
        nan = np.full(n, np.nan, dtype=np.float32)
        return {
            "risk_expected_abs_error_kms": nan,
            "risk_p68_abs_error_kms": nan,
            "risk_p90_abs_error_kms": nan,
            "risk_p_abs_error_le_40": nan,
            "risk_p_abs_error_le_60": nan,
            "risk_reliability_score": nan,
        }
    ref = np.asarray(risk_model.get("feature_matrix", []), dtype=np.float32)
    abs_resid = np.asarray(risk_model.get("abs_residual_kms", []), dtype=np.float32)
    if ref.ndim != 2 or ref.shape[0] == 0 or abs_resid.size != ref.shape[0]:
        n = len(pred_sigma)
        nan = np.full(n, np.nan, dtype=np.float32)
        return {
            "risk_expected_abs_error_kms": nan,
            "risk_p68_abs_error_kms": nan,
            "risk_p90_abs_error_kms": nan,
            "risk_p_abs_error_le_40": nan,
            "risk_p_abs_error_le_60": nan,
            "risk_reliability_score": nan,
        }
    center = np.asarray(risk_model["feature_center"], dtype=np.float32)
    scale = np.asarray(risk_model["feature_scale"], dtype=np.float32)
    target = quality_risk_feature_matrix(
        data,
        pred_sigma,
        pred_sigma_err,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    target = (target - center) / np.clip(scale, 1e-6, None)
    k = int(max(4, min(int(risk_model.get("n_neighbors", 96)), ref.shape[0])))
    expected, p68, p90, p40, p60, score = [], [], [], [], [], []
    for start in range(0, target.shape[0], chunk_size):
        block = target[start : start + chunk_size]
        dist2 = np.sum((block[:, None, :] - ref[None, :, :]) ** 2, axis=2)
        nn = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        local = abs_resid[nn]
        expected.append(np.mean(local, axis=1))
        p68.append(np.percentile(local, 68.0, axis=1))
        p90.append(np.percentile(local, 90.0, axis=1))
        p40.append(np.mean(local <= 40.0, axis=1))
        p60.append(np.mean(local <= 60.0, axis=1))
        score.append(1.0 / (1.0 + np.mean(local, axis=1) / 40.0))
    return {
        "risk_expected_abs_error_kms": np.concatenate(expected).astype(np.float32),
        "risk_p68_abs_error_kms": np.concatenate(p68).astype(np.float32),
        "risk_p90_abs_error_kms": np.concatenate(p90).astype(np.float32),
        "risk_p_abs_error_le_40": np.concatenate(p40).astype(np.float32),
        "risk_p_abs_error_le_60": np.concatenate(p60).astype(np.float32),
        "risk_reliability_score": np.concatenate(score).astype(np.float32),
    }


def subset_summaries(
    values: np.ndarray,
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    pred_sigma_err: np.ndarray | None = None,
    *,
    label: str,
    n_bins: int = 6,
) -> list[dict[str, float | str]]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & np.isfinite(true_sigma) & np.isfinite(pred_sigma)
    if np.count_nonzero(finite) < n_bins * 5:
        return []

    edges = np.quantile(values[finite], np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return []

    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = finite & (values >= lo) & (values <= hi)
        else:
            mask = finite & (values >= lo) & (values < hi)
        if np.count_nonzero(mask) == 0:
            continue
        err = pred_sigma_err[mask] if pred_sigma_err is not None else None
        row = summarize(true_sigma[mask], pred_sigma[mask], pred_sigma_err=err)
        row.update({"label": label, "bin_lo": float(lo), "bin_hi": float(hi)})
        rows.append(row)
    return rows


def plot_binned_rmse(ax, rows: list[dict[str, float | str]], xlabel: str) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    centers = np.asarray([(float(row["bin_lo"]) + float(row["bin_hi"])) / 2.0 for row in rows])
    widths = np.asarray([float(row["bin_hi"]) - float(row["bin_lo"]) for row in rows])
    rmse = np.asarray([float(row["rmse_kms"]) for row in rows])
    mae = np.asarray([float(row["mae_kms"]) for row in rows])
    bias = np.asarray([float(row["bias_kms"]) for row in rows])
    ax.plot(centers, rmse, marker="o", lw=1.2, label="RMSE")
    ax.plot(centers, mae, marker="s", lw=1.0, label="MAE")
    ax.plot(centers, bias, marker=".", lw=1.0, color="0.35", label="bias")
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlim(centers.min() - 0.55 * widths[0], centers.max() + 0.55 * widths[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("km/s")
    ax.legend(frameon=False, fontsize=8)


def plot_diagnostics(
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    true_host: np.ndarray,
    pred_host: np.ndarray,
    snr: np.ndarray,
    pred_sigma_err: np.ndarray | None,
    metrics: dict[str, float],
    output: Path,
    calibrated_sigma: np.ndarray | None = None,
    redshift: np.ndarray | None = None,
    sigma_mask_fraction: np.ndarray | None = None,
    valid_fraction: np.ndarray | None = None,
    subset_metrics: dict[str, list[dict[str, float | str]]] | None = None,
) -> None:
    residual = pred_sigma - true_sigma
    abs_residual = np.abs(residual)
    sigma_arrays = [true_sigma, pred_sigma]
    if calibrated_sigma is not None:
        sigma_arrays.append(calibrated_sigma)
    sigma_min = min(np.nanmin(array) for array in sigma_arrays)
    sigma_max = max(np.nanmax(array) for array in sigma_arrays)
    pad = 0.04 * (sigma_max - sigma_min)

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
        sc = axes[0, 0].scatter(
            true_sigma,
            pred_sigma,
            c=true_host,
            s=8,
            alpha=0.45,
            cmap="viridis_r",
            edgecolors="none",
            label="raw prediction",
        )
        if calibrated_sigma is not None:
            axes[0, 0].scatter(
                true_sigma,
                calibrated_sigma,
                s=8,
                alpha=0.20,
                color="tab:orange",
                edgecolors="none",
                label="affine calibrated",
            )
        axes[0, 0].plot(
            [sigma_min - pad, sigma_max + pad],
            [sigma_min - pad, sigma_max + pad],
            color="0.2",
            lw=1.0,
            label="1:1 relation",
        )
        axes[0, 0].set_xlim(sigma_min - pad, sigma_max + pad)
        axes[0, 0].set_ylim(sigma_min - pad, sigma_max + pad)
        axes[0, 0].set_xlabel("True sigma [km/s]")
        axes[0, 0].set_ylabel("Predicted sigma [km/s]")
        axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
        fig.colorbar(sc, ax=axes[0, 0], label="Host fraction")

        err_color = pred_sigma_err if pred_sigma_err is not None else true_sigma
        err_label = "Predicted sigma error [km/s]" if pred_sigma_err is not None else "True sigma [km/s]"
        sc_err = axes[0, 1].scatter(snr, residual, c=err_color, s=8, alpha=0.45, cmap="magma", edgecolors="none")
        axes[0, 1].axhline(0.0, color="0.25", lw=1.0)
        axes[0, 1].set_xlabel("Mock S/N")
        axes[0, 1].set_ylabel("Delta sigma [km/s]")
        fig.colorbar(sc_err, ax=axes[0, 1], label=err_label)

        axes[1, 0].hist(residual, bins=50, histtype="stepfilled", color="0.35", alpha=0.75)
        axes[1, 0].axvline(0.0, color="0.15", lw=1.0)
        axes[1, 0].set_xlabel("Delta sigma [km/s]")
        axes[1, 0].set_ylabel("Number")

        axes[1, 1].scatter(true_host, residual, c=pred_host, s=8, alpha=0.45, cmap="cividis", edgecolors="none")
        axes[1, 1].axhline(0.0, color="0.25", lw=1.0)
        axes[1, 1].set_xlabel("True host fraction")
        axes[1, 1].set_ylabel("Delta sigma [km/s]")

        fig.suptitle(
            "Test split: "
            f"N={metrics['n']}, RMSE={metrics['rmse_kms']:.1f} km/s, "
            f"MAE={metrics['mae_kms']:.1f} km/s, bias={metrics['bias_kms']:.1f} km/s, "
            f"corr={metrics['sigma_corr']:.2f}",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
        axes[0, 0].scatter(true_sigma, residual, c=true_host, s=8, alpha=0.45, cmap="viridis_r", edgecolors="none")
        axes[0, 0].axhline(0.0, color="0.25", lw=1.0)
        axes[0, 0].set_xlabel("True sigma [km/s]")
        axes[0, 0].set_ylabel("Delta sigma [km/s]")

        if redshift is not None:
            axes[0, 1].scatter(redshift, residual, c=true_host, s=8, alpha=0.45, cmap="viridis_r", edgecolors="none")
            axes[0, 1].axhline(0.0, color="0.25", lw=1.0)
            axes[0, 1].set_xlabel("Redshift")
            axes[0, 1].set_ylabel("Delta sigma [km/s]")
        else:
            axes[0, 1].set_axis_off()

        if sigma_mask_fraction is not None:
            axes[1, 0].scatter(sigma_mask_fraction, abs_residual, c=true_sigma, s=8, alpha=0.45, cmap="magma", edgecolors="none")
            axes[1, 0].set_xlabel("Sigma-window pixel fraction")
            axes[1, 0].set_ylabel("|Delta sigma| [km/s]")
        else:
            axes[1, 0].set_axis_off()

        if pred_sigma_err is not None:
            axes[1, 1].scatter(pred_sigma_err, abs_residual, c=true_host, s=8, alpha=0.45, cmap="viridis_r", edgecolors="none")
            limit = float(np.nanpercentile(np.concatenate([pred_sigma_err, abs_residual]), 98))
            axes[1, 1].plot([0, limit], [0, limit], color="0.25", lw=1.0, label="1:1")
            axes[1, 1].plot([0, limit], [0, 2.0 * limit], color="0.55", lw=0.8, ls="--", label="2sigma")
            axes[1, 1].set_xlim(0, limit)
            axes[1, 1].set_ylim(0, min(2.1 * limit, float(np.nanpercentile(abs_residual, 99.5))))
            axes[1, 1].set_xlabel("Predicted sigma error [km/s]")
            axes[1, 1].set_ylabel("|Delta sigma| [km/s]")
            axes[1, 1].legend(frameon=False, fontsize=8)
        else:
            axes[1, 1].set_axis_off()

        fig.suptitle("Residual diagnostics by label and predicted uncertainty", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        if subset_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
            plot_binned_rmse(axes[0, 0], subset_metrics.get("true_sigma", []), "True sigma [km/s]")
            plot_binned_rmse(axes[0, 1], subset_metrics.get("true_host", []), "True host fraction")
            plot_binned_rmse(axes[1, 0], subset_metrics.get("snr", []), "S/N")
            plot_binned_rmse(axes[1, 1], subset_metrics.get("redshift", []), "Redshift")
            fig.suptitle("Binned recovery performance", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    science_bin_edges = parse_float_sequence(args.science_sigma_bin_edges, DEFAULT_SCIENCE_SIGMA_BIN_EDGES)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wave_grid, template_library, fe_template, _, _, sigma_min, sigma_max = load_h5_assets(args.h5)
    physical_config = load_h5_physical_config(args.h5)
    state = torch.load(args.model, map_location=device)
    if args.channels == "auto":
        first_conv = state.get("backbone.0.weight")
        input_channels = int(first_conv.shape[1]) if first_conv is not None else 3
        channels = AUTO_CHANNELS.get(input_channels)
        if channels is None:
            raise ValueError(f"Cannot infer channel names for checkpoint with {input_channels} input channels.")
    else:
        channels = tuple(item.strip() for item in args.channels.split(",") if item.strip())
    model = HybridAGNNet(
        input_dim=len(wave_grid),
        template_library=template_library,
        dv_per_pixel=velocity_pixel_scale(wave_grid),
        input_channels=len(channels),
        wave_grid=wave_grid,
        fe_template=fe_template,
        instrument_sigma_kms=physical_config["instrument_sigma_kms"],
        template_sigma_inst_kms=physical_config["template_sigma_inst_kms"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    def collect_predictions(split: str):
        dataset = MockAGNDataset(
            args.h5,
            channels=channels,
            flux_normalization=None,
            return_sfh=False,
            split=split,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        true_sigma, pred_sigma, true_host, pred_host, snr_values, sigma_err_values = [], [], [], [], [], []
        diag_values: dict[str, list[np.ndarray]] = {
            "fit_recon_mae": [],
            "sigma_recon_mae": [],
            "sigma_recon_resid_std": [],
            "pred_host_sigma_fraction": [],
            "host_absorption_depth_proxy": [],
            "flux_absorption_depth_proxy": [],
            "sigma_residual_to_feature": [],
        }
        with torch.no_grad():
            for batch_x, batch_sigma, batch_host, batch_snr in tqdm(loader, desc=f"Evaluating {split}"):
                batch_x = batch_x.to(device)
                pred, logvar, _, host, recon, agn_part, stellar_part = model(batch_x)
                pred_err = torch.exp(0.5 * torch.clamp(logvar.squeeze(), min=-6.0, max=4.0)) * SIGMA_SCALE_KMS
                true_sigma.append(batch_sigma.numpy().reshape(-1))
                pred_sigma.append(pred.cpu().numpy().reshape(-1))
                true_host.append(batch_host.numpy().reshape(-1))
                pred_host.append(host.cpu().numpy().reshape(-1))
                snr_values.append(batch_snr.numpy().reshape(-1))
                sigma_err_values.append(pred_err.cpu().numpy().reshape(-1))
                diagnostics = batch_reconstruction_diagnostics(batch_x, recon, agn_part, stellar_part)
                for key, values in diagnostics.items():
                    diag_values[key].append(values)
        indices = dataset.indices if dataset.indices is not None else np.arange(len(dataset), dtype=np.int64)
        extras: dict[str, np.ndarray] = {}
        with h5py.File(args.h5, "r") as hf:
            for key in (
                "redshift",
                "mass_weighted_age_gyr",
                "mass_weighted_logzsol",
                "region_code",
                "sigma_inst_kms",
                "sigma_observed_eff_kms",
                "sigma_resolution_ratio",
                "cat_contamination_ew",
                "ca_hk_mask_fraction",
                "fe_mask_fraction",
                "cat_mask_fraction",
            ):
                if key in hf:
                    extras[key] = hf[key][indices]
            if "sigma_mask" in hf:
                extras["sigma_mask_fraction"] = hf["sigma_mask"][indices].mean(axis=1)
            if "valid_pixels" in hf:
                extras["valid_fraction"] = hf["valid_pixels"][indices].mean(axis=1)
        pred_concat = np.concatenate(pred_sigma)
        extras.update({key: np.concatenate(values) for key, values in diag_values.items() if values})
        if "sigma_inst_kms" in extras:
            extras["pred_sigma_resolution_ratio"] = pred_concat / np.clip(np.asarray(extras["sigma_inst_kms"], dtype=float), 1e-6, None)
        return {
            "true_sigma": np.concatenate(true_sigma),
            "pred_sigma": pred_concat,
            "true_host": np.concatenate(true_host),
            "pred_host": np.concatenate(pred_host),
            "snr": np.concatenate(snr_values),
            "pred_sigma_err": np.concatenate(sigma_err_values),
            **extras,
        }

    test = collect_predictions(args.split)
    true_sigma = test["true_sigma"]
    pred_sigma = test["pred_sigma"]
    true_host = test["true_host"]
    pred_host = test["pred_host"]
    snr_values = test["snr"]
    pred_sigma_err = test["pred_sigma_err"]
    metrics = summarize(true_sigma, pred_sigma, pred_sigma_err=pred_sigma_err)
    subset_metrics = {
        "true_sigma": subset_summaries(true_sigma, true_sigma, pred_sigma, pred_sigma_err, label="true_sigma"),
        "true_host": subset_summaries(true_host, true_sigma, pred_sigma, pred_sigma_err, label="true_host"),
        "snr": subset_summaries(snr_values, true_sigma, pred_sigma, pred_sigma_err, label="snr"),
    }
    if "redshift" in test:
        subset_metrics["redshift"] = subset_summaries(
            test["redshift"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="redshift",
        )
    if "sigma_mask_fraction" in test:
        subset_metrics["sigma_mask_fraction"] = subset_summaries(
            test["sigma_mask_fraction"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="sigma_mask_fraction",
        )
    if "valid_fraction" in test:
        subset_metrics["valid_fraction"] = subset_summaries(
            test["valid_fraction"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="valid_fraction",
        )
    if "sigma_resolution_ratio" in test:
        subset_metrics["sigma_resolution_ratio"] = subset_summaries(
            test["sigma_resolution_ratio"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="sigma_resolution_ratio",
        )
    if "cat_contamination_ew" in test:
        subset_metrics["cat_contamination_ew"] = subset_summaries(
            test["cat_contamination_ew"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="cat_contamination_ew",
        )
    if "mass_weighted_age_gyr" in test:
        subset_metrics["mass_weighted_age_gyr"] = subset_summaries(
            test["mass_weighted_age_gyr"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="mass_weighted_age_gyr",
        )
    if "mass_weighted_logzsol" in test:
        subset_metrics["mass_weighted_logzsol"] = subset_summaries(
            test["mass_weighted_logzsol"],
            true_sigma,
            pred_sigma,
            pred_sigma_err,
            label="mass_weighted_logzsol",
        )
    metrics["subsets"] = subset_metrics
    metrics["science_sigma_bins_raw"] = summarize_fixed_bins(true_sigma, pred_sigma, science_bin_edges, pred_sigma_err=pred_sigma_err)
    calibrated_sigma = None
    calibrated_sigma_err = None
    calibration_payload = None
    if args.calibrate_split.lower() != "none":
        cal = collect_predictions(args.calibrate_split)
        cal_true, cal_pred = cal["true_sigma"], cal["pred_sigma"]
        slope, intercept = fit_affine_calibration(cal_pred, cal_true)
        calibrated_sigma = slope * pred_sigma + intercept
        cal_affine_pred = slope * cal_pred + intercept
        cal_base_err = np.abs(slope) * cal["pred_sigma_err"]
        error_calibration = fit_error_calibration(cal_true - cal_affine_pred, cal_base_err)
        calibrated_cal_err = apply_error_calibration(
            cal_base_err,
            error_calibration["scale"],
            error_calibration["floor_kms"],
        )
        calibrated_sigma_err = apply_error_calibration(
            np.abs(slope) * pred_sigma_err,
            error_calibration["scale"],
            error_calibration["floor_kms"],
        )
        quality_risk_model = fit_quality_risk_model(
            cal,
            cal_pred,
            calibrated_cal_err,
            cal_true - cal_pred,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            max_samples=args.quality_risk_calibration_max_samples,
            n_neighbors=args.quality_risk_neighbors,
            seed=args.quality_risk_seed,
        )
        quality_risk = estimate_quality_risk(
            quality_risk_model,
            test,
            pred_sigma,
            calibrated_sigma_err,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        test.update(quality_risk)
        metrics["calibration_split"] = args.calibrate_split
        metrics["calibration_slope"] = slope
        metrics["calibration_intercept"] = intercept
        metrics.update({f"calibrated_{key}": value for key, value in summarize(true_sigma, calibrated_sigma).items() if key != "n"})
        calibrated_error_metrics = summarize(true_sigma, calibrated_sigma, pred_sigma_err=calibrated_sigma_err)
        metrics["science_sigma_bins_calibrated"] = summarize_fixed_bins(true_sigma, calibrated_sigma, science_bin_edges, pred_sigma_err=calibrated_sigma_err)
        metrics["error_calibration"] = {
            "split": args.calibrate_split,
            "scale": float(error_calibration["scale"]),
            "floor_kms": float(error_calibration["floor_kms"]),
            "validation_coverage_1sigma": float(error_calibration.get("coverage_1sigma", float("nan"))),
            "validation_coverage_2sigma": float(error_calibration.get("coverage_2sigma", float("nan"))),
            "validation_normalized_resid_std": float(error_calibration.get("normalized_resid_std", float("nan"))),
            "test_coverage_1sigma": calibrated_error_metrics.get("coverage_1sigma"),
            "test_coverage_2sigma": calibrated_error_metrics.get("coverage_2sigma"),
            "test_median_sigma_err_kms": calibrated_error_metrics.get("median_pred_sigma_err_kms"),
            "test_normalized_resid_std": calibrated_error_metrics.get("normalized_resid_std"),
        }

        low_uncertainty_threshold = float(np.nanpercentile(calibrated_cal_err, args.low_uncertainty_percentile))
        low_uncertainty = calibrated_sigma_err <= low_uncertainty_threshold
        high_uncertainty = calibrated_sigma_err >= float(np.nanpercentile(calibrated_sigma_err, 100.0 - args.low_uncertainty_percentile))
        risk_expected = quality_risk["risk_expected_abs_error_kms"]
        risk_p40 = quality_risk["risk_p_abs_error_le_40"]
        risk_p60 = quality_risk["risk_p_abs_error_le_60"]
        has_risk = np.isfinite(risk_expected)
        risk_gold = (
            has_risk
            & (risk_expected <= float(args.quality_risk_gold_max_expected_error))
            & (risk_p40 >= 0.60)
        )
        risk_silver = (
            has_risk
            & (risk_expected <= float(args.quality_risk_silver_max_expected_error))
            & (risk_p60 >= 0.70)
        )
        high_confidence = low_uncertainty.copy()
        high_confidence &= pred_host >= args.quality_min_host
        if "sigma_mask_fraction" in test:
            high_confidence &= test["sigma_mask_fraction"] >= args.quality_min_sigma_mask_fraction
        if "valid_fraction" in test:
            high_confidence &= test["valid_fraction"] >= args.quality_min_valid_fraction
        resolution_for_quality = test.get("pred_sigma_resolution_ratio", test.get("sigma_resolution_ratio"))
        if resolution_for_quality is not None:
            high_confidence &= resolution_for_quality >= args.quality_min_resolution_ratio
            high_confidence &= resolution_for_quality >= args.quality_high_conf_min_resolution_ratio
        if np.any(has_risk):
            high_confidence &= risk_gold
        real_sdss_max_error = float(max(low_uncertainty_threshold, args.quality_risk_silver_max_expected_error))
        metrics["quality_risk"] = {
            "enabled": bool(np.any(has_risk)),
            "n_calibration_samples": int(quality_risk_model.get("n_calibration_samples", 0)) if quality_risk_model else 0,
            "n_neighbors": int(quality_risk_model.get("n_neighbors", 0)) if quality_risk_model else 0,
            "gold_max_expected_abs_error_kms": float(args.quality_risk_gold_max_expected_error),
            "silver_max_expected_abs_error_kms": float(args.quality_risk_silver_max_expected_error),
        }
        metrics["quality_subsets"] = [
            summarize_subset("all_test_raw", np.ones_like(true_sigma, dtype=bool), true_sigma, pred_sigma, pred_sigma_err),
            summarize_subset("all_test_calibrated", np.ones_like(true_sigma, dtype=bool), true_sigma, calibrated_sigma, calibrated_sigma_err),
            summarize_subset(
                f"lowest_{args.low_uncertainty_percentile:.0f}pct_uncertainty_calibrated",
                low_uncertainty,
                true_sigma,
                calibrated_sigma,
                calibrated_sigma_err,
            ),
            summarize_subset(
                f"highest_{args.low_uncertainty_percentile:.0f}pct_uncertainty_calibrated",
                high_uncertainty,
                true_sigma,
                calibrated_sigma,
                calibrated_sigma_err,
            ),
            summarize_subset("high_confidence_calibrated", high_confidence, true_sigma, calibrated_sigma, calibrated_sigma_err),
            summarize_subset("risk_gold_calibrated", risk_gold, true_sigma, calibrated_sigma, calibrated_sigma_err),
            summarize_subset("risk_silver_calibrated", risk_silver, true_sigma, calibrated_sigma, calibrated_sigma_err),
            summarize_subset("risk_low_calibrated", has_risk & ~risk_silver, true_sigma, calibrated_sigma, calibrated_sigma_err),
        ]
        calibration_payload = {
            "model_path": str(Path(args.model).resolve()),
            "h5_path": str(Path(args.h5).resolve()),
            "channels": list(channels),
            "sigma_affine_slope": slope,
            "sigma_affine_intercept": intercept,
            "sigma_error_scale": float(error_calibration["scale"]),
            "sigma_error_floor_kms": float(error_calibration["floor_kms"]),
            "low_uncertainty_percentile": float(args.low_uncertainty_percentile),
            "low_uncertainty_threshold_kms": low_uncertainty_threshold,
            "science_sigma_bin_edges": list(science_bin_edges),
            "central_value_recommendation": "raw",
            "central_value_note": "Use raw network sigma as the quoted central value for real SDSS unless a domain-specific central-value calibration has been validated. The global affine mapping is retained for diagnostics/error calibration only.",
            "training_label_range": {
                "sigma_min_kms": float(sigma_min),
                "sigma_max_kms": float(sigma_max),
                "note": "Inference reports whether a real spectrum lands outside or near the trained sigma label range.",
            },
            "quality_cuts": {
                "max_calibrated_sigma_err_kms": low_uncertainty_threshold,
                "min_pred_host_fraction": float(args.quality_min_host),
                "min_sigma_mask_fraction": float(args.quality_min_sigma_mask_fraction),
                "min_valid_fraction": float(args.quality_min_valid_fraction),
                "min_sigma_resolution_ratio": float(args.quality_min_resolution_ratio),
                "high_conf_min_sigma_resolution_ratio": float(args.quality_high_conf_min_resolution_ratio),
                "max_fit_recon_mae": float(args.quality_max_fit_recon_mae),
                "max_sigma_recon_mae": float(args.quality_max_sigma_recon_mae),
                "max_sigma_residual_to_feature": float(args.quality_max_sigma_residual_to_feature),
                "min_host_sigma_fraction": float(args.quality_min_host_sigma_fraction),
                "min_absorption_depth": float(args.quality_min_absorption_depth),
                "risk_gold_max_expected_abs_error_kms": float(args.quality_risk_gold_max_expected_error),
                "risk_silver_max_expected_abs_error_kms": float(args.quality_risk_silver_max_expected_error),
            },
            "real_sdss_quality_cuts": {
                "max_calibrated_sigma_err_kms": real_sdss_max_error,
                "min_pred_host_fraction": float(args.quality_min_host),
                "min_sigma_mask_fraction": float(args.quality_min_sigma_mask_fraction),
                "min_valid_fraction": float(args.quality_min_valid_fraction),
                "min_sigma_resolution_ratio": float(args.quality_min_resolution_ratio),
                "real_min_sigma_resolution_ratio": float(args.quality_min_resolution_ratio),
                "advisory_sigma_resolution_ratio": 1.25,
                "high_conf_min_sigma_resolution_ratio": float(args.quality_high_conf_min_resolution_ratio),
                "max_fit_recon_mae": float(args.quality_max_fit_recon_mae),
                "max_sigma_recon_mae": float(args.quality_max_sigma_recon_mae),
                "max_sigma_residual_to_feature": float(args.quality_max_sigma_residual_to_feature),
                "min_host_sigma_fraction": float(args.quality_min_host_sigma_fraction),
                "min_absorption_depth": float(args.quality_min_absorption_depth),
                "risk_gold_max_expected_abs_error_kms": float(args.quality_risk_gold_max_expected_error),
                "risk_silver_max_expected_abs_error_kms": float(args.quality_risk_silver_max_expected_error),
                "science_ready_grades": ["A", "B"],
                "golden_sample_definition": "grade A, above the high-confidence resolution threshold, with acceptable decomposition/reconstruction diagnostics, and not near the training-label floor/ceiling",
                "note": "For real SDSS spectra this is a measurement-quality flag, not an extra training/test metric. The real-SDSS error gate is allowed to be looser than the mock lowest-uncertainty gate and is then constrained by the empirical risk model.",
            },
            "real_sdss_error_model": {
                "systematic_frac": float(args.real_sdss_systematic_frac),
                "low_resolution_extra_frac": float(args.real_sdss_lowres_extra_frac),
                "note": "Use these as additional floors for real SDSS inference, on top of validation-calibrated mock uncertainty.",
            },
            "validation_error_calibration": error_calibration,
            "quality_risk_model": quality_risk_model,
        }
    print(json.dumps(metrics, indent=2))

    output = Path(args.output)
    plot_diagnostics(
        true_sigma,
        pred_sigma,
        true_host,
        pred_host,
        snr_values,
        pred_sigma_err,
        metrics,
        output,
        calibrated_sigma=calibrated_sigma,
        redshift=test.get("redshift"),
        sigma_mask_fraction=test.get("sigma_mask_fraction"),
        valid_fraction=test.get("valid_fraction"),
        subset_metrics=subset_metrics,
    )
    print(f"Saved {output}")

    metrics_json = Path(args.metrics_json) if args.metrics_json else output.with_suffix(".json")
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {metrics_json}")
    if calibration_payload is not None:
        calibration_json = Path(args.calibration_json_out) if args.calibration_json_out else metrics_json.with_name(metrics_json.stem + "_calibration.json")
        calibration_json.write_text(json.dumps(calibration_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {calibration_json}")


if __name__ == "__main__":
    main()
