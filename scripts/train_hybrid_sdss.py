"""
Train HybridAGNNet on the SDSS-like mock HDF5 produced by generate_sdss_quasar_mocks.py.

Run from the project root, for example:
python scripts/train_hybrid_sdss.py --h5 ./mock_data/mock_sdss_quasar_dataset.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader_v2 import MockAGNDataset
from physmodel import HybridAGNNet, physics_hybrid_loss


DEFAULT_SCIENCE_SIGMA_BIN_EDGES = (30.0, 60.0, 80.0, 120.0, 180.0, 250.0, 400.0, 700.0)
DEFAULT_SCIENCE_SIGMA_BIN_WEIGHTS = (1.2, 1.4, 2.0, 2.0, 1.8, 1.5, 1.3)
SIGMA_SCALE_KMS = 120.0


def velocity_pixel_scale(wave: np.ndarray) -> float:
    return 299792.458 * np.median(np.diff(np.log(wave)))


def load_h5_assets(h5_path: str):
    with h5py.File(h5_path, "r") as hf:
        wave_grid = hf["wave_grid"][:].astype(np.float32)
        template_library = hf["template_library"][:].astype(np.float32)
        if "feii_template_bank" in hf:
            fe_template = hf["feii_template_bank"][:].astype(np.float32)
        elif "park_fe_template" in hf:
            fe_template = hf["park_fe_template"][:].astype(np.float32)
        else:
            fe_template = None
        n_z = len(hf["sps_logzsol"]) if "sps_logzsol" in hf else 10
        n_age = len(hf["sps_age_gyr"]) if "sps_age_gyr" in hf else template_library.shape[0] // n_z
        config = json.loads(hf.attrs.get("config_json", "{}"))
        sigma_min = float(config.get("sigma_min", 30.0))
        sigma_max = float(config.get("sigma_max", 750.0))
    return wave_grid, template_library, fe_template, n_z, n_age, sigma_min, sigma_max


def load_h5_physical_config(h5_path: str) -> dict[str, float]:
    with h5py.File(h5_path, "r") as hf:
        config = json.loads(hf.attrs.get("config_json", "{}"))
        if "sigma_inst_kms" in hf:
            values = hf["sigma_inst_kms"][:]
            values = values[np.isfinite(values) & (values > 0)]
            instrument_sigma_kms = float(np.median(values)) if values.size else float(config.get("sdss_sigma_inst", 70.0))
        else:
            instrument_sigma_kms = float(config.get("sdss_sigma_inst", 70.0))
        template_sigma_inst_kms = float(config.get("template_sigma_inst", 0.0))
        common_resolution_sigma = float(config.get("common_resolution_sigma", 0.0))
    return {
        "instrument_sigma_kms": instrument_sigma_kms,
        "template_sigma_inst_kms": template_sigma_inst_kms,
        "common_resolution_sigma": common_resolution_sigma,
    }



def parse_float_sequence(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None or str(text).strip() == "":
        return default
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or not np.isfinite(values).all():
        raise ValueError(f"Invalid float sequence: {text!r}")
    return values


def sigma_bin_weight_array(
    sigma: np.ndarray,
    bin_edges: tuple[float, ...],
    bin_weights: tuple[float, ...],
    boost: float,
) -> np.ndarray:
    if boost <= 0 or len(bin_edges) < 2:
        return np.ones_like(sigma, dtype=np.float32)
    edges = np.asarray(bin_edges, dtype=float)
    weights = np.asarray(bin_weights, dtype=float)
    n_bins = len(edges) - 1
    if weights.size != n_bins:
        weights = np.ones(n_bins, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.mean() if weights.mean() > 0 else np.ones(n_bins, dtype=float)
    indices = np.clip(np.searchsorted(edges, sigma, side="right") - 1, 0, n_bins - 1)
    return (1.0 + boost * (weights[indices] - 1.0)).astype(np.float32)


def sigma_science_bin_weights(
    batch_sigma: torch.Tensor,
    bin_edges: tuple[float, ...],
    bin_weights: tuple[float, ...],
    boost: float,
) -> torch.Tensor | None:
    if boost <= 0 or len(bin_edges) < 2:
        return None
    sigma_np = batch_sigma.detach().cpu().numpy().reshape(-1)
    weights_np = sigma_bin_weight_array(sigma_np, bin_edges, bin_weights, boost)
    return torch.as_tensor(weights_np, dtype=batch_sigma.dtype, device=batch_sigma.device)


def sigma_binned_consistency_loss(
    pred_sigma: torch.Tensor,
    true_sigma: torch.Tensor,
    bin_edges: tuple[float, ...],
    sigma_scale_kms: float = 120.0,
    min_count: int = 8,
    include_correlation: bool = True,
) -> torch.Tensor:
    pred = pred_sigma.squeeze()
    true = true_sigma.squeeze()
    if pred.numel() < min_count or len(bin_edges) < 2:
        return pred.new_tensor(0.0)
    total = pred.new_tensor(0.0)
    used = 0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (true >= float(lo)) & (true < float(hi))
        if int(mask.sum().item()) < min_count:
            continue
        p = pred[mask]
        t = true[mask]
        residual = (p - t) / sigma_scale_kms
        zero = residual.new_zeros(1)
        bias_loss = torch.nn.functional.smooth_l1_loss(residual.mean().view(1), zero, beta=0.25)
        p_std = torch.sqrt(torch.mean((p - p.mean()) ** 2).clamp_min(1e-8))
        t_std = torch.sqrt(torch.mean((t - t.mean()) ** 2).clamp_min(1e-8))
        std_loss = torch.nn.functional.smooth_l1_loss(((p_std - t_std) / sigma_scale_kms).view(1), zero, beta=0.25)
        if include_correlation:
            corr = torch.mean((p - p.mean()) * (t - t.mean())) / (p_std * t_std)
            corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            total = total + 0.45 * bias_loss + 0.35 * std_loss + 0.20 * (1.0 - corr)
        else:
            total = total + 0.55 * bias_loss + 0.45 * std_loss
        used += 1
    return total / used if used else pred.new_tensor(0.0)


def sigma_pairwise_order_loss(
    pred_sigma: torch.Tensor,
    true_sigma: torch.Tensor,
    min_delta_kms: float = 18.0,
    scale_kms: float = 55.0,
) -> torch.Tensor:
    pred = pred_sigma.squeeze()
    true = true_sigma.squeeze()
    if pred.numel() < 4:
        return pred.new_tensor(0.0)
    true_delta = true[:, None] - true[None, :]
    pred_delta = pred[:, None] - pred[None, :]
    pair_mask = torch.triu((true_delta.abs() >= float(min_delta_kms)), diagonal=1)
    if int(pair_mask.sum().item()) == 0:
        return pred.new_tensor(0.0)
    sign = torch.sign(true_delta[pair_mask])
    ordered_margin = sign * pred_delta[pair_mask] / max(float(scale_kms), 1e-6)
    return torch.nn.functional.softplus(-ordered_margin).mean()


def sigma_log_residual_loss(
    pred_sigma: torch.Tensor,
    true_sigma: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    beta: float = 0.18,
) -> torch.Tensor:
    pred = pred_sigma.squeeze().clamp_min(1.0)
    true = true_sigma.squeeze().clamp_min(1.0)
    residual = torch.log(pred) - torch.log(true)
    per_sample = torch.nn.functional.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=float(beta),
        reduction="none",
    )
    if sample_weight is not None:
        weight = sample_weight.to(per_sample.device).squeeze().clamp_min(0.0)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)
    return per_sample.mean()


def line_broadening_augmented_batch(
    batch_x: torch.Tensor,
    batch_sigma: torch.Tensor,
    *,
    dv_per_pixel: float,
    sigma_max: float,
    probability: float,
    min_extra_kms: float,
    max_extra_kms: float,
    max_kernel_size: int = 151,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Create a same-object line-broadening augmentation for sigma supervision.

    This is a deliberately conservative augmentation: it smooths the observed
    flux and sigma-window-normalized flux channels, keeps the same masks/noise
    metadata, and relabels the target as sqrt(sigma_star^2 + sigma_extra^2).
    It gives the network an explicit contrastive cue that, all else held fixed,
    broader absorption lines must imply a larger velocity dispersion.
    """
    if probability <= 0 or max_extra_kms <= 0 or batch_x.dim() != 3:
        return None
    n_batch, n_channels, n_pix = batch_x.shape
    if n_batch == 0 or n_pix < 8:
        return None
    with torch.no_grad():
        selected = torch.rand(n_batch, device=batch_x.device) < float(probability)
        if not torch.any(selected):
            return None
        indices = torch.nonzero(selected, as_tuple=False).view(-1)
        x_aug = batch_x[indices].clone()
        base_sigma = batch_sigma[indices].view(-1, 1)
        lo = max(float(min_extra_kms), 0.0)
        hi = max(float(max_extra_kms), lo)
        extra = lo + (hi - lo) * torch.rand_like(base_sigma)
        aug_sigma = torch.sqrt(base_sigma.clamp_min(0.0) ** 2 + extra**2).clamp(max=float(sigma_max))
        effective_extra = torch.sqrt((aug_sigma**2 - base_sigma.clamp_min(0.0) ** 2).clamp_min(0.0)).view(-1)
        if torch.max(effective_extra).item() <= 1e-3:
            return None

        sigma_pix = (effective_extra / max(float(dv_per_pixel), 1e-6)).clamp_min(0.05)
        kernel_size = int(np.ceil(float(torch.max(sigma_pix).cpu()) * 8.0))
        kernel_size = max(7, min(int(max_kernel_size), kernel_size | 1))
        pad = kernel_size // 2
        grid = torch.arange(kernel_size, device=batch_x.device, dtype=batch_x.dtype).view(1, -1) - pad
        kernels = torch.exp(-0.5 * (grid / sigma_pix.view(-1, 1).clamp_min(1e-6)) ** 2)
        kernels = kernels / kernels.sum(dim=1, keepdim=True).clamp_min(1e-6)

        values = x_aug[:, 0:1, :]
        padded = F.pad(values, (pad, pad), mode="reflect")
        smoothed = F.conv1d(padded.transpose(0, 1), kernels.view(len(indices), 1, kernel_size), groups=len(indices))
        x_aug[:, 0, :] = smoothed.transpose(0, 1).squeeze(1)

        if n_channels > 3:
            sigma_mask = x_aug[:, -1, :].clamp(0.0, 1.0)
            for row in range(x_aug.size(0)):
                mask = sigma_mask[row] > 0.5
                if int(mask.sum().item()) > 4:
                    vals = x_aug[row, 0, mask]
                    center = torch.median(vals)
                    scale = torch.median(torch.abs(vals - center)).clamp_min(1e-3)
                    x_aug[row, 3, :] = ((x_aug[row, 0, :] - center) / scale).clamp(-8.0, 8.0) * sigma_mask[row]
        return x_aug, aug_sigma, indices


def sigma_bin_classification_loss(
    logits: torch.Tensor | None,
    true_sigma: torch.Tensor,
    bin_edges: tuple[float, ...],
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits is None or len(bin_edges) < 2:
        return true_sigma.new_tensor(0.0)
    n_bins = len(bin_edges) - 1
    if logits.size(1) != n_bins:
        return true_sigma.new_tensor(0.0)
    edges = true_sigma.new_tensor(bin_edges)
    target = torch.bucketize(true_sigma.squeeze(), edges[1:-1], right=False).long()
    target = target.clamp(0, n_bins - 1)
    per_sample = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    if sample_weight is not None:
        weight = sample_weight.to(per_sample.device).squeeze().clamp_min(0.0)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)
    return per_sample.mean()


def fixed_sigma_bin_metrics(
    true_sigma: np.ndarray,
    pred_sigma: np.ndarray,
    bin_edges: tuple[float, ...],
    bin_weights: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, float]], float]:
    rows = []
    weighted_sum = 0.0
    weight_sum = 0.0
    weights = np.ones(max(len(bin_edges) - 1, 0), dtype=float)
    if bin_weights is not None and len(bin_weights) == len(weights):
        weights = np.asarray(bin_weights, dtype=float)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (true_sigma >= float(lo)) & (true_sigma < float(hi))
        if not np.any(mask):
            continue
        row = sigma_metrics(true_sigma[mask], pred_sigma[mask])
        row.update({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum())})
        rows.append(row)
        weighted_sum += float(row["rmse"]) * float(weights[i])
        weight_sum += float(weights[i])
    return rows, float(weighted_sum / weight_sum) if weight_sum > 0 else float("nan")



def h5_has_split(h5_path: str, split: str) -> bool:
    with h5py.File(h5_path, "r") as hf:
        return f"{split}_indices" in hf or "split" in hf


def dataset_h5_indices(dataset) -> np.ndarray | None:
    if hasattr(dataset, "indices") and dataset.indices is not None:
        return np.asarray(dataset.indices, dtype=np.int64)
    return None


def sigma_extreme_weights(
    sigma: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    low: float,
    high: float,
    boost: float,
) -> np.ndarray:
    if boost <= 0:
        return np.ones_like(sigma, dtype=np.float32)
    midpoint = 0.5 * (sigma_min + sigma_max)
    half_width = max(0.5 * (sigma_max - sigma_min), 1.0)
    distance = np.clip(np.abs(sigma - midpoint) / half_width, 0.0, 1.0)
    weights = 1.0 + boost * distance**2
    weights += 0.5 * boost * ((sigma < low) | (sigma > high)).astype(np.float32)
    return weights.astype(np.float32)


def build_weighted_sampler(
    dataset,
    h5_path: str,
    sigma_min: float,
    sigma_max: float,
    low: float,
    high: float,
    boost: float,
    science_bin_edges: tuple[float, ...] = DEFAULT_SCIENCE_SIGMA_BIN_EDGES,
    science_bin_weights: tuple[float, ...] = DEFAULT_SCIENCE_SIGMA_BIN_WEIGHTS,
    science_bin_boost: float = 0.0,
):
    if boost <= 0:
        return None
    indices = dataset_h5_indices(dataset)
    if indices is None:
        return None
    with h5py.File(h5_path, "r") as hf:
        sigma = hf["sigma_star"][indices].astype(np.float32)
    weights = sigma_extreme_weights(sigma, sigma_min, sigma_max, low, high, boost)
    weights *= sigma_bin_weight_array(sigma, science_bin_edges, science_bin_weights, science_bin_boost)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def sigma_batch_weights(
    batch_sigma: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    low: float,
    high: float,
    boost: float,
) -> torch.Tensor | None:
    if boost <= 0:
        return None
    sigma = batch_sigma.squeeze()
    midpoint = 0.5 * (sigma_min + sigma_max)
    half_width = max(0.5 * (sigma_max - sigma_min), 1.0)
    distance = ((sigma - midpoint).abs() / half_width).clamp(0.0, 1.0)
    weights = 1.0 + boost * distance.pow(2)
    weights = weights + 0.5 * boost * ((sigma < low) | (sigma > high)).float()
    return weights


def sigma_quality_weights(
    batch_x: torch.Tensor,
    batch_sigma: torch.Tensor | None = None,
    batch_host: torch.Tensor | None = None,
    min_weight: float = 0.25,
    target_sigma_pixels: float = 350.0,
    host_floor: float = 0.10,
    instrument_sigma_kms: float = 70.0,
    resolution_min_weight: float = 0.35,
    resolution_full_ratio: float = 2.0,
) -> torch.Tensor:
    if batch_x.dim() != 3 or batch_x.size(1) < 5:
        return batch_x.new_ones(batch_x.size(0))

    sigma_mask = batch_x[:, -1, :].clamp(0.0, 1.0)
    n_sigma_pixels = sigma_mask.sum(dim=1)
    coverage_quality = (n_sigma_pixels / max(float(target_sigma_pixels), 1.0)).clamp(0.0, 1.0)

    ivar = batch_x[:, 1, :].clamp_min(0.0) if batch_x.size(1) > 1 else torch.ones_like(sigma_mask)
    mean_sigma_ivar = (ivar * sigma_mask).sum(dim=1) / n_sigma_pixels.clamp_min(1.0)
    ivar_quality = torch.sqrt(mean_sigma_ivar / (mean_sigma_ivar + 1.0)).clamp(0.0, 1.0)

    if batch_host is not None:
        host = batch_host.squeeze().clamp(0.0, 1.0)
        raw_host_quality = ((host - host_floor) / max(0.40 - host_floor, 1e-3)).clamp(0.0, 1.0)
        host_quality = torch.where(torch.isfinite(host), 0.35 + 0.65 * raw_host_quality, torch.ones_like(host))
    else:
        host_quality = torch.ones_like(coverage_quality)

    quality = (coverage_quality * ivar_quality * host_quality).clamp(0.0, 1.0)
    if batch_sigma is not None:
        ratio = batch_sigma.squeeze().to(quality.device) / max(float(instrument_sigma_kms), 1e-6)
        resolution_core = ((ratio - 1.0) / max(float(resolution_full_ratio) - 1.0, 1e-3)).clamp(0.0, 1.0)
        resolution_quality = float(resolution_min_weight) + (1.0 - float(resolution_min_weight)) * resolution_core
        quality = quality * resolution_quality
    return min_weight + (1.0 - min_weight) * quality


def sigma_metrics(true_sigma: np.ndarray, pred_sigma: np.ndarray) -> dict[str, float]:
    residual = pred_sigma - true_sigma
    if true_sigma.size > 1 and np.std(pred_sigma) > 0 and np.std(true_sigma) > 0:
        corr = float(np.corrcoef(true_sigma, pred_sigma)[0, 1])
    else:
        corr = 0.0
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "corr": corr,
        "pred_std": float(np.std(pred_sigma)),
        "true_std": float(np.std(true_sigma)),
    }


def evaluate_sigma_metrics(model, loader, device, science_bin_edges=None, science_bin_weights=None) -> dict[str, float]:
    true_sigma, pred_sigma = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            batch_sigma = batch[1]
            pred, _, _, _, _, _, _ = model(batch_x)
            true_sigma.append(batch_sigma.numpy().reshape(-1))
            pred_sigma.append(pred.cpu().numpy().reshape(-1))
    true = np.concatenate(true_sigma)
    pred = np.concatenate(pred_sigma)
    metrics = sigma_metrics(true, pred)
    if science_bin_edges is not None:
        rows, weighted_rmse = fixed_sigma_bin_metrics(true, pred, tuple(science_bin_edges), tuple(science_bin_weights) if science_bin_weights is not None else None)
        metrics["science_sigma_bins"] = rows
        metrics["science_bin_weighted_rmse"] = weighted_rmse
        low_mid = true < 250.0
        if np.any(low_mid):
            metrics["science_low_mid_rmse"] = sigma_metrics(true[low_mid], pred[low_mid])["rmse"]
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="./mock_data/mock_sdss_quasar_dataset.h5")
    parser.add_argument("--save", default="best_sdss_hybrid_model.pth")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help="Validation DataLoader workers. Default 0 is more stable for h5py+MPS inside VS Code/Jupyter on macOS.",
    )
    parser.add_argument(
        "--val-max-batches",
        type=int,
        default=64,
        help=(
            "Maximum validation batches on non-full-validation epochs. "
            "Use 0 or a negative value to validate on the full validation set every epoch."
        ),
    )
    parser.add_argument(
        "--full-val-every",
        type=int,
        default=4,
        help=(
            "Run a full validation pass every N epochs. Best-checkpoint selection and early stopping "
            "use only full-validation epochs when val-max-batches is positive. Use 1 for full validation every epoch."
        ),
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep training DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--channels", default="flux,ivar,mask,sigma_flux,inst_sigma,region_cat,region_fe,region_cahk,sigma_mask")
    parser.add_argument("--sigma-warmup-epochs", type=int, default=3)
    parser.add_argument("--sigma-warmup-min-host", type=float, default=0.30)
    parser.add_argument("--sigma-warmup-min-snr", type=float, default=20.0)
    parser.add_argument("--extreme-sigma-low", type=float, default=80.0)
    parser.add_argument("--extreme-sigma-high", type=float, default=500.0)
    parser.add_argument("--extreme-sigma-sampler-boost", type=float, default=2.5)
    parser.add_argument("--extreme-sigma-loss-boost", type=float, default=1.5)
    parser.add_argument("--science-sigma-bin-edges", default="30,60,80,120,180,250,400,700")
    parser.add_argument("--science-sigma-bin-loss-weights", default="1.2,1.4,2.0,2.0,1.8,1.5,1.3")
    parser.add_argument("--science-bin-sampler-boost", type=float, default=1.0)
    parser.add_argument("--science-bin-loss-boost", type=float, default=1.0)
    parser.add_argument("--sigma-bin-consistency-weight", type=float, default=6.0)
    parser.add_argument("--sigma-pairwise-weight", type=float, default=0.0)
    parser.add_argument("--sigma-pairwise-min-delta", type=float, default=18.0)
    parser.add_argument("--sigma-pairwise-scale", type=float, default=55.0)
    parser.add_argument("--sigma-log-weight", type=float, default=0.0)
    parser.add_argument("--sigma-bin-class-weight", type=float, default=0.0)
    parser.add_argument("--line-broadening-aug-prob", type=float, default=0.0)
    parser.add_argument("--line-broadening-aug-min-kms", type=float, default=35.0)
    parser.add_argument("--line-broadening-aug-max-kms", type=float, default=150.0)
    parser.add_argument("--line-broadening-aug-weight", type=float, default=0.0)
    parser.add_argument("--line-broadening-order-weight", type=float, default=0.0)
    parser.add_argument("--sigma-weight", type=float, default=20.0)
    parser.add_argument("--sigma-distribution-weight", type=float, default=0.5)
    parser.add_argument("--recon-weight", type=float, default=25.0)
    parser.add_argument("--sigma-recon-weight", type=float, default=8.0)
    parser.add_argument("--sigma-deriv-weight", type=float, default=2.0)
    parser.add_argument("--sigma-deriv-corr-weight", type=float, default=0.0)
    parser.add_argument("--host-sigma-fraction-weight", type=float, default=2.0)
    parser.add_argument(
        "--allow-mps-reconstruction-grad",
        action="store_true",
        help=(
            "Allow reconstruction/decomposition losses to backpropagate through the "
            "per-sample broadening kernels and the shared encoder on Apple MPS. "
            "By default reconstruction losses remain active, but their encoder and "
            "velocity/Fe broadening paths are detached because MPS backward can produce "
            "non-finite gradients there. Sigma, host-fraction, SFH, and uncertainty "
            "losses still train the encoder."
        ),
    )
    parser.add_argument(
        "--allow-mps-global-sigma-losses",
        action="store_true",
        help=(
            "Allow batch-level sigma-distribution/bin-consistency/pairwise/log-sigma "
            "regularizers and the derivative-correlation reconstruction regularizer to "
            "backpropagate on Apple MPS. These regularizers are useful "
            "on CUDA/CPU, but MPS can return non-finite gradients for their reduction-heavy "
            "backward path. By default only the correlation-heavy pieces are disabled on "
            "MPS while MPS-stable pairwise/log-sigma/bin-moment supervision remains active."
        ),
    )
    parser.add_argument(
        "--full-loss-ramp-epochs",
        type=int,
        default=3,
        help="Ramp non-sigma auxiliary losses over this many epochs after sigma warmup.",
    )
    parser.add_argument("--host-weight", type=float, default=15.0)
    parser.add_argument("--smooth-weight", type=float, default=2.0)
    parser.add_argument("--sfh-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-recon-score-weight", type=float, default=100.0)
    parser.add_argument("--checkpoint-host-score-weight", type=float, default=50.0)
    parser.add_argument("--checkpoint-bin-rmse-weight", type=float, default=0.50)
    parser.add_argument("--checkpoint-lowmid-rmse-weight", type=float, default=0.0)
    parser.add_argument("--quality-loss-min-weight", type=float, default=0.25)
    parser.add_argument("--quality-loss-target-sigma-pixels", type=float, default=350.0)
    parser.add_argument("--quality-loss-host-floor", type=float, default=0.10)
    parser.add_argument("--quality-loss-resolution-min-weight", type=float, default=0.35)
    parser.add_argument("--quality-loss-resolution-full-ratio", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Stop after this many non-improving full-training epochs; 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.25, help="Minimum checkpoint-score improvement required to reset patience.")
    parser.add_argument("--min-full-training-epochs", type=int, default=15, help="Minimum number of post-warmup epochs to run before early stopping.")
    parser.add_argument("--history-json", default=None, help="Optional path to write per-epoch training history.")
    parser.add_argument("--resume-from", default=None, help="Optional model checkpoint to load before continuing training.")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    science_bin_edges = parse_float_sequence(args.science_sigma_bin_edges, DEFAULT_SCIENCE_SIGMA_BIN_EDGES)
    science_bin_loss_weights = parse_float_sequence(args.science_sigma_bin_loss_weights, DEFAULT_SCIENCE_SIGMA_BIN_WEIGHTS)
    if len(science_bin_loss_weights) != max(len(science_bin_edges) - 1, 0):
        raise ValueError("science-sigma-bin-loss-weights must have len(edges)-1 entries")
    print(f"Science sigma bins: {science_bin_edges}; weights={science_bin_loss_weights}")

    wave_grid, template_library, fe_template, n_z, n_age, sigma_min, sigma_max = load_h5_assets(args.h5)
    physical_config = load_h5_physical_config(args.h5)
    dv_pixel = velocity_pixel_scale(wave_grid)

    channels = tuple(item.strip() for item in args.channels.split(",") if item.strip())
    dataset_kwargs = dict(
        channels=channels,
        flux_normalization=None,
        return_sfh=True,
    )
    if h5_has_split(args.h5, args.train_split) and h5_has_split(args.h5, args.val_split):
        train_dataset = MockAGNDataset(args.h5, split=args.train_split, **dataset_kwargs)
        val_dataset = MockAGNDataset(args.h5, split=args.val_split, **dataset_kwargs)
        warmup_train_dataset = MockAGNDataset(
            args.h5,
            split=args.train_split,
            min_host_fraction=args.sigma_warmup_min_host,
            min_snr=args.sigma_warmup_min_snr,
            **dataset_kwargs,
        )
        if len(warmup_train_dataset) == 0:
            warmup_train_dataset = train_dataset
    else:
        dataset = MockAGNDataset(args.h5, **dataset_kwargs)
        val_size = int(args.val_frac * len(dataset))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(20260526)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        warmup_train_dataset = train_dataset

    train_sampler = build_weighted_sampler(
        train_dataset,
        args.h5,
        sigma_min,
        sigma_max,
        args.extreme_sigma_low,
        args.extreme_sigma_high,
        args.extreme_sigma_sampler_boost,
        science_bin_edges,
        science_bin_loss_weights,
        args.science_bin_sampler_boost,
    )
    warmup_train_sampler = build_weighted_sampler(
        warmup_train_dataset,
        args.h5,
        sigma_min,
        sigma_max,
        args.extreme_sigma_low,
        args.extreme_sigma_high,
        args.extreme_sigma_sampler_boost,
        science_bin_edges,
        science_bin_loss_weights,
        args.science_bin_sampler_boost,
    )

    train_worker_kwargs = {}
    warmup_worker_kwargs = {}
    if args.num_workers > 0:
        train_worker_kwargs["persistent_workers"] = bool(args.persistent_workers)
        train_worker_kwargs["prefetch_factor"] = 2
        warmup_worker_kwargs["persistent_workers"] = False
        warmup_worker_kwargs["prefetch_factor"] = 2
    val_worker_kwargs = {}
    if args.val_num_workers > 0:
        val_worker_kwargs["persistent_workers"] = False
        val_worker_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        **train_worker_kwargs,
    )
    warmup_train_loader = DataLoader(
        warmup_train_dataset,
        batch_size=args.batch_size,
        shuffle=(warmup_train_sampler is None),
        sampler=warmup_train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        **warmup_worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=(device.type == "cuda"),
        **val_worker_kwargs,
    )

    model = HybridAGNNet(
        input_dim=len(wave_grid),
        template_library=template_library,
        dv_per_pixel=dv_pixel,
        input_channels=len(channels),
        wave_grid=wave_grid,
        fe_template=fe_template,
        instrument_sigma_kms=physical_config["instrument_sigma_kms"],
        template_sigma_inst_kms=physical_config["template_sigma_inst_kms"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    ).to(device)

    resume_from = Path(args.resume_from).expanduser() if args.resume_from else None
    resumed_from_text = None
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"--resume-from checkpoint does not exist: {resume_from}")
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state)
        resumed_from_text = str(resume_from.resolve())
        print(f"Resumed model weights from {resumed_from_text}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_full_val_score = float("inf")
    best_epoch = None
    stale_full_epochs = 0
    history = []
    if args.history_json and resume_from is not None:
        history_path = Path(args.history_json)
        if history_path.exists():
            try:
                previous = json.loads(history_path.read_text(encoding="utf-8"))
                history = list(previous.get("history", []))
                best_epoch = previous.get("best_epoch")
                previous_score = previous.get("best_full_val_score")
                if previous_score is not None and np.isfinite(float(previous_score)):
                    best_full_val_score = float(previous_score)
                print(
                    f"Resuming history from {history_path}: "
                    f"{len(history)} completed epochs, best_epoch={best_epoch}, "
                    f"best_score={best_full_val_score if np.isfinite(best_full_val_score) else None}."
                )
            except Exception as exc:
                print(f"Warning: could not read existing history JSON for resume: {exc}")
                history = []
                best_epoch = None
                best_full_val_score = float("inf")
    start_epoch = min(len(history), args.epochs) if resume_from is not None else 0
    if start_epoch >= args.epochs:
        print(f"Training history already has {start_epoch} epochs >= requested {args.epochs}; nothing to do.")
        return
    full_loss_kwargs = dict(
        sigma_weight=args.sigma_weight,
        sigma_distribution_weight=args.sigma_distribution_weight,
        recon_weight=args.recon_weight,
        sigma_recon_weight=args.sigma_recon_weight,
        sigma_deriv_weight=args.sigma_deriv_weight,
        sigma_deriv_corr_weight=args.sigma_deriv_corr_weight,
        host_sigma_fraction_weight=args.host_sigma_fraction_weight,
        host_weight=args.host_weight,
        smooth_weight=args.smooth_weight,
        sfh_weight=args.sfh_weight,
        uncertainty_weight=args.uncertainty_weight,
    )
    train_full_loss_kwargs = dict(full_loss_kwargs)
    mps_bin_scale = 1.0
    mps_bin_include_correlation = True
    mps_pairwise_scale = 1.0
    mps_log_scale = 1.0
    if device.type == "mps" and not args.allow_mps_reconstruction_grad:
        if hasattr(model, "detach_mps_broadening_for_reconstruction"):
            model.detach_mps_broadening_for_reconstruction = True
        if hasattr(model, "detach_mps_reconstruction_encoder"):
            model.detach_mps_reconstruction_encoder = True
        if hasattr(model, "agn_branch") and hasattr(model.agn_branch, "detach_broadened_fe_on_mps"):
            model.agn_branch.detach_broadened_fe_on_mps = True
        if any(
            train_full_loss_kwargs[key] > 0
            for key in (
                "recon_weight",
                "sigma_recon_weight",
                "sigma_deriv_weight",
                "sigma_deriv_corr_weight",
                "host_sigma_fraction_weight",
            )
        ):
            print(
                "[MPS safety] Reconstruction/decomposition losses remain active, but "
                "their shared-encoder, stellar velocity-broadening, and Fe-template-broadening "
                "gradients are detached to avoid non-finite MPS backward. "
                "The encoder is still trained by supervised sigma, host-fraction, SFH, "
                "and uncertainty losses."
            )
    else:
        if hasattr(model, "detach_mps_broadening_for_reconstruction"):
            model.detach_mps_broadening_for_reconstruction = False
        if hasattr(model, "detach_mps_reconstruction_encoder"):
            model.detach_mps_reconstruction_encoder = False
        if hasattr(model, "agn_branch") and hasattr(model.agn_branch, "detach_broadened_fe_on_mps"):
            model.agn_branch.detach_broadened_fe_on_mps = False
    if device.type == "mps" and not args.allow_mps_global_sigma_losses:
        if (
            train_full_loss_kwargs.get("sigma_distribution_weight", 0.0) > 0
            or train_full_loss_kwargs.get("sigma_deriv_corr_weight", 0.0) > 0
            or args.sigma_bin_consistency_weight > 0
            or args.sigma_pairwise_weight > 0
            or args.sigma_log_weight > 0
        ):
            print(
                "[MPS safety] Disabling correlation-heavy sigma-distribution and "
                "derivative-correlation regularizers for MPS stability. Bin consistency "
                "uses its MPS-safe bias/std moment form; pairwise and log-sigma "
                "regularizers remain active when their weights are nonzero. The main "
                "supervised sigma loss, science-bin sample weights, quality weights, "
                "host/SFH/uncertainty losses, and reconstruction/decomposition heads "
                "remain active. Pass "
                "--allow-mps-global-sigma-losses only for debugging these auxiliary "
                "regularizers."
            )
        train_full_loss_kwargs["sigma_distribution_weight"] = 0.0
        train_full_loss_kwargs["sigma_deriv_corr_weight"] = 0.0
        mps_bin_include_correlation = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_totals = np.zeros(5, dtype=float)
        train_optimization_loss_sum = 0.0
        train_aux_weighted_sums = {
            "bin": 0.0,
            "pairwise": 0.0,
            "log_sigma": 0.0,
            "bin_class": 0.0,
            "line_aug": 0.0,
            "line_aug_order": 0.0,
        }
        active_train_loader = warmup_train_loader if epoch < args.sigma_warmup_epochs else train_loader
        pbar = tqdm(active_train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        if epoch < args.sigma_warmup_epochs:
            loss_kwargs = dict(
                sigma_weight=60.0,
                sigma_distribution_weight=0.0,
                recon_weight=0.0,
                sigma_recon_weight=0.0,
                sigma_deriv_weight=0.0,
                sigma_deriv_corr_weight=0.0,
                host_sigma_fraction_weight=0.0,
                host_weight=0.0,
                smooth_weight=0.0,
                sfh_weight=0.0,
                uncertainty_weight=0.0,
            )
            bin_weight = 0.0
            pairwise_weight = 0.0
            log_weight = 0.0
            bin_class_weight = args.sigma_bin_class_weight
        else:
            full_epoch = epoch - args.sigma_warmup_epochs + 1
            ramp_epochs = max(int(args.full_loss_ramp_epochs), 1)
            aux_ramp = min(1.0, full_epoch / float(ramp_epochs))
            loss_kwargs = dict(train_full_loss_kwargs)
            for key in (
                "recon_weight",
                "sigma_recon_weight",
                "sigma_deriv_weight",
                "sigma_deriv_corr_weight",
                "host_sigma_fraction_weight",
                "host_weight",
                "smooth_weight",
                "sfh_weight",
                "uncertainty_weight",
            ):
                loss_kwargs[key] *= aux_ramp
            bin_weight = args.sigma_bin_consistency_weight * aux_ramp * mps_bin_scale
            pairwise_weight = args.sigma_pairwise_weight * aux_ramp * mps_pairwise_scale
            log_weight = args.sigma_log_weight * aux_ramp * mps_log_scale
            bin_class_weight = args.sigma_bin_class_weight * max(0.5, aux_ramp)
        for batch_x, batch_sigma, batch_host, _, batch_sfh in pbar:
            batch_x = batch_x.to(device)
            batch_sigma = batch_sigma.to(device)
            batch_host = batch_host.to(device)
            batch_sfh = batch_sfh.to(device)
            sigma_sample_weight = sigma_batch_weights(
                batch_sigma,
                sigma_min,
                sigma_max,
                args.extreme_sigma_low,
                args.extreme_sigma_high,
                args.extreme_sigma_loss_boost,
            )
            science_weight = sigma_science_bin_weights(
                batch_sigma,
                science_bin_edges,
                science_bin_loss_weights,
                args.science_bin_loss_boost,
            )
            if science_weight is not None:
                sigma_sample_weight = science_weight if sigma_sample_weight is None else sigma_sample_weight * science_weight
            quality_weight = sigma_quality_weights(
                batch_x,
                batch_sigma=batch_sigma,
                batch_host=batch_host,
                min_weight=args.quality_loss_min_weight,
                target_sigma_pixels=args.quality_loss_target_sigma_pixels,
                host_floor=args.quality_loss_host_floor,
                instrument_sigma_kms=physical_config["instrument_sigma_kms"],
                resolution_min_weight=args.quality_loss_resolution_min_weight,
                resolution_full_ratio=args.quality_loss_resolution_full_ratio,
            )
            sigma_sample_weight = quality_weight if sigma_sample_weight is None else sigma_sample_weight * quality_weight

            optimizer.zero_grad(set_to_none=True)
            pred_sigma, pred_logvar, pred_sfh, pred_host, recon_spec, _, stellar_part = model(batch_x)
            loss_tuple = physics_hybrid_loss(
                batch_x,
                recon_spec,
                pred_sigma,
                pred_logvar,
                batch_sigma,
                pred_sfh,
                pred_host,
                batch_host,
                true_sfh=batch_sfh,
                stellar_part=stellar_part,
                sigma_sample_weight=sigma_sample_weight,
                n_z=n_z,
                n_age=n_age,
                **loss_kwargs,
            )
            bin_consistency_loss = (
                sigma_binned_consistency_loss(
                    pred_sigma,
                    batch_sigma,
                    science_bin_edges,
                    include_correlation=mps_bin_include_correlation,
                )
                if bin_weight > 0
                else pred_sigma.new_tensor(0.0)
            )
            pairwise_loss = (
                sigma_pairwise_order_loss(
                    pred_sigma,
                    batch_sigma,
                    min_delta_kms=args.sigma_pairwise_min_delta,
                    scale_kms=args.sigma_pairwise_scale,
                )
                if pairwise_weight > 0
                else pred_sigma.new_tensor(0.0)
            )
            log_sigma_loss = (
                sigma_log_residual_loss(pred_sigma, batch_sigma, sigma_sample_weight)
                if log_weight > 0
                else pred_sigma.new_tensor(0.0)
            )
            bin_class_loss = (
                sigma_bin_classification_loss(
                    getattr(model, "last_sigma_bin_logits", None),
                    batch_sigma,
                    science_bin_edges,
                    sigma_sample_weight,
                )
                if bin_class_weight > 0
                else pred_sigma.new_tensor(0.0)
            )
            aug_loss = pred_sigma.new_tensor(0.0)
            aug_order_loss = pred_sigma.new_tensor(0.0)
            if args.line_broadening_aug_weight > 0 or args.line_broadening_order_weight > 0:
                augmented = line_broadening_augmented_batch(
                    batch_x,
                    batch_sigma,
                    dv_per_pixel=dv_pixel,
                    sigma_max=sigma_max,
                    probability=args.line_broadening_aug_prob,
                    min_extra_kms=args.line_broadening_aug_min_kms,
                    max_extra_kms=args.line_broadening_aug_max_kms,
                )
                if augmented is not None:
                    aug_x, aug_sigma, aug_indices = augmented
                    aug_pred, _, _, _, _, _, _ = model(aug_x)
                    aug_residual = (aug_pred.squeeze() - aug_sigma.squeeze()) / SIGMA_SCALE_KMS
                    aug_per_sample = torch.nn.functional.smooth_l1_loss(
                        aug_residual,
                        torch.zeros_like(aug_residual),
                        beta=0.5,
                        reduction="none",
                    )
                    if sigma_sample_weight is not None:
                        aug_weight = sigma_sample_weight[aug_indices].to(aug_per_sample.device).squeeze().clamp_min(0.0)
                        aug_loss = (aug_per_sample * aug_weight).sum() / aug_weight.sum().clamp_min(1e-6)
                    else:
                        aug_loss = aug_per_sample.mean()
                    delta_target = (aug_sigma.squeeze() - batch_sigma[aug_indices].squeeze()).clamp_min(0.0)
                    order_mask = delta_target > max(args.sigma_pairwise_min_delta, 1.0)
                    if torch.any(order_mask):
                        delta_pred = aug_pred.squeeze()[order_mask] - pred_sigma.detach().squeeze()[aug_indices][order_mask]
                        aug_order_loss = torch.nn.functional.softplus(
                            -(delta_pred - 0.35 * delta_target[order_mask]) / max(args.sigma_pairwise_scale, 1e-6)
                        ).mean()
            loss = (
                loss_tuple[0]
                + bin_weight * bin_consistency_loss
                + pairwise_weight * pairwise_loss
                + log_weight * log_sigma_loss
                + bin_class_weight * bin_class_loss
                + args.line_broadening_aug_weight * aug_loss
                + args.line_broadening_order_weight * aug_order_loss
            )
            if not torch.isfinite(loss):
                with torch.no_grad():
                    components = [float(item.detach().cpu()) for item in loss_tuple]
                    pred_min = float(torch.nan_to_num(pred_sigma.detach()).min().cpu())
                    pred_max = float(torch.nan_to_num(pred_sigma.detach()).max().cpu())
                    true_min = float(batch_sigma.detach().min().cpu())
                    true_max = float(batch_sigma.detach().max().cpu())
                raise FloatingPointError(
                    "Non-finite training loss encountered: "
                    f"epoch={epoch + 1}, "
                    f"loss_tuple={components}, "
                    f"bin={float(bin_consistency_loss.detach().cpu())}, "
                    f"pairwise={float(pairwise_loss.detach().cpu())}, "
                        f"log_sigma={float(log_sigma_loss.detach().cpu())}, "
                        f"bin_class={float(bin_class_loss.detach().cpu())}, "
                        f"line_aug={float(aug_loss.detach().cpu())}, "
                        f"line_aug_order={float(aug_order_loss.detach().cpu())}, "
                        f"pred_sigma=[{pred_min:.2f}, {pred_max:.2f}], "
                        f"true_sigma=[{true_min:.2f}, {true_max:.2f}]"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                bad_grads = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        grad = parameter.grad.detach()
                        finite = grad[torch.isfinite(grad)]
                        finite_max = float(finite.abs().max().cpu()) if finite.numel() else float("nan")
                        bad_grads.append(f"{name}: finite_max={finite_max:.3e}, shape={tuple(grad.shape)}")
                        if len(bad_grads) >= 8:
                            break
                loss_components = {
                    "total_tuple": [float(item.detach().cpu()) for item in loss_tuple],
                    "bin": float(bin_consistency_loss.detach().cpu()),
                    "pairwise": float(pairwise_loss.detach().cpu()),
                    "log_sigma": float(log_sigma_loss.detach().cpu()),
                    "bin_class": float(bin_class_loss.detach().cpu()),
                    "line_aug": float(aug_loss.detach().cpu()),
                    "line_aug_order": float(aug_order_loss.detach().cpu()),
                }
                active_weights = {
                    "sigma_weight": float(loss_kwargs.get("sigma_weight", 0.0)),
                    "sigma_distribution_weight": float(loss_kwargs.get("sigma_distribution_weight", 0.0)),
                    "recon_weight": float(loss_kwargs.get("recon_weight", 0.0)),
                    "sigma_recon_weight": float(loss_kwargs.get("sigma_recon_weight", 0.0)),
                    "sigma_deriv_weight": float(loss_kwargs.get("sigma_deriv_weight", 0.0)),
                    "sigma_deriv_corr_weight": float(loss_kwargs.get("sigma_deriv_corr_weight", 0.0)),
                    "host_sigma_fraction_weight": float(loss_kwargs.get("host_sigma_fraction_weight", 0.0)),
                    "host_weight": float(loss_kwargs.get("host_weight", 0.0)),
                    "smooth_weight": float(loss_kwargs.get("smooth_weight", 0.0)),
                    "sfh_weight": float(loss_kwargs.get("sfh_weight", 0.0)),
                    "uncertainty_weight": float(loss_kwargs.get("uncertainty_weight", 0.0)),
                    "bin_weight": float(bin_weight),
                    "pairwise_weight": float(pairwise_weight),
                    "log_weight": float(log_weight),
                    "bin_class_weight": float(bin_class_weight),
                    "line_broadening_aug_weight": float(args.line_broadening_aug_weight),
                    "line_broadening_order_weight": float(args.line_broadening_order_weight),
                }
                raise FloatingPointError(
                    "Non-finite gradient encountered: "
                    f"epoch={epoch + 1}, grad_norm={float(grad_norm.detach().cpu())}, "
                    f"loss={float(loss.detach().cpu())}, "
                    f"loss_components={loss_components}, "
                    f"active_weights={active_weights}, "
                    f"bad_grads={bad_grads}"
                )
            optimizer.step()

            train_totals += np.array([item.item() for item in loss_tuple], dtype=float)
            train_optimization_loss_sum += float(loss.detach().cpu())
            train_aux_weighted_sums["bin"] += float((bin_weight * bin_consistency_loss).detach().cpu())
            train_aux_weighted_sums["pairwise"] += float((pairwise_weight * pairwise_loss).detach().cpu())
            train_aux_weighted_sums["log_sigma"] += float((log_weight * log_sigma_loss).detach().cpu())
            train_aux_weighted_sums["bin_class"] += float((bin_class_weight * bin_class_loss).detach().cpu())
            train_aux_weighted_sums["line_aug"] += float(
                (args.line_broadening_aug_weight * aug_loss).detach().cpu()
            )
            train_aux_weighted_sums["line_aug_order"] += float(
                (args.line_broadening_order_weight * aug_order_loss).detach().cpu()
            )
            pbar.set_postfix({"loss": f"{loss.item():.2f}", "sigma": f"{loss_tuple[1].item():.2f}"})

        model.eval()
        val_totals = np.zeros(5, dtype=float)
        true_sigma_for_metrics = []
        pred_sigma_for_metrics = []
        full_validation_epoch = (
            args.val_max_batches <= 0
            or args.full_val_every <= 1
            or (epoch + 1) % args.full_val_every == 0
            or (epoch + 1) >= args.epochs
        )
        val_batch_limit = None if full_validation_epoch else max(int(args.val_max_batches), 1)
        val_batches_seen = 0
        if epoch < args.sigma_warmup_epochs:
            val_loss_kwargs = dict(
                sigma_weight=60.0,
                sigma_distribution_weight=0.0,
                recon_weight=0.0,
                sigma_recon_weight=0.0,
                sigma_deriv_weight=0.0,
                sigma_deriv_corr_weight=0.0,
                host_sigma_fraction_weight=0.0,
                host_weight=0.0,
                smooth_weight=0.0,
                sfh_weight=0.0,
                uncertainty_weight=0.0,
            )
        else:
            val_loss_kwargs = train_full_loss_kwargs
        val_desc = "Val full" if full_validation_epoch else f"Val fast({val_batch_limit})"
        with torch.no_grad():
            for batch_i, (batch_x, batch_sigma, batch_host, _, batch_sfh) in enumerate(tqdm(val_loader, desc=val_desc, leave=False)):
                if val_batch_limit is not None and batch_i >= val_batch_limit:
                    break
                batch_x = batch_x.to(device)
                batch_sigma = batch_sigma.to(device)
                batch_host = batch_host.to(device)
                batch_sfh = batch_sfh.to(device)

                pred_sigma, pred_logvar, pred_sfh, pred_host, recon_spec, _, stellar_part = model(batch_x)
                loss_tuple = physics_hybrid_loss(
                    batch_x,
                    recon_spec,
                    pred_sigma,
                    pred_logvar,
                    batch_sigma,
                    pred_sfh,
                    pred_host,
                    batch_host,
                    true_sfh=batch_sfh,
                    stellar_part=stellar_part,
                    n_z=n_z,
                    n_age=n_age,
                    **val_loss_kwargs,
                )
                val_totals += np.array([item.item() for item in loss_tuple], dtype=float)
                true_sigma_for_metrics.append(batch_sigma.detach().cpu().numpy().reshape(-1))
                pred_sigma_for_metrics.append(pred_sigma.detach().cpu().numpy().reshape(-1))
                val_batches_seen += 1

        train_avg = train_totals / max(len(active_train_loader), 1)
        train_optimization_loss_avg = train_optimization_loss_sum / max(len(active_train_loader), 1)
        train_aux_weighted_avg = {
            key: value / max(len(active_train_loader), 1)
            for key, value in train_aux_weighted_sums.items()
        }
        val_avg = val_totals / max(val_batches_seen, 1)
        if true_sigma_for_metrics and pred_sigma_for_metrics:
            val_true_sigma = np.concatenate(true_sigma_for_metrics)
            val_pred_sigma = np.concatenate(pred_sigma_for_metrics)
            val_sigma_metrics = sigma_metrics(val_true_sigma, val_pred_sigma)
            rows, weighted_rmse = fixed_sigma_bin_metrics(
                val_true_sigma,
                val_pred_sigma,
                tuple(science_bin_edges),
                tuple(science_bin_loss_weights) if science_bin_loss_weights is not None else None,
            )
            val_sigma_metrics["science_sigma_bins"] = rows
            val_sigma_metrics["science_bin_weighted_rmse"] = weighted_rmse
            low_mid = val_true_sigma < 250.0
            if np.any(low_mid):
                val_sigma_metrics["science_low_mid_rmse"] = sigma_metrics(val_true_sigma[low_mid], val_pred_sigma[low_mid])["rmse"]
        else:
            val_sigma_metrics = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "corr": float("nan"),
                "pred_std": float("nan"),
                "true_std": float("nan"),
            }
        if epoch >= args.sigma_warmup_epochs and full_validation_epoch and np.isfinite(float(val_sigma_metrics["rmse"])):
            scheduler.step(val_sigma_metrics["rmse"])

        print(
            f"Epoch {epoch + 1}: "
            f"train loss={train_optimization_loss_avg:.3f}, main={train_avg[0]:.3f}, "
            f"sigma={train_avg[1]:.3f}, recon={train_avg[2]:.4f}, host={train_avg[3]:.4f}; "
            f"val main={val_avg[0]:.3f}, sigma={val_avg[1]:.3f}, recon={val_avg[2]:.4f}, host={val_avg[3]:.4f}; "
            f"val mode={'full' if full_validation_epoch else 'fast'}, batches={val_batches_seen}/{len(val_loader)}; "
            f"val sigma RMSE={val_sigma_metrics['rmse']:.1f}, "
            f"binRMSE={val_sigma_metrics.get('science_bin_weighted_rmse', float('nan')):.1f}, "
            f"corr={val_sigma_metrics['corr']:.3f}, pred_std={val_sigma_metrics['pred_std']:.1f}"
        )

        eligible_for_checkpoint = epoch >= args.sigma_warmup_epochs and full_validation_epoch
        selection_score = (
            val_sigma_metrics["rmse"]
            + args.checkpoint_bin_rmse_weight * float(val_sigma_metrics.get("science_bin_weighted_rmse", val_sigma_metrics["rmse"]))
            + args.checkpoint_lowmid_rmse_weight * float(val_sigma_metrics.get("science_low_mid_rmse", val_sigma_metrics["rmse"]))
            + args.checkpoint_recon_score_weight * float(val_avg[2])
            + args.checkpoint_host_score_weight * float(val_avg[3])
        )
        epoch_record = {
            "epoch": epoch + 1,
            "phase": "warmup" if epoch < args.sigma_warmup_epochs else "full",
            "lr": float(optimizer.param_groups[0]["lr"]),
            "validation_mode": "full" if full_validation_epoch else "fast",
            "validation_batches": int(val_batches_seen),
            "validation_total_batches": int(len(val_loader)),
            "train_optimization_loss": float(train_optimization_loss_avg),
            "train_main_loss": float(train_avg[0]),
            "train_aux_weighted_loss": float(sum(train_aux_weighted_avg.values())),
            "train_aux_weighted_components": {
                key: float(value) for key, value in train_aux_weighted_avg.items()
            },
            "val_main_loss": float(val_avg[0]),
            "train_total": float(train_avg[0]),
            "train_sigma": float(train_avg[1]),
            "train_recon": float(train_avg[2]),
            "train_host": float(train_avg[3]),
            "val_total": float(val_avg[0]),
            "val_sigma_loss": float(val_avg[1]),
            "val_recon": float(val_avg[2]),
            "val_host": float(val_avg[3]),
            "val_sigma_rmse": float(val_sigma_metrics["rmse"]),
            "val_sigma_mae": float(val_sigma_metrics["mae"]),
            "val_sigma_bias": float(val_sigma_metrics["bias"]),
            "val_sigma_corr": float(val_sigma_metrics["corr"]),
            "val_pred_std": float(val_sigma_metrics["pred_std"]),
            "val_science_bin_weighted_rmse": float(val_sigma_metrics.get("science_bin_weighted_rmse", float("nan"))),
            "val_science_low_mid_rmse": float(val_sigma_metrics.get("science_low_mid_rmse", float("nan"))),
            "val_science_sigma_bins": val_sigma_metrics.get("science_sigma_bins", []),
            "selection_score": float(selection_score),
            "saved_best": False,
        }

        improvement = best_full_val_score - float(selection_score)
        improved = eligible_for_checkpoint and improvement > args.early_stopping_min_delta
        if improved:
            best_full_val_score = float(selection_score)
            best_epoch = epoch + 1
            stale_full_epochs = 0
            epoch_record["saved_best"] = True
            torch.save(model.state_dict(), args.save)
            print(
                f"Saved new best full-training model to {args.save} "
                f"(score={selection_score:.2f}, sigma_rmse={val_sigma_metrics['rmse']:.1f})"
            )
        elif eligible_for_checkpoint:
            stale_full_epochs += 1

        history.append(epoch_record)
        if args.history_json:
            history_path = Path(args.history_json)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_full_val_score": best_full_val_score if np.isfinite(best_full_val_score) else None,
                        "resumed_from": resumed_from_text,
                        "start_epoch": int(start_epoch + 1) if start_epoch else 1,
                        "stopped_early": False,
                        "history": history,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        full_epochs_done = max(epoch + 1 - args.sigma_warmup_epochs, 0)
        if (
            args.early_stopping_patience > 0
            and eligible_for_checkpoint
            and full_epochs_done >= args.min_full_training_epochs
            and stale_full_epochs >= args.early_stopping_patience
        ):
            print(
                "Early stopping: "
                f"no checkpoint-score improvement > {args.early_stopping_min_delta:.3f} "
                f"for {stale_full_epochs} full-training epochs. "
                f"Best epoch={best_epoch}, best score={best_full_val_score:.2f}."
            )
            if args.history_json:
                history_path = Path(args.history_json)
                history_path.write_text(
                    json.dumps(
                        {
                            "best_epoch": best_epoch,
                            "best_full_val_score": best_full_val_score if np.isfinite(best_full_val_score) else None,
                            "resumed_from": resumed_from_text,
                            "start_epoch": int(start_epoch + 1) if start_epoch else 1,
                            "stopped_early": True,
                            "history": history,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            break


if __name__ == "__main__":
    main()
