from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_loader_v2 import MockAGNDataset
from physmodel import HybridAGNNet
from train_hybrid_sdss import load_h5_assets, load_h5_physical_config, velocity_pixel_scale
from evaluate_hybrid_sdss import summarize


DEFAULT_RUN = os.environ.get("DL_VELDISP_RUN_TAG", "sdss_greene2006_science_run_v6")
RUN = DEFAULT_RUN
SIGMA_SCALE_KMS = 120.0


def _default_h5_path(run_tag: str) -> Path:
    candidates = sorted((ROOT / "mock_data").glob(f"mock_{run_tag}_*.h5"))
    if candidates:
        return candidates[-1]
    return ROOT / "mock_data" / f"mock_{run_tag}_100k.h5"


H5_PATH = _default_h5_path(RUN)
MODEL_PATH = ROOT / "models" / f"best_{RUN}.pth"
CALIBRATION_PATH = ROOT / "figures" / f"{RUN}_calibration.json"
METRICS_PATH = ROOT / "figures" / f"{RUN}_test_metrics.json"
PRED_CACHE = ROOT / "figures" / f"{RUN}_test_predictions_for_figure4.npz"
OUT = ROOT / "draft" / "DL_veldisp" / "figures" / f"mock_test_evaluation_{RUN.split('_')[-1]}_apj.pdf"


def configure_paths(
    *,
    run_tag: str = DEFAULT_RUN,
    h5_path: str | None = None,
    model_path: str | None = None,
    calibration_json: str | None = None,
    metrics_json: str | None = None,
    prediction_cache: str | None = None,
    output_pdf: str | None = None,
) -> None:
    global RUN, H5_PATH, MODEL_PATH, CALIBRATION_PATH, METRICS_PATH, PRED_CACHE, OUT
    RUN = run_tag
    H5_PATH = Path(h5_path) if h5_path else _default_h5_path(run_tag)
    MODEL_PATH = Path(model_path) if model_path else ROOT / "models" / f"best_{run_tag}.pth"
    CALIBRATION_PATH = Path(calibration_json) if calibration_json else ROOT / "figures" / f"{run_tag}_calibration.json"
    METRICS_PATH = Path(metrics_json) if metrics_json else ROOT / "figures" / f"{run_tag}_test_metrics.json"
    PRED_CACHE = Path(prediction_cache) if prediction_cache else ROOT / "figures" / f"{run_tag}_test_predictions_for_figure4.npz"
    OUT = Path(output_pdf) if output_pdf else ROOT / "draft" / "DL_veldisp" / "figures" / f"mock_test_evaluation_{run_tag.split('_')[-1]}_apj.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the ApJ mock-test summary figure.")
    parser.add_argument("--run-tag", default=DEFAULT_RUN)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--prediction-cache", default=None)
    parser.add_argument("--output-pdf", default=None)
    return parser.parse_args()


PLOT_SIGMA_MIN = 25.0
PLOT_SIGMA_MAX = 260.0
SCIENCE_SIGMA_MAX = 250.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def collect_test_predictions(batch_size: int = 256, num_workers: int = 2) -> dict[str, np.ndarray]:
    if PRED_CACHE.exists():
        cached = np.load(PRED_CACHE)
        print(f"loaded cached predictions: {PRED_CACHE}")
        return {key: cached[key] for key in cached.files}

    calibration = json.loads(CALIBRATION_PATH.read_text())
    channels = tuple(calibration["channels"])
    wave_grid, template_library, fe_template, _, _, sigma_min, sigma_max = load_h5_assets(str(H5_PATH))
    physical_config = load_h5_physical_config(str(H5_PATH))
    state = torch.load(MODEL_PATH, map_location="cpu")
    in_channels = state["backbone.0.weight"].shape[1]
    if len(channels) != in_channels:
        raise RuntimeError(f"channel mismatch: calibration has {len(channels)}, checkpoint expects {in_channels}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = HybridAGNNet(
        input_dim=len(wave_grid),
        template_library=template_library,
        dv_per_pixel=velocity_pixel_scale(wave_grid),
        input_channels=in_channels,
        wave_grid=wave_grid,
        fe_template=fe_template,
        instrument_sigma_kms=physical_config["instrument_sigma_kms"],
        template_sigma_inst_kms=physical_config["template_sigma_inst_kms"],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    dataset = MockAGNDataset(
        str(H5_PATH),
        channels=channels,
        flux_normalization=None,
        return_sfh=False,
        split="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    true_sigma, pred_sigma, true_host, pred_host, snr, pred_err = [], [], [], [], [], []
    with torch.no_grad():
        for batch_x, batch_sigma, batch_host, batch_snr in tqdm(loader, desc="Mock test inference"):
            batch_x = batch_x.to(device)
            pred, logvar, _, host, _, _, _ = model(batch_x)
            err = torch.exp(0.5 * torch.clamp(logvar.squeeze(), min=-6.0, max=4.0)) * SIGMA_SCALE_KMS
            true_sigma.append(batch_sigma.numpy().reshape(-1))
            pred_sigma.append(pred.cpu().numpy().reshape(-1))
            true_host.append(batch_host.numpy().reshape(-1))
            pred_host.append(host.cpu().numpy().reshape(-1))
            snr.append(batch_snr.numpy().reshape(-1))
            pred_err.append(err.cpu().numpy().reshape(-1))

    indices = dataset.indices if dataset.indices is not None else np.arange(len(dataset), dtype=np.int64)
    extras = {}
    with h5py.File(H5_PATH, "r") as hf:
        for key in ("sigma_mask_fraction", "valid_fraction", "sigma_resolution_ratio"):
            if key in hf:
                extras[key] = hf[key][indices]

    data = {
        "true_sigma": np.concatenate(true_sigma),
        "pred_sigma_raw": np.concatenate(pred_sigma),
        "true_host": np.concatenate(true_host),
        "pred_host": np.concatenate(pred_host),
        "snr": np.concatenate(snr),
        "pred_err_raw": np.concatenate(pred_err),
        **extras,
    }
    PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_CACHE, **data)
    print(f"saved prediction cache: {PRED_CACHE}")
    return data


def high_confidence_mask(data: dict[str, np.ndarray], cal: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = float(cal.get("sigma_affine_slope", cal.get("calibration_slope", 1.0)))
    intercept = float(cal.get("sigma_affine_intercept", cal.get("calibration_intercept", 0.0)))
    err_scale = float(cal.get("sigma_error_scale", cal.get("error_calibration", {}).get("scale", 1.0)))
    err_floor = float(cal.get("sigma_error_floor_kms", cal.get("error_calibration", {}).get("floor_kms", 0.0)))
    calibrated_sigma = slope * data["pred_sigma_raw"] + intercept
    calibrated_err = np.sqrt((err_scale * abs(slope) * data["pred_err_raw"]) ** 2 + err_floor**2)
    cuts = dict(cal.get("quality_cuts", {}))
    mask = calibrated_err <= float(cuts.get("max_calibrated_sigma_err_kms", np.nanpercentile(calibrated_err, 35)))
    mask &= data["pred_host"] >= float(cuts.get("min_pred_host_fraction", 0.45))
    if "sigma_mask_fraction" in data:
        mask &= data["sigma_mask_fraction"] >= float(cuts.get("min_sigma_mask_fraction", 0.015))
    if "valid_fraction" in data:
        mask &= data["valid_fraction"] >= float(cuts.get("min_valid_fraction", 0.20))
    if "sigma_resolution_ratio" in data:
        mask &= data["sigma_resolution_ratio"] >= float(cuts.get("min_sigma_resolution_ratio", 0.80))
    return mask, calibrated_sigma, calibrated_err


def binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        if np.count_nonzero(m) >= 10:
            centers.append(0.5 * (lo + hi))
            medians.append(np.nanmedian(y[m]))
    return np.asarray(centers), np.asarray(medians)


def add_density_panel(ax, x, y, title, summary, cmap, edges, show_legend=True):
    hb = ax.hexbin(
        x,
        y,
        gridsize=54,
        extent=(PLOT_SIGMA_MIN, PLOT_SIGMA_MAX, PLOT_SIGMA_MIN, PLOT_SIGMA_MAX),
        mincnt=1,
        bins="log",
        cmap=cmap,
        linewidths=0.0,
        alpha=0.96,
    )
    ax.plot([PLOT_SIGMA_MIN, PLOT_SIGMA_MAX], [PLOT_SIGMA_MIN, PLOT_SIGMA_MAX], color="0.12", lw=0.95, label="1:1 relation")
    bx, by = binned_median(x, y, edges)
    ax.plot(bx, by, color="#b44745", marker="o", ms=3.0, lw=1.05, label="bin median")
    ax.set_xlim(PLOT_SIGMA_MIN, PLOT_SIGMA_MAX)
    ax.set_ylim(PLOT_SIGMA_MIN, PLOT_SIGMA_MAX)
    ax.set_xlabel(r"True $\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"Predicted $\sigma_\star$ [km s$^{-1}$]")
    ax.text(0.04, 0.96, title, transform=ax.transAxes, ha="left", va="top")
    ax.text(
        0.04,
        0.07,
        rf"$N={summary['n']}$, RMSE={summary['rmse_kms']:.0f}" "\n" rf"$r={summary['sigma_corr']:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    if show_legend:
        ax.legend(frameon=False, loc="upper right", handlelength=1.55, borderaxespad=0.25)
    return hb


def main() -> None:
    setup_style()
    data = collect_test_predictions()
    cal = json.loads(CALIBRATION_PATH.read_text())
    _ = json.loads(METRICS_PATH.read_text())
    high_mask, calibrated_sigma, calibrated_err = high_confidence_mask(data, cal)

    true = data["true_sigma"]
    direct = data["pred_sigma_raw"]
    residual = direct - true
    direct_metrics = summarize(true, direct)
    corrected_metrics = summarize(true, calibrated_sigma)

    sigma_lo = 20.0
    sigma_hi = 720.0
    residual_limit = float(
        np.clip(np.ceil(np.nanpercentile(np.abs(residual), 99.8) / 50.0) * 50.0, 350.0, 700.0)
    )
    err_vmin, err_vmax = np.nanpercentile(calibrated_err, [2.0, 98.0])
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.15), constrained_layout=True)

    ax = axes[0, 0]
    direct_points = ax.scatter(
        true,
        direct,
        c=data["true_host"],
        s=4.5,
        alpha=0.32,
        cmap="viridis_r",
        edgecolors="none",
        rasterized=True,
        label="network prediction",
    )
    ax.scatter(
        true,
        calibrated_sigma,
        s=4.0,
        alpha=0.16,
        color="#f28e2b",
        edgecolors="none",
        rasterized=True,
        label="validation-corrected",
    )
    ax.plot([sigma_lo, sigma_hi], [sigma_lo, sigma_hi], color="0.12", lw=1.05, label="1:1 relation")
    ax.set_xlim(sigma_lo, sigma_hi)
    ax.set_ylim(sigma_lo, sigma_hi)
    ax.set_xlabel(r"True $\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"Predicted $\sigma_\star$ [km s$^{-1}$]")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9.0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.45)
    cbar = fig.colorbar(direct_points, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label(r"True host fraction, $f_{\rm host}$")

    ax = axes[0, 1]
    err_points = ax.scatter(
        data["snr"],
        residual,
        c=calibrated_err,
        s=4.5,
        alpha=0.30,
        cmap="magma",
        vmin=err_vmin,
        vmax=err_vmax,
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(0.0, color="0.12", lw=0.95)
    ax.set_xlim(3.0, 82.0)
    ax.set_ylim(-residual_limit, residual_limit)
    ax.set_xlabel("Mock S/N")
    ax.set_ylabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9.0)
    cbar = fig.colorbar(err_points, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label(r"Calibrated $\sigma_\star$ uncertainty [km s$^{-1}$]")

    ax = axes[1, 0]
    bins = np.linspace(-residual_limit, residual_limit, 65)
    ax.hist(residual, bins=bins, histtype="stepfilled", color="0.42", alpha=0.78)
    ax.axvline(0.0, color="0.12", lw=0.95)
    ax.set_xlim(-residual_limit, residual_limit)
    ax.set_xlabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel("Number")
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9.0)

    ax = axes[1, 1]
    ax.scatter(
        data["true_host"],
        residual,
        c=data["pred_host"],
        s=4.5,
        alpha=0.30,
        cmap="cividis",
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(0.0, color="0.12", lw=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-residual_limit, residual_limit)
    ax.set_xlabel(r"True host fraction, $f_{\rm host}$")
    ax.set_ylabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.text(0.03, 0.97, "(d)", transform=ax.transAxes, ha="left", va="top", fontsize=9.0)

    fig.suptitle(
        "Independent mock test: "
        f"N={direct_metrics['n']}, RMSE={direct_metrics['rmse_kms']:.1f} km s$^{{-1}}$, "
        f"MAE={direct_metrics['mae_kms']:.1f} km s$^{{-1}}$, "
        f"bias={direct_metrics['bias_kms']:.1f} km s$^{{-1}}$, "
        f"$r$={direct_metrics['sigma_corr']:.2f}",
        fontsize=10.0,
    )

    for axis in axes.flat:
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        if axis.get_yscale() != "log":
            axis.yaxis.set_minor_locator(AutoMinorLocator(4))
        axis.tick_params(which="major", length=4, width=0.8)
        axis.tick_params(which="minor", length=2, width=0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"saved: {OUT}")
    print("full-test direct", direct_metrics)
    print("full-test validation-corrected", corrected_metrics)
    print("high-confidence N", int(np.count_nonzero(high_mask)))


if __name__ == "__main__":
    args = parse_args()
    configure_paths(
        run_tag=args.run_tag,
        h5_path=args.h5,
        model_path=args.model,
        calibration_json=args.calibration_json,
        metrics_json=args.metrics_json,
        prediction_cache=args.prediction_cache,
        output_pdf=args.output_pdf,
    )
    main()
