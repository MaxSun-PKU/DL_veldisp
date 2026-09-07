from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from infer_sdss_quasar_sigma import AUTO_CHANNELS, build_rest_frame_channels, read_sdss_spectrum
from physmodel import HybridAGNNet
from train_hybrid_sdss import load_h5_assets, load_h5_physical_config, velocity_pixel_scale


DEFAULT_RUN = os.environ.get("DL_VELDISP_RUN_TAG", "sdss_greene2006_science_run_v6")
RUN = DEFAULT_RUN
TARGET_NAME = "J033013.26-053236.0"
TRUTH_LABEL = "lit"
TITLE_SUFFIX: str | None = None
ZOOM_C_MAX = 5785.0
ZOOM_C_TITLE = "Line-width-sensitive residual check"


def _default_h5_path(run_tag: str) -> Path:
    candidates = sorted((ROOT / "mock_data").glob(f"mock_{run_tag}_*.h5"))
    if candidates:
        return candidates[-1]
    return ROOT / "mock_data" / f"mock_{run_tag}_100k.h5"


H5_PATH = _default_h5_path(RUN)
MODEL_PATH = ROOT / "models" / f"best_{RUN}.pth"
CALIBRATION_PATH = ROOT / "figures" / f"{RUN}_calibration.json"
VALIDATION_CSV = ROOT / "figures" / f"{RUN}_greene2006_external_validation.csv"
OUT_ROOT = ROOT / "figures" / f"{RUN}_greene2006_reconstruction_zoom_apj.pdf"
OUT_DRAFT = ROOT / "draft" / "DL_veldisp" / "figures" / f"greene2006_reconstruction_zoom_{RUN.split('_')[-1]}.pdf"


def configure_paths(
    *,
    run_tag: str = DEFAULT_RUN,
    target_name: str = TARGET_NAME,
    h5_path: str | None = None,
    model_path: str | None = None,
    calibration_json: str | None = None,
    validation_csv: str | None = None,
    output_pdf: str | None = None,
    draft_pdf: str | None = None,
    truth_label: str = "lit",
    title_suffix: str | None = None,
    zoom_c_max: float = 5785.0,
    zoom_c_title: str = "Line-width-sensitive residual check",
) -> None:
    global RUN, TARGET_NAME, H5_PATH, MODEL_PATH, CALIBRATION_PATH, VALIDATION_CSV, OUT_ROOT, OUT_DRAFT
    global TRUTH_LABEL, TITLE_SUFFIX
    global ZOOM_C_MAX, ZOOM_C_TITLE
    RUN = run_tag
    TARGET_NAME = target_name
    H5_PATH = Path(h5_path) if h5_path else _default_h5_path(run_tag)
    MODEL_PATH = Path(model_path) if model_path else ROOT / "models" / f"best_{run_tag}.pth"
    CALIBRATION_PATH = Path(calibration_json) if calibration_json else ROOT / "figures" / f"{run_tag}_calibration.json"
    VALIDATION_CSV = Path(validation_csv) if validation_csv else ROOT / "figures" / f"{run_tag}_greene2006_external_validation.csv"
    OUT_ROOT = Path(output_pdf) if output_pdf else ROOT / "figures" / f"{run_tag}_greene2006_reconstruction_zoom_apj.pdf"
    OUT_DRAFT = Path(draft_pdf) if draft_pdf else ROOT / "draft" / "DL_veldisp" / "figures" / f"greene2006_reconstruction_zoom_{run_tag.split('_')[-1]}.pdf"
    TRUTH_LABEL = truth_label
    TITLE_SUFFIX = title_suffix
    ZOOM_C_MAX = float(zoom_c_max)
    ZOOM_C_TITLE = zoom_c_title


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a real-SDSS reconstruction zoom figure.")
    parser.add_argument("--run-tag", default=DEFAULT_RUN)
    parser.add_argument("--target-name", default=TARGET_NAME)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--draft-pdf", default=None)
    parser.add_argument("--truth-label", default="lit")
    parser.add_argument("--title-suffix", default=None)
    parser.add_argument("--zoom-c-max", type=float, default=5785.0)
    parser.add_argument("--zoom-c-title", default="Line-width-sensitive residual check")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 8.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def read_target_row() -> dict[str, str]:
    with open(VALIDATION_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("sdss_name") == TARGET_NAME:
                return dict(row)
    raise RuntimeError(f"{TARGET_NAME} not found in {VALIDATION_CSV}")


def _csv_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def target_quality_label(row: dict[str, str]) -> str:
    if _csv_bool(row.get("golden_sample", "")):
        return "high confidence"
    if _csv_bool(row.get("high_confidence", "")):
        return "high confidence"
    grade = str(row.get("science_grade", "")).strip()
    if grade:
        return f"Grade {grade}"
    tier = str(row.get("measurement_tier", "")).strip().replace("_", " ")
    return tier if tier else "real SDSS"


def load_model_and_channels():
    wave_grid, template_library, fe_template, _, _, sigma_min, sigma_max = load_h5_assets(str(H5_PATH))
    physical_config = load_h5_physical_config(str(H5_PATH))
    state = torch.load(MODEL_PATH, map_location="cpu")
    in_channels = int(state["backbone.0.weight"].shape[1])
    if CALIBRATION_PATH.exists():
        calibration = json.loads(CALIBRATION_PATH.read_text())
        channels = tuple(calibration.get("channels", ()))
        if len(channels) != in_channels:
            channels = AUTO_CHANNELS[in_channels]
    else:
        channels = AUTO_CHANNELS[in_channels]

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
    return model, device, wave_grid, channels, physical_config


def contiguous_spans(wave: np.ndarray, mask: np.ndarray, xlim: tuple[float, float]) -> list[tuple[float, float]]:
    m = (wave >= xlim[0]) & (wave <= xlim[1]) & mask
    if not np.any(m):
        return []
    idx = np.flatnonzero(m)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(float(wave[s]), float(wave[e])) for s, e in zip(starts, ends)]


def shade_sigma_windows(ax, wave: np.ndarray, sigma_mask: np.ndarray, xlim: tuple[float, float]) -> None:
    for lo, hi in contiguous_spans(wave, sigma_mask, xlim):
        ax.axvspan(lo, hi, color="#dcefd8", alpha=0.65, lw=0, zorder=0)


def display_curve_mask(
    wave: np.ndarray,
    sigma_mask: np.ndarray,
    xlim: tuple[float, float],
) -> np.ndarray:
    """Hide internal Fe-window intervals excluded from the sigma fit."""
    in_view = (wave >= xlim[0]) & (wave <= xlim[1])
    inside_nominal_fe = (wave >= 5250.0) & (wave <= 5820.0)
    return in_view & (~inside_nominal_fe | sigma_mask)


def shade_excluded_intervals(
    ax,
    wave: np.ndarray,
    sigma_mask: np.ndarray,
    xlim: tuple[float, float],
) -> None:
    inside_nominal_fe = (wave >= 5250.0) & (wave <= 5820.0)
    excluded = inside_nominal_fe & ~sigma_mask
    for lo, hi in contiguous_spans(wave, excluded, xlim):
        ax.axvspan(lo, hi, color="#d9d9d9", alpha=0.92, lw=0, zorder=1)


def plot_components(ax, wave, flux, recon, host, agn, sigma_mask, xlim, *, show_ylabel=True):
    shade_sigma_windows(ax, wave, sigma_mask, xlim)
    shade_excluded_intervals(ax, wave, sigma_mask, xlim)
    in_view = (wave >= xlim[0]) & (wave <= xlim[1])
    keep = display_curve_mask(wave, sigma_mask, xlim)[in_view]
    plot_wave = wave[in_view]
    ax.plot(plot_wave, np.where(keep, flux[in_view], np.nan), color="0.10", lw=1.0, label="SDSS flux", zorder=3)
    ax.plot(
        plot_wave,
        np.where(keep, recon[in_view], np.nan),
        color="#d62728",
        lw=1.35,
        label="model reconstruction",
        zorder=4,
    )
    ax.plot(
        plot_wave,
        np.where(keep, host[in_view], np.nan),
        color="#1f77b4",
        lw=1.2,
        label="host component",
        zorder=2,
    )
    ax.plot(
        plot_wave,
        np.where(keep, agn[in_view], np.nan),
        color="#f5a623",
        lw=1.2,
        label="AGN component",
        zorder=2,
    )
    ax.set_xlim(*xlim)
    if show_ylabel:
        ax.set_ylabel("Normalized flux")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))


def main() -> None:
    setup_style()
    row = read_target_row()
    model, device, wave_grid, channels, physical_config = load_model_and_channels()
    spectrum = read_sdss_spectrum(row["local_path"])
    redshift = float(row["redshift"])
    x, meta = build_rest_frame_channels(
        spectrum["observed_wave"],
        spectrum["flux"],
        spectrum["ivar"],
        spectrum["valid"],
        spectrum.get("inst_sigma_kms"),
        redshift,
        wave_grid,
        channels,
        fallback_inst_sigma_kms=physical_config["instrument_sigma_kms"],
        target_common_resolution_sigma_kms=physical_config.get("common_resolution_sigma", 0.0),
    )
    with torch.no_grad():
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        pred, logvar, _, host_frac, recon, agn_part, stellar_part = model(tensor)

    flux = x[0].astype(float)
    sigma_mask = x[-1] > 0.5
    recon_np = recon.cpu().numpy().reshape(-1).astype(float)
    agn_np = agn_part.cpu().numpy().reshape(-1).astype(float)
    host_np = stellar_part.cpu().numpy().reshape(-1).astype(float)
    residual = flux - recon_np
    pred_sigma = float(row["recommended_sigma_kms"])
    pred_sigma_err = float(row["recommended_sigma_err_kms"])
    truth_sigma = float(row.get("sigma_truth_kms") or row["sigma_lit_kms"])
    truth_err = float(row.get("sigma_truth_err_kms") or row["sigma_lit_err_kms"])

    fig = plt.figure(figsize=(7.35, 6.0), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.65, 0.86, 0.16, 1.95],
        hspace=0.24,
        wspace=0.22,
        left=0.095,
        right=0.985,
        bottom=0.095,
        top=0.865,
    )
    ax_main = fig.add_subplot(gs[0, :])
    ax_resid = fig.add_subplot(gs[1, :], sharex=ax_main)
    ax_b = fig.add_subplot(gs[3, 0])
    ax_c = fig.add_subplot(gs[3, 1])

    full_xlim = (5150.0, 5850.0)
    zoom_b = (5230.0, 5465.0)
    zoom_c = (5525.0, ZOOM_C_MAX)
    wave_label = r"Rest-frame wavelength [$\mathrm{\AA}$]"

    plot_components(ax_main, wave_grid, flux, recon_np, host_np, agn_np, sigma_mask, full_xlim)
    ax_main.text(0.012, 0.90, "(a)", transform=ax_main.transAxes, ha="left", va="top", fontsize=11)
    ax_main.text(
        0.985,
        0.02,
        rf"$\hat{{\sigma}}_\star={pred_sigma:.0f}\pm{pred_sigma_err:.0f}$, "
        rf"$\sigma_{{\star,\rm {TRUTH_LABEL}}}={truth_sigma:.0f}\pm{truth_err:.0f}$ km s$^{{-1}}$",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.6},
    )
    ax_main.set_ylim(0.12, 1.32)

    in_full = (wave_grid >= full_xlim[0]) & (wave_grid <= full_xlim[1])
    ax_resid.axhline(0.0, color="0.30", lw=0.75)
    ax_resid.fill_between(
        wave_grid[in_full],
        -0.06,
        0.06,
        color="0.88",
        zorder=0,
        lw=0,
        label=r"$\pm0.06$",
    )
    shade_excluded_intervals(ax_resid, wave_grid, sigma_mask, full_xlim)
    residual_keep = display_curve_mask(wave_grid, sigma_mask, full_xlim)[in_full]
    ax_resid.plot(
        wave_grid[in_full],
        np.where(residual_keep, residual[in_full], np.nan),
        color="0.30",
        lw=1.0,
    )
    ax_resid.set_xlim(*full_xlim)
    ax_resid.set_ylim(-0.14, 0.22)
    ax_resid.set_ylabel("Residual")
    ax_resid.set_xlabel(wave_label)
    ax_resid.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_resid.yaxis.set_minor_locator(AutoMinorLocator(4))

    plot_components(ax_b, wave_grid, flux, recon_np, host_np, agn_np, sigma_mask, zoom_b)
    ax_b.text(
        0.0,
        1.065,
        "(b)",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        clip_on=False,
    )
    ax_b.set_title("Fe/metal absorption subwindow", pad=8)
    ax_b.set_xlabel(wave_label)
    ax_b.set_ylim(0.12, 1.32)

    plot_components(ax_c, wave_grid, flux, recon_np, host_np, agn_np, sigma_mask, zoom_c)
    ax_c.text(
        0.0,
        1.065,
        "(c)",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        clip_on=False,
    )
    ax_c.set_title(ZOOM_C_TITLE, pad=8)
    ax_c.set_xlabel(wave_label)
    ax_c.set_ylim(0.00, 1.36)

    for ax in (ax_main, ax_resid, ax_b, ax_c):
        ax.tick_params(which="major", length=4.0, width=0.9)
        ax.tick_params(which="minor", length=2.2, width=0.7)

    handles = [
        Line2D([0], [0], color="0.10", lw=1.4, label="SDSS flux"),
        Line2D([0], [0], color="#d62728", lw=1.6, label="model reconstruction"),
        Line2D([0], [0], color="#1f77b4", lw=1.5, label="host component"),
        Line2D([0], [0], color="#f5a623", lw=1.5, label="AGN component"),
        Line2D([0], [0], color="#9bcf9b", lw=6.0, alpha=0.55, label=r"$\sigma_\star$ window"),
    ]
    suffix = TITLE_SUFFIX if TITLE_SUFFIX else target_quality_label(row)
    fig.suptitle(f"{TARGET_NAME} ({suffix})", y=0.985, fontsize=11.0)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.0,
    )

    for output in (OUT_ROOT, OUT_DRAFT):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"saved: {output}")
    plt.close(fig)
    print(
        json.dumps(
            {
                "sdss_name": TARGET_NAME,
                "device": str(device),
                "channels": channels,
                "pred_sigma_model_forward": float(pred.cpu().numpy().reshape(-1)[0]),
                "host_fraction_model_forward": float(host_frac.cpu().numpy().reshape(-1)[0]),
                "median_snr": meta["median_snr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    args = parse_args()
    configure_paths(
        run_tag=args.run_tag,
        target_name=args.target_name,
        h5_path=args.h5,
        model_path=args.model,
        calibration_json=args.calibration_json,
        validation_csv=args.validation_csv,
        output_pdf=args.output_pdf,
        draft_pdf=args.draft_pdf,
        truth_label=args.truth_label,
        title_suffix=args.title_suffix,
        zoom_c_max=args.zoom_c_max,
        zoom_c_title=args.zoom_c_title,
    )
    main()
