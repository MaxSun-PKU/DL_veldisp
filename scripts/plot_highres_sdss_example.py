#!/usr/bin/env python
"""Make a qualitative high-resolution-vs-SDSS example figure.

The Keck/LRIS panel is cropped from the published Bennert+2010 EPS figure,
because the machine-readable 1D Keck spectra are not present in this checkout.
The SDSS spectrum is redrawn from the matched SDSS FITS file.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "external_validation" / "highres_truth"


def read_sdss(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = {name.lower(): name for name in data.names}
        wave = 10.0 ** np.asarray(data[names["loglam"]], dtype=float)
        flux = np.asarray(data[names["flux"]], dtype=float)
        ivar = np.asarray(data[names["ivar"]], dtype=float)
    return wave, flux, ivar


def render_eps_panel(eps_path: Path, rendered_png: Path) -> None:
    rendered_png.parent.mkdir(parents=True, exist_ok=True)
    if rendered_png.exists() and rendered_png.stat().st_size > 0:
        return
    cmd = [
        "gs",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        "-r260",
        "-dEPSCrop",
        f"-sOutputFile={rendered_png}",
        str(eps_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


PANEL_CROPS = {
    "0206-0017": (555, 85, 932, 345),
    "1611+5211": (555, 1282, 932, 1595),
}


def crop_bennert_panel(rendered_png: Path, crop_png: Path, object_id: str) -> None:
    crop_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(rendered_png).convert("RGB")
    # Crop a panel from Bennert+2010 Fig. 4.
    # Coordinates are defined on the Ghostscript-rendered f4.eps image.
    if object_id not in PANEL_CROPS:
        raise ValueError(f"No Bennert Fig. 4 crop is configured for {object_id}.")
    w, h = image.size
    scale_x = w / 1754.0
    scale_y = h / 1754.0
    x0, y0, x1, y1 = PANEL_CROPS[object_id]
    box = (int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y))
    image.crop(box).save(crop_png)


def robust_norm(wave_rest: np.ndarray, flux: np.ndarray, lo: float, hi: float) -> np.ndarray:
    region = (wave_rest >= lo) & (wave_rest <= hi) & np.isfinite(flux)
    scale = np.nanmedian(flux[region]) if np.any(region) else np.nanmedian(flux[np.isfinite(flux)])
    if not np.isfinite(scale) or abs(scale) < 1e-6:
        scale = 1.0
    return flux / scale


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 1.0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", default="1611+5211")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE / "highres_sigma_sdss_v12_main_manifest.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["literature_id"].eq(args.object_id)].iloc[0]
    z = float(row["z"])
    sigma_truth = float(row["sigma_truth_kms"])
    sigma_err = float(row["sigma_truth_err_kms"])
    sdss_path = Path(row["local_path"])

    safe_id = args.object_id.replace("+", "p").replace("-", "m")
    if args.output_pdf is None:
        args.output_pdf = PROJECT_ROOT / "figures" / f"highres_vs_sdss_{safe_id}_example.pdf"
    if args.output_png is None:
        args.output_png = PROJECT_ROOT / "tmp" / "highres_compare" / f"highres_vs_sdss_{safe_id}_example.png"

    tmp_dir = PROJECT_ROOT / "tmp" / "highres_compare"
    rendered_png = tmp_dir / "bennert2010_fig4_cat.png"
    crop_png = tmp_dir / f"bennert2010_fig4_{safe_id}_crop.png"
    render_eps_panel(BASE / "bennert2010_source" / "f4.eps", rendered_png)
    crop_bennert_panel(rendered_png, crop_png, args.object_id)

    obs_wave, flux, ivar = read_sdss(sdss_path)
    rest_wave = obs_wave / (1.0 + z)
    valid = np.isfinite(rest_wave) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    region = (rest_wave >= 8460.0) & (rest_wave <= 8710.0) & valid
    wave_r = rest_wave[region]
    flux_n = robust_norm(rest_wave, flux, 8480.0, 8700.0)[region]
    ivar_n = ivar[region] * np.nanmedian(flux[region]) ** 2 if np.any(region) else ivar[region]
    err_n = np.zeros_like(flux_n)
    good_err = ivar_n > 0
    err_n[good_err] = 1.0 / np.sqrt(ivar_n[good_err])
    smooth = gaussian_filter1d(flux_n, 1.0) if flux_n.size else flux_n

    setup_style()
    fig = plt.figure(figsize=(8.2, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.15])
    ax_img = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    panel = Image.open(crop_png)
    ax_img.imshow(panel)
    ax_img.axis("off")
    ax_img.set_title("(a) Keck/LRIS Ca II triplet panel")
    ax_img.text(
        0.02,
        -0.08,
        "Bennert+2010 Fig. 4; black=data, red=model",
        transform=ax_img.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
    )

    if wave_r.size:
        ax.fill_between(wave_r, flux_n - err_n, flux_n + err_n, color="0.86", lw=0, label=r"SDSS $1\sigma$")
        ax.plot(wave_r, flux_n, color="0.15", lw=0.7, alpha=0.78, label="SDSS spectrum")
        ax.plot(wave_r, smooth, color="#2c7fb8", lw=1.2, label="SDSS smoothed")
    for line, label in [(8498.0, "Ca II"), (8542.0, "Ca II"), (8662.0, "Ca II")]:
        ax.axvline(line, color="#b24c3b", lw=0.8, ls="--", alpha=0.75)
    ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
    ax.set_xlim(8465.0, 8700.0)
    if wave_r.size:
        finite = np.isfinite(flux_n)
        lo, hi = np.nanpercentile(flux_n[finite], [1, 99]) if np.any(finite) else (0.7, 1.3)
        pad = 0.18 * max(hi - lo, 0.2)
        ax.set_ylim(max(0.2, lo - pad), hi + pad)
    ax.set_xlabel(r"Rest wavelength [$\AA$]")
    ax.set_ylabel("Normalized flux")
    ax.set_title("(b) Matched SDSS fiber spectrum")
    ax.legend(frameon=False, loc="lower left", fontsize=8.0)
    ax.text(
        0.98,
        0.95,
        rf"{args.object_id}, $z={z:.4f}$" + "\n" + rf"$\sigma_{{\rm HR}}={sigma_truth:.0f}\pm{sigma_err:.0f}$ km s$^{{-1}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.5),
    )

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    fig.savefig(args.output_png, dpi=220)
    plt.close(fig)
    print(args.output_pdf)
    print(args.output_png)


if __name__ == "__main__":
    main()
