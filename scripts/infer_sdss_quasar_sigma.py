"""
Run HybridAGNNet inference on real SDSS quasar spectra.

The script reads SDSS/BOSS/eBOSS spec*.fits files, shifts them to the rest
frame using a supplied redshift, resamples them onto the model wavelength
grid, constructs the same input channels used in training, and writes a
CSV/PDF summary. The wavelength grid, SPS basis, Fe II templates, and physical
configuration can come from the full training HDF5 or from the compact file
written by export_inference_assets.py. When the COADD table provides WDISP, it
is converted to instrumental Gaussian sigma in km/s and can be used as an
input channel. If a calibration JSON from evaluate_hybrid_sdss.py is
supplied, the output includes raw and affine-calibrated sigma predictions,
calibrated error bars, science-grade quality flags, and explicit failure
reasons.  The default reported central value is the raw network sigma; use
--central-calibration affine only after validating that affine mapping on the
same spectral domain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_sdss_quasar_mocks import (  # noqa: E402
    AGN_EMISSION_MASKS,
    GREENE_INTERNAL_MASKS,
    GREENE_SIGMA_WINDOWS,
    PARK_CONTINUUM_WINDOWS,
    median_in_window,
)
from physmodel import HybridAGNNet  # noqa: E402
from sdss_reference_spectra import estimate_inst_sigma_kms_from_columns  # noqa: E402
from train_hybrid_sdss import load_h5_assets, load_h5_physical_config, velocity_pixel_scale  # noqa: E402
from evaluate_hybrid_sdss import estimate_quality_risk  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    assets = parser.add_mutually_exclusive_group(required=True)
    assets.add_argument(
        "--assets",
        dest="h5",
        help="Compact inference-assets HDF5 containing the wavelength grid, templates, and model configuration.",
    )
    assets.add_argument(
        "--h5",
        dest="h5",
        help="Full mock-training HDF5. Retained as a backward-compatible alternative to --assets.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest-csv", default=None, help="CSV with local_path, z, and optionally sdss_name columns.")
    parser.add_argument("--input-dir", default=None, help="Directory of spec*.fits files; used if no manifest is supplied.")
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument(
        "--central-calibration",
        choices=("raw", "affine"),
        default="raw",
        help=(
            "Which central sigma value to expose as pred_sigma_kms. "
            "Use raw for real-SDSS sanity checks unless the affine calibration "
            "has been validated over the same sigma/domain; affine is still "
            "written as pred_sigma_affine_kms when calibration is available."
        ),
    )
    parser.add_argument("--literature-csv", default=None, help="Optional CSV with sdss_name,sigma_lit_kms,sigma_lit_err_kms.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--channels", default="auto")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--quality-max-error", type=float, default=None)
    parser.add_argument("--quality-min-host", type=float, default=None)
    parser.add_argument("--quality-min-sigma-mask-fraction", type=float, default=None)
    parser.add_argument("--quality-min-valid-fraction", type=float, default=None)
    parser.add_argument("--quality-min-resolution-ratio", type=float, default=None)
    parser.add_argument("--quality-high-conf-min-resolution-ratio", type=float, default=None)
    parser.add_argument("--quality-max-fit-recon-mae", type=float, default=None)
    parser.add_argument("--quality-max-sigma-recon-mae", type=float, default=None)
    parser.add_argument("--quality-max-sigma-residual-to-feature", type=float, default=None)
    parser.add_argument("--quality-min-host-sigma-fraction", type=float, default=None)
    parser.add_argument("--quality-min-absorption-depth", type=float, default=None)
    parser.add_argument("--quality-risk-gold-max-expected-error", type=float, default=None)
    parser.add_argument("--quality-risk-silver-max-expected-error", type=float, default=None)
    parser.add_argument("--systematic-frac", type=float, default=None)
    parser.add_argument("--lowres-extra-frac", type=float, default=None)
    return parser.parse_args()


def _import_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("Real SDSS inference requires astropy in the latis-mps environment.") from exc
    return fits


def _column_lookup(names: Iterable[str]) -> dict[str, str]:
    return {name.lower(): name for name in names}


def estimate_log_l5100(observed_wave: np.ndarray, flux: np.ndarray, redshift: float) -> float:
    """Estimate log10(lambda L_lambda at rest 5100 A) from SDSS flux units.

    SDSS spectra are in 1e-17 erg s^-1 cm^-2 A^-1.  This total-continuum
    estimate is used only as an optional auxiliary channel for covariant-AGN
    mock training; it is not a replacement for host/AGN decomposition.
    """
    try:
        from astropy.cosmology import FlatLambdaCDM
    except Exception:
        return 45.0
    if redshift <= 0 or not np.isfinite(redshift):
        return 45.0
    target_obs = 5100.0 * (1.0 + redshift)
    finite = np.isfinite(observed_wave) & np.isfinite(flux) & (observed_wave > 0)
    if np.count_nonzero(finite) < 2:
        return 45.0
    if target_obs < np.nanmin(observed_wave[finite]) or target_obs > np.nanmax(observed_wave[finite]):
        return 45.0
    f_lambda = float(np.interp(target_obs, observed_wave[finite], flux[finite])) * 1e-17
    if not np.isfinite(f_lambda) or f_lambda <= 0:
        return 45.0
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    dl_cm = float(cosmo.luminosity_distance(float(redshift)).to("cm").value)
    lambda_l_lambda = 4.0 * np.pi * dl_cm**2 * target_obs * f_lambda
    if not np.isfinite(lambda_l_lambda) or lambda_l_lambda <= 0:
        return 45.0
    return float(np.clip(np.log10(lambda_l_lambda), 42.0, 48.5))


def _find_spectrum_table(hdul):
    for hdu in hdul[1:]:
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if not names:
            continue
        lookup = _column_lookup(names)
        if "flux" in lookup and "ivar" in lookup and ("loglam" in lookup or "wave" in lookup or "wavelength" in lookup):
            return hdu, lookup
    raise ValueError("No SDSS-like flux/ivar/loglam table found")


def _read_redshift_from_fits(hdul) -> float | None:
    for hdu in hdul:
        header = getattr(hdu, "header", {})
        for key in ("Z", "REDSHIFT"):
            if key in header:
                try:
                    return float(header[key])
                except (TypeError, ValueError):
                    pass
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if not names:
            continue
        lookup = _column_lookup(names)
        if "z" in lookup and len(data) > 0:
            try:
                return float(data[lookup["z"]][0])
            except (TypeError, ValueError, IndexError):
                pass
    return None


def read_sdss_spectrum(path: str | Path) -> dict[str, np.ndarray | float | None]:
    fits = _import_fits()
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu, lookup = _find_spectrum_table(hdul)
        data = hdu.data
        flux = np.asarray(data[lookup["flux"]], dtype=np.float32)
        ivar = np.asarray(data[lookup["ivar"]], dtype=np.float32)
        if "loglam" in lookup:
            observed_wave = 10.0 ** np.asarray(data[lookup["loglam"]], dtype=np.float64)
        elif "wave" in lookup:
            observed_wave = np.asarray(data[lookup["wave"]], dtype=np.float64)
        else:
            observed_wave = np.asarray(data[lookup["wavelength"]], dtype=np.float64)
        inst_sigma_kms = estimate_inst_sigma_kms_from_columns(data, lookup, observed_wave)

        valid = np.isfinite(observed_wave) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
        for mask_name in ("and_mask", "andmask"):
            if mask_name in lookup:
                valid &= np.asarray(data[lookup[mask_name]]) == 0
                break
        order = np.argsort(observed_wave)
        return {
            "observed_wave": observed_wave[order].astype(np.float32),
            "flux": flux[order].astype(np.float32),
            "ivar": ivar[order].astype(np.float32),
            "valid": valid[order],
            "inst_sigma_kms": inst_sigma_kms[order].astype(np.float32),
            "redshift": _read_redshift_from_fits(hdul),
        }


def load_manifest(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
    elif args.input_dir:
        for path in sorted(Path(args.input_dir).rglob("spec*.fits*")):
            rows.append({"local_path": str(path), "sdss_name": path.stem, "z": ""})
    else:
        raise ValueError("Pass --manifest-csv or --input-dir")
    if args.max_spectra is not None:
        rows = rows[: args.max_spectra]
    return rows


def build_rest_frame_channels(
    observed_wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    observed_valid: np.ndarray,
    inst_sigma_kms: np.ndarray | None,
    redshift: float,
    wave_grid: np.ndarray,
    channels: tuple[str, ...],
    fallback_inst_sigma_kms: float = 70.0,
    target_common_resolution_sigma_kms: float = 0.0,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    observed_wave_for_lum = np.asarray(observed_wave, dtype=np.float64)
    flux_for_lum = np.asarray(flux, dtype=np.float32)
    rest_wave = observed_wave / (1.0 + redshift)
    order = np.argsort(rest_wave)
    rest_wave = rest_wave[order]
    flux = flux[order]
    ivar = ivar[order]
    observed_valid = observed_valid[order]
    if inst_sigma_kms is not None:
        inst_sigma_kms = np.asarray(inst_sigma_kms, dtype=np.float32)[order]

    interp_flux = np.interp(wave_grid, rest_wave, flux, left=np.nan, right=np.nan).astype(np.float32)
    interp_ivar = np.interp(wave_grid, rest_wave, ivar, left=0.0, right=0.0).astype(np.float32)
    interp_valid = np.interp(wave_grid, rest_wave, observed_valid.astype(np.float32), left=0.0, right=0.0) > 0.5
    if inst_sigma_kms is not None:
        interp_inst_sigma = np.interp(wave_grid, rest_wave, inst_sigma_kms, left=np.nan, right=np.nan).astype(np.float32)
    else:
        interp_inst_sigma = np.full_like(wave_grid, np.nan, dtype=np.float32)
    valid = interp_valid & np.isfinite(interp_flux) & np.isfinite(interp_ivar) & (interp_ivar > 0)

    scale = median_in_window(wave_grid[valid], interp_flux[valid]) if np.any(valid) else 1.0
    flux_norm = np.zeros_like(interp_flux, dtype=np.float32)
    flux_norm[valid] = interp_flux[valid] / max(scale, 1e-6)
    ivar_norm_physical = np.zeros_like(interp_ivar, dtype=np.float32)
    ivar_norm_physical[valid] = interp_ivar[valid] * max(scale, 1e-6) ** 2

    fit_mask = valid.copy()
    for center, half_width in AGN_EMISSION_MASKS:
        fit_mask &= np.abs(wave_grid - center) > half_width

    sigma_mask = np.zeros_like(valid, dtype=bool)
    region_mask_fractions = {}
    for lo, hi in GREENE_SIGMA_WINDOWS.values():
        sigma_mask |= (wave_grid >= lo) & (wave_grid <= hi)
    sigma_mask &= valid
    for lo, hi in GREENE_INTERNAL_MASKS:
        sigma_mask &= ~((wave_grid >= lo) & (wave_grid <= hi))
    for name, (lo, hi) in GREENE_SIGMA_WINDOWS.items():
        key = "fe_mask_fraction" if name == "fe_5250_5820" else f"{name}_mask_fraction"
        region_mask_fractions[key] = float(np.mean((wave_grid >= lo) & (wave_grid <= hi) & sigma_mask))

    continuum_mask = np.zeros_like(valid, dtype=bool)
    for lo, hi in PARK_CONTINUUM_WINDOWS:
        continuum_mask |= (wave_grid >= lo) & (wave_grid <= hi)
    continuum_mask &= fit_mask

    positive_ivar = ivar_norm_physical[ivar_norm_physical > 0]
    model_ivar = ivar_norm_physical.copy()
    if positive_ivar.size:
        model_ivar = model_ivar / max(float(np.median(positive_ivar)), 1e-6)
    model_ivar = np.clip(model_ivar, 0.0, 10.0).astype(np.float32)

    if np.any(sigma_mask):
        values = flux_norm[sigma_mask]
        center = float(np.median(values))
        sigma_scale = float(np.median(np.abs(values - center)))
        sigma_flux = np.clip((flux_norm - center) / max(sigma_scale, 1e-3), -8.0, 8.0) * sigma_mask.astype(np.float32)
    else:
        sigma_flux = np.zeros_like(flux_norm, dtype=np.float32)

    inst_valid = valid & np.isfinite(interp_inst_sigma) & (interp_inst_sigma > 0)
    inst_sigma_scalar = (
        float(np.median(interp_inst_sigma[sigma_mask & inst_valid]))
        if np.any(sigma_mask & inst_valid)
        else float(np.median(interp_inst_sigma[inst_valid]))
        if np.any(inst_valid)
        else float(fallback_inst_sigma_kms)
    )
    was_common_resolution_degraded = False
    if target_common_resolution_sigma_kms and target_common_resolution_sigma_kms > inst_sigma_scalar:
        dv_pixel = 299792.458 * np.median(np.diff(np.log(wave_grid)))
        sigma_extra = np.sqrt(max(float(target_common_resolution_sigma_kms) ** 2 - inst_sigma_scalar**2, 0.0))
        sigma_pix = max(sigma_extra / max(float(dv_pixel), 1e-6), 0.0)
        if sigma_pix > 0.05:
            fill_value = float(np.nanmedian(flux_norm[valid])) if np.any(valid) else 0.0
            filled_flux = flux_norm.copy()
            filled_flux[~valid] = fill_value
            smoothed = gaussian_filter1d(filled_flux, sigma_pix, mode="nearest")
            flux_norm[valid] = smoothed[valid].astype(np.float32)
            inst_sigma_scalar = float(target_common_resolution_sigma_kms)
            was_common_resolution_degraded = True
            if np.any(sigma_mask):
                values = flux_norm[sigma_mask]
                center = float(np.median(values))
                sigma_scale = float(np.median(np.abs(values - center)))
                sigma_flux = np.clip((flux_norm - center) / max(sigma_scale, 1e-3), -8.0, 8.0) * sigma_mask.astype(np.float32)

    inst_sigma_channel = np.full_like(flux_norm, inst_sigma_scalar / 100.0, dtype=np.float32)
    if target_common_resolution_sigma_kms <= 0 and np.any(inst_valid):
        inst_sigma_channel = np.where(
            np.isfinite(interp_inst_sigma) & (interp_inst_sigma > 0),
            interp_inst_sigma / 100.0,
            inst_sigma_scalar / 100.0,
        ).astype(np.float32)
    log_l5100 = estimate_log_l5100(observed_wave_for_lum, flux_for_lum, redshift)
    log_l5100_channel = np.full_like(flux_norm, (log_l5100 - 45.0) / 2.0, dtype=np.float32)

    built = []
    if region_mask_fractions.get("cat_mask_fraction", 0.0) > 0.002:
        region_code = 0
    elif region_mask_fractions.get("fe_mask_fraction", 0.0) > 0.002:
        region_code = 1
    else:
        region_code = 2
    for channel in channels:
        if channel == "flux":
            built.append(flux_norm)
        elif channel == "ivar":
            built.append(model_ivar)
        elif channel == "mask":
            built.append(fit_mask.astype(np.float32))
        elif channel == "sigma_flux":
            built.append(sigma_flux.astype(np.float32))
        elif channel == "sigma_mask":
            built.append(sigma_mask.astype(np.float32))
        elif channel in ("inst_sigma", "instrument_sigma"):
            built.append(np.clip(inst_sigma_channel, 0.0, 3.0).astype(np.float32))
        elif channel in ("region_cat", "cat_region"):
            built.append(np.full_like(flux_norm, 1.0 if region_code == 0 else 0.0, dtype=np.float32))
        elif channel in ("region_fe", "fe_region"):
            built.append(np.full_like(flux_norm, 1.0 if region_code == 1 else 0.0, dtype=np.float32))
        elif channel in ("region_cahk", "region_ca_hk", "ca_hk_region"):
            built.append(np.full_like(flux_norm, 1.0 if region_code == 2 else 0.0, dtype=np.float32))
        elif channel in ("agn_log_l5100", "log_l5100", "l5100"):
            built.append(np.clip(log_l5100_channel, -2.0, 2.0).astype(np.float32))
        else:
            raise ValueError(f"Unsupported inference channel: {channel}")

    snr_region = (wave_grid > 5050.0) & (wave_grid < 5150.0) & valid
    if not np.any(snr_region):
        snr_region = valid
    snr_values = np.abs(flux_norm[snr_region]) * np.sqrt(np.clip(ivar_norm_physical[snr_region], 0.0, None))
    median_snr = float(np.median(snr_values[np.isfinite(snr_values)])) if snr_values.size else 0.0
    meta = {
        "valid_fraction": float(np.mean(valid)),
        "fit_mask_fraction": float(np.mean(fit_mask)),
        "sigma_mask_fraction": float(np.mean(sigma_mask)),
        "continuum_mask_fraction": float(np.mean(continuum_mask)),
        "median_snr": median_snr,
        "median_inst_sigma_kms": inst_sigma_scalar,
        "log_l5100": log_l5100,
        "common_resolution_degraded": float(was_common_resolution_degraded),
        "region_code": float(region_code),
        "scale": float(scale),
        "flux_norm": flux_norm,
        "ivar_norm_physical": ivar_norm_physical,
        "valid": valid,
        **region_mask_fractions,
    }
    return np.stack(built, axis=0).astype(np.float32), meta


def load_calibration(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_literature(path: str | None) -> dict[str, dict[str, float]]:
    if not path or not Path(path).exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("sdss_name") or "").strip()
            value = row.get("sigma_lit_kms", "")
            if not name or value in ("", None):
                continue
            try:
                sigma_lit = float(value)
            except ValueError:
                continue
            err_value = row.get("sigma_lit_err_kms", "")
            try:
                sigma_err = float(err_value) if err_value not in ("", None) else np.nan
            except ValueError:
                sigma_err = np.nan
            rows[name] = {"sigma_lit_kms": sigma_lit, "sigma_lit_err_kms": sigma_err}
    return rows


def robust_absorption_depth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return 0.0
    return float(np.nanpercentile(values, 80.0) - np.nanpercentile(values, 20.0))


def reconstruction_diagnostics(
    x: np.ndarray,
    recon: np.ndarray,
    agn_part: np.ndarray,
    stellar_part: np.ndarray,
) -> dict[str, float]:
    flux = np.asarray(x[0], dtype=np.float32)
    fit_mask = np.asarray(x[2] > 0.5, dtype=bool) if x.shape[0] > 2 else np.isfinite(flux)
    sigma_mask = np.asarray(x[-1] > 0.5, dtype=bool)
    valid_fit = fit_mask & np.isfinite(flux) & np.isfinite(recon)
    valid_sigma = sigma_mask & np.isfinite(flux) & np.isfinite(recon)
    resid = flux - recon
    fit_mae = float(np.nanmean(np.abs(resid[valid_fit]))) if np.any(valid_fit) else np.nan
    sigma_mae = float(np.nanmean(np.abs(resid[valid_sigma]))) if np.any(valid_sigma) else np.nan
    sigma_std = float(np.nanstd(resid[valid_sigma])) if np.any(valid_sigma) else np.nan
    host_sigma_power = float(np.nansum(np.abs(stellar_part[valid_sigma]))) if np.any(valid_sigma) else 0.0
    agn_sigma_power = float(np.nansum(np.abs(agn_part[valid_sigma]))) if np.any(valid_sigma) else 0.0
    host_sigma_fraction = host_sigma_power / max(host_sigma_power + agn_sigma_power, 1e-6)
    host_absorption_depth = robust_absorption_depth(stellar_part[valid_sigma]) if np.any(valid_sigma) else 0.0
    flux_absorption_depth = robust_absorption_depth(flux[valid_sigma]) if np.any(valid_sigma) else 0.0
    residual_to_feature = sigma_mae / max(flux_absorption_depth, 1e-4) if np.isfinite(sigma_mae) else np.inf
    return {
        "fit_recon_mae": fit_mae,
        "sigma_recon_mae": sigma_mae,
        "sigma_recon_resid_std": sigma_std,
        "pred_host_sigma_fraction": float(np.clip(host_sigma_fraction, 0.0, 1.0)),
        "host_absorption_depth_proxy": host_absorption_depth,
        "flux_absorption_depth_proxy": flux_absorption_depth,
        "sigma_residual_to_feature": float(residual_to_feature),
    }


def assign_science_grade(
    *,
    high_confidence: bool,
    pred_sigma_err: float,
    max_error: float | None,
    pred_host: float,
    min_host: float,
    resolution_ratio: float,
    min_resolution_ratio: float,
    sigma_recon_mae: float,
    max_sigma_recon_mae: float,
    host_sigma_fraction: float,
    min_host_sigma_fraction: float,
    absorption_depth: float,
    min_absorption_depth: float,
) -> str:
    if high_confidence:
        return "A"
    soft_checks = [
        max_error is None or pred_sigma_err <= 1.35 * float(max_error),
        pred_host >= 0.85 * float(min_host),
        resolution_ratio >= 0.90 * float(min_resolution_ratio),
        np.isfinite(sigma_recon_mae) and sigma_recon_mae <= 1.35 * float(max_sigma_recon_mae),
        host_sigma_fraction >= 0.85 * float(min_host_sigma_fraction),
        absorption_depth >= 0.80 * float(min_absorption_depth),
    ]
    return "B" if sum(bool(item) for item in soft_checks) >= 5 else "C"


def build_science_quality_report(
    *,
    pred_sigma: float,
    pred_sigma_err: float,
    max_error: float | None,
    pred_host: float,
    min_host: float,
    sigma_mask_fraction: float,
    min_sigma_mask: float,
    valid_fraction: float,
    min_valid: float,
    resolution_ratio: float,
    min_resolution_ratio: float,
    advisory_resolution_ratio: float,
    high_conf_min_resolution_ratio: float,
    fit_recon_mae: float,
    max_fit_recon_mae: float,
    sigma_recon_mae: float,
    max_sigma_recon_mae: float,
    sigma_residual_to_feature: float,
    max_sigma_residual_to_feature: float,
    host_sigma_fraction: float,
    min_host_sigma_fraction: float,
    absorption_depth: float,
    min_absorption_depth: float,
    sigma_min: float,
    sigma_max: float,
    risk_expected_abs_error: float | None = None,
    risk_p_abs_error_le_40: float | None = None,
    risk_p_abs_error_le_60: float | None = None,
    risk_gold_max_expected_error: float = 30.0,
    risk_silver_max_expected_error: float = 45.0,
) -> dict[str, float | str | bool | int]:
    """Return an auditable real-SDSS measurement-quality summary."""
    label_pad = max(10.0, 0.05 * (float(sigma_max) - float(sigma_min)))
    below_training = bool(pred_sigma < float(sigma_min))
    above_training = bool(pred_sigma > float(sigma_max))
    near_floor = bool(pred_sigma <= float(sigma_min) + label_pad)
    near_ceiling = bool(pred_sigma >= float(sigma_max) - label_pad)
    outside_training = bool(below_training or above_training)
    risk_available = bool(risk_expected_abs_error is not None and np.isfinite(risk_expected_abs_error))
    risk_p40 = float(risk_p_abs_error_le_40) if risk_p_abs_error_le_40 is not None and np.isfinite(risk_p_abs_error_le_40) else np.nan
    risk_p60 = float(risk_p_abs_error_le_60) if risk_p_abs_error_le_60 is not None and np.isfinite(risk_p_abs_error_le_60) else np.nan
    risk_gold_ok = (
        (not risk_available)
        or (
            float(risk_expected_abs_error) <= float(risk_gold_max_expected_error)
            and (not np.isfinite(risk_p40) or risk_p40 >= 0.60)
        )
    )
    risk_silver_ok = (
        (not risk_available)
        or (
            float(risk_expected_abs_error) <= float(risk_silver_max_expected_error)
            and (not np.isfinite(risk_p60) or risk_p60 >= 0.70)
        )
    )

    checks = {
        "error": bool(max_error is None or pred_sigma_err <= float(max_error)),
        "host_fraction": bool(pred_host >= float(min_host)),
        "sigma_window_coverage": bool(sigma_mask_fraction >= float(min_sigma_mask)),
        "valid_pixel_fraction": bool(valid_fraction >= float(min_valid)),
        "resolution": bool(resolution_ratio >= float(min_resolution_ratio)),
        "fit_reconstruction": bool(np.isfinite(fit_recon_mae) and fit_recon_mae <= float(max_fit_recon_mae)),
        "sigma_reconstruction": bool(np.isfinite(sigma_recon_mae) and sigma_recon_mae <= float(max_sigma_recon_mae)),
        "sigma_residual_to_feature": bool(
            np.isfinite(sigma_residual_to_feature)
            and sigma_residual_to_feature <= float(max_sigma_residual_to_feature)
        ),
        "host_sigma_window_signal": bool(host_sigma_fraction >= float(min_host_sigma_fraction)),
        "absorption_depth": bool(absorption_depth >= float(min_absorption_depth)),
        "empirical_risk": bool(risk_silver_ok),
        "training_label_range": bool(not outside_training),
    }
    soft_checks = {
        "error": bool(max_error is None or pred_sigma_err <= 1.35 * float(max_error)),
        "host_fraction": bool(pred_host >= 0.85 * float(min_host)),
        "sigma_window_coverage": bool(sigma_mask_fraction >= 0.85 * float(min_sigma_mask)),
        "valid_pixel_fraction": bool(valid_fraction >= 0.85 * float(min_valid)),
        "resolution": bool(resolution_ratio >= 0.75 * float(min_resolution_ratio)),
        "fit_reconstruction": bool(np.isfinite(fit_recon_mae) and fit_recon_mae <= 1.35 * float(max_fit_recon_mae)),
        "sigma_reconstruction": bool(np.isfinite(sigma_recon_mae) and sigma_recon_mae <= 1.35 * float(max_sigma_recon_mae)),
        "sigma_residual_to_feature": bool(
            np.isfinite(sigma_residual_to_feature)
            and sigma_residual_to_feature <= 1.35 * float(max_sigma_residual_to_feature)
        ),
        "host_sigma_window_signal": bool(host_sigma_fraction >= 0.85 * float(min_host_sigma_fraction)),
        "absorption_depth": bool(absorption_depth >= 0.80 * float(min_absorption_depth)),
        "empirical_risk": bool(
            (not risk_available)
            or float(risk_expected_abs_error) <= 1.25 * float(risk_silver_max_expected_error)
            or (np.isfinite(risk_p60) and risk_p60 >= 0.60)
        ),
        "training_label_range": bool(not outside_training),
    }
    pass_count = int(sum(checks.values()))
    soft_pass_count = int(sum(soft_checks.values()))
    core_failure_names = (
        "resolution",
        "fit_reconstruction",
        "sigma_reconstruction",
        "sigma_residual_to_feature",
        "host_sigma_window_signal",
        "absorption_depth",
        "empirical_risk",
    )
    core_hard_failure_count = int(sum(not checks[name] for name in core_failure_names if name in checks))
    strict_checks = {
        **checks,
        "high_conf_resolution": bool(resolution_ratio >= float(high_conf_min_resolution_ratio)),
        "gold_empirical_risk": bool(risk_gold_ok),
        "training_label_margin": bool(not near_floor and not near_ceiling and not outside_training),
    }
    strict_pass_count = int(sum(strict_checks.values()))
    high_confidence = bool(all(strict_checks.values()))
    min_soft_pass = max(7, int(np.ceil(0.75 * len(soft_checks))))
    grade = (
        "A"
        if high_confidence
        else ("B" if soft_pass_count >= min_soft_pass and core_hard_failure_count <= 1 and not outside_training else "C")
    )
    science_ready = bool(grade in ("A", "B") and not outside_training)
    publication_ready = bool(science_ready and high_confidence)
    if publication_ready:
        measurement_tier = "gold"
    elif science_ready and not outside_training:
        measurement_tier = "silver"
    else:
        measurement_tier = "low_confidence"
    failed = [name for name, ok in checks.items() if not ok]
    high_conf_failed = [name for name, ok in strict_checks.items() if not ok]
    advisory_flags = []
    if resolution_ratio < float(advisory_resolution_ratio):
        advisory_flags.append("near_or_below_sdss_resolution")
    if near_floor:
        advisory_flags.append("near_training_sigma_floor")
    if near_ceiling:
        advisory_flags.append("near_training_sigma_ceiling")
    if below_training:
        advisory_flags.append("below_training_sigma_min")
    if above_training:
        advisory_flags.append("above_training_sigma_max")
    return {
        "science_grade": grade,
        "science_ready": science_ready,
        "publication_ready": publication_ready,
        "golden_sample": publication_ready,
        "low_confidence": measurement_tier == "low_confidence",
        "measurement_tier": measurement_tier,
        "high_confidence": high_confidence,
        "quality_pass_count": pass_count,
        "quality_soft_pass_count": soft_pass_count,
        "quality_strict_pass_count": strict_pass_count,
        "quality_core_hard_failure_count": core_hard_failure_count,
        "quality_pass_fraction": float(pass_count / len(checks)),
        "quality_strict_pass_fraction": float(strict_pass_count / len(strict_checks)),
        "quality_failed_checks": ";".join(failed),
        "high_confidence_failed_checks": ";".join(high_conf_failed),
        "quality_advisory_flags": ";".join(advisory_flags),
        "below_training_sigma_min": below_training,
        "above_training_sigma_max": above_training,
        "outside_training_sigma_range": outside_training,
        "near_training_sigma_floor": near_floor,
        "near_training_sigma_ceiling": near_ceiling,
        "advisory_sigma_resolution_ratio": float(advisory_resolution_ratio),
        "high_conf_min_sigma_resolution_ratio": float(high_conf_min_resolution_ratio),
        "max_fit_recon_mae": float(max_fit_recon_mae),
        "max_sigma_residual_to_feature": float(max_sigma_residual_to_feature),
        "risk_model_available": risk_available,
        "risk_gold_max_expected_abs_error_kms": float(risk_gold_max_expected_error),
        "risk_silver_max_expected_abs_error_kms": float(risk_silver_max_expected_error),
    }


def infer_rows(args: argparse.Namespace) -> tuple[list[dict[str, float | str | bool]], dict]:
    wave_grid, template_library, fe_template, _, _, sigma_min, sigma_max = load_h5_assets(args.h5)
    physical_config = load_h5_physical_config(args.h5)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location=device)
    if args.channels == "auto":
        first_conv = state.get("backbone.0.weight")
        input_channels = int(first_conv.shape[1]) if first_conv is not None else 5
        channels = AUTO_CHANNELS.get(input_channels)
        if channels is None:
            raise ValueError(f"Cannot infer channel names for checkpoint with {input_channels} input channels")
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

    calibration = load_calibration(args.calibration_json)
    quality_cuts = dict(calibration.get("quality_cuts", {}))
    real_quality_cuts = dict(calibration.get("real_sdss_quality_cuts", {}))
    if real_quality_cuts:
        quality_cuts.update(real_quality_cuts)
    max_error = args.quality_max_error if args.quality_max_error is not None else quality_cuts.get("max_calibrated_sigma_err_kms")
    min_host = args.quality_min_host if args.quality_min_host is not None else quality_cuts.get("min_pred_host_fraction", 0.45)
    min_sigma_mask = (
        args.quality_min_sigma_mask_fraction
        if args.quality_min_sigma_mask_fraction is not None
        else quality_cuts.get("min_sigma_mask_fraction", 0.015)
    )
    min_valid = args.quality_min_valid_fraction if args.quality_min_valid_fraction is not None else quality_cuts.get("min_valid_fraction", 0.20)
    legacy_resolution_ratio = float(quality_cuts.get("min_sigma_resolution_ratio", 0.80))
    default_real_resolution_ratio = float(quality_cuts.get("real_min_sigma_resolution_ratio", min(1.0, legacy_resolution_ratio)))
    min_resolution_ratio = (
        args.quality_min_resolution_ratio
        if args.quality_min_resolution_ratio is not None
        else default_real_resolution_ratio
    )
    advisory_resolution_ratio = float(quality_cuts.get("advisory_sigma_resolution_ratio", max(1.25, float(min_resolution_ratio))))
    high_conf_min_resolution_ratio = (
        args.quality_high_conf_min_resolution_ratio
        if args.quality_high_conf_min_resolution_ratio is not None
        else quality_cuts.get("high_conf_min_sigma_resolution_ratio", advisory_resolution_ratio)
    )
    max_fit_recon_mae = (
        args.quality_max_fit_recon_mae
        if args.quality_max_fit_recon_mae is not None
        else quality_cuts.get("max_fit_recon_mae", 0.18)
    )
    max_sigma_recon_mae = (
        args.quality_max_sigma_recon_mae
        if args.quality_max_sigma_recon_mae is not None
        else quality_cuts.get("max_sigma_recon_mae", 0.16)
    )
    max_sigma_residual_to_feature = (
        args.quality_max_sigma_residual_to_feature
        if args.quality_max_sigma_residual_to_feature is not None
        else quality_cuts.get("max_sigma_residual_to_feature", 0.85)
    )
    min_host_sigma_fraction = (
        args.quality_min_host_sigma_fraction
        if args.quality_min_host_sigma_fraction is not None
        else quality_cuts.get("min_host_sigma_fraction", 0.42)
    )
    min_absorption_depth = (
        args.quality_min_absorption_depth
        if args.quality_min_absorption_depth is not None
        else quality_cuts.get("min_absorption_depth", 0.025)
    )
    risk_gold_max_expected_error = (
        args.quality_risk_gold_max_expected_error
        if args.quality_risk_gold_max_expected_error is not None
        else quality_cuts.get("risk_gold_max_expected_abs_error_kms", 30.0)
    )
    risk_silver_max_expected_error = (
        args.quality_risk_silver_max_expected_error
        if args.quality_risk_silver_max_expected_error is not None
        else quality_cuts.get("risk_silver_max_expected_abs_error_kms", 45.0)
    )
    quality_risk_model = calibration.get("quality_risk_model", {})
    real_error_model = calibration.get("real_sdss_error_model", {})
    systematic_frac = args.systematic_frac if args.systematic_frac is not None else real_error_model.get("systematic_frac", 0.12)
    lowres_extra_frac = (
        args.lowres_extra_frac
        if args.lowres_extra_frac is not None
        else real_error_model.get("low_resolution_extra_frac", 0.25)
    )

    literature = load_literature(args.literature_csv)
    output_rows = []
    for row in load_manifest(args):
        path = row.get("local_path") or row.get("path") or row.get("filename")
        if not path:
            continue
        spectrum = read_sdss_spectrum(path)
        redshift_text = row.get("z", "")
        redshift = float(redshift_text) if redshift_text not in ("", None) else spectrum["redshift"]
        if redshift is None or not np.isfinite(redshift):
            raise ValueError(f"No redshift available for {path}")

        x, meta = build_rest_frame_channels(
            spectrum["observed_wave"],
            spectrum["flux"],
            spectrum["ivar"],
            spectrum["valid"],
            spectrum.get("inst_sigma_kms"),
            float(redshift),
            wave_grid,
            channels,
            fallback_inst_sigma_kms=physical_config["instrument_sigma_kms"],
            target_common_resolution_sigma_kms=physical_config.get("common_resolution_sigma", 0.0),
        )
        with torch.no_grad():
            tensor = torch.from_numpy(x).unsqueeze(0).to(device)
            pred, logvar, _, host, recon, agn_part, stellar_part = model(tensor)
        pred_sigma_raw = float(pred.cpu().numpy().reshape(-1)[0])
        region_sigma = getattr(model, "last_region_sigma", None)
        if region_sigma is not None:
            region_sigma_values = region_sigma.detach().cpu().numpy().reshape(-1)
        else:
            region_sigma_values = np.full(3, np.nan, dtype=float)
        region_weights = getattr(model, "last_region_weights", None)
        if region_weights is not None:
            region_weight_values = region_weights.detach().cpu().numpy().reshape(-1)
        else:
            region_weight_values = np.full(3, np.nan, dtype=float)
        pred_sigma_err_raw = float(np.exp(0.5 * np.clip(float(logvar.cpu().numpy().reshape(-1)[0]), -6.0, 4.0)) * SIGMA_SCALE_KMS)
        pred_host = float(host.cpu().numpy().reshape(-1)[0])
        recon_diag = reconstruction_diagnostics(
            x,
            recon.cpu().numpy().reshape(-1),
            agn_part.cpu().numpy().reshape(-1),
            stellar_part.cpu().numpy().reshape(-1),
        )

        slope = float(calibration.get("sigma_affine_slope", calibration.get("calibration_slope", 1.0)))
        intercept = float(calibration.get("sigma_affine_intercept", calibration.get("calibration_intercept", 0.0)))
        error_calibration = calibration.get("error_calibration", {})
        err_scale = float(calibration.get("sigma_error_scale", error_calibration.get("scale", 1.0)))
        err_floor = float(calibration.get("sigma_error_floor_kms", error_calibration.get("floor_kms", 0.0)))
        pred_sigma_affine = slope * pred_sigma_raw + intercept
        pred_sigma = pred_sigma_affine if args.central_calibration == "affine" else pred_sigma_raw
        stat_calibrated_err = float(np.sqrt((err_scale * abs(slope) * pred_sigma_err_raw) ** 2 + err_floor**2))
        median_inst_sigma = float(meta["median_inst_sigma_kms"])
        resolution_ratio = pred_sigma_raw / max(median_inst_sigma, 1e-6)
        lowres_strength = float(np.clip((1.5 - resolution_ratio) / 0.5, 0.0, 1.0))
        systematic_floor = float(
            np.sqrt((float(systematic_frac) * pred_sigma) ** 2 + (lowres_strength * float(lowres_extra_frac) * pred_sigma) ** 2)
        )
        pred_sigma_err_pre_risk = float(np.sqrt(stat_calibrated_err**2 + systematic_floor**2))
        risk_input = {
            "pred_host": np.asarray([pred_host], dtype=np.float32),
            "snr": np.asarray([meta["median_snr"]], dtype=np.float32),
            "sigma_mask_fraction": np.asarray([meta["sigma_mask_fraction"]], dtype=np.float32),
            "valid_fraction": np.asarray([meta["valid_fraction"]], dtype=np.float32),
            "sigma_inst_kms": np.asarray([median_inst_sigma], dtype=np.float32),
            "fit_recon_mae": np.asarray([recon_diag["fit_recon_mae"]], dtype=np.float32),
            "sigma_recon_mae": np.asarray([recon_diag["sigma_recon_mae"]], dtype=np.float32),
            "sigma_residual_to_feature": np.asarray([recon_diag["sigma_residual_to_feature"]], dtype=np.float32),
            "pred_host_sigma_fraction": np.asarray([recon_diag["pred_host_sigma_fraction"]], dtype=np.float32),
            "flux_absorption_depth_proxy": np.asarray([recon_diag["flux_absorption_depth_proxy"]], dtype=np.float32),
        }
        risk_report = estimate_quality_risk(
            quality_risk_model,
            risk_input,
            np.asarray([pred_sigma], dtype=np.float32),
            np.asarray([pred_sigma_err_pre_risk], dtype=np.float32),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
        )
        risk_expected_abs_error = float(risk_report["risk_expected_abs_error_kms"][0])
        risk_p68_abs_error = float(risk_report["risk_p68_abs_error_kms"][0])
        risk_p90_abs_error = float(risk_report["risk_p90_abs_error_kms"][0])
        risk_p_abs_error_le_40 = float(risk_report["risk_p_abs_error_le_40"][0])
        risk_p_abs_error_le_60 = float(risk_report["risk_p_abs_error_le_60"][0])
        risk_reliability_score = float(risk_report["risk_reliability_score"][0])
        risk_sigma_floor = risk_p68_abs_error if np.isfinite(risk_p68_abs_error) else 0.0
        pred_sigma_err = float(max(pred_sigma_err_pre_risk, risk_sigma_floor))
        quality_report = build_science_quality_report(
            pred_sigma=pred_sigma,
            pred_sigma_err=pred_sigma_err,
            max_error=max_error,
            pred_host=pred_host,
            min_host=float(min_host),
            sigma_mask_fraction=float(meta["sigma_mask_fraction"]),
            min_sigma_mask=float(min_sigma_mask),
            valid_fraction=float(meta["valid_fraction"]),
            min_valid=float(min_valid),
            resolution_ratio=resolution_ratio,
            min_resolution_ratio=float(min_resolution_ratio),
            advisory_resolution_ratio=float(advisory_resolution_ratio),
            high_conf_min_resolution_ratio=float(high_conf_min_resolution_ratio),
            fit_recon_mae=recon_diag["fit_recon_mae"],
            max_fit_recon_mae=float(max_fit_recon_mae),
            sigma_recon_mae=recon_diag["sigma_recon_mae"],
            max_sigma_recon_mae=float(max_sigma_recon_mae),
            sigma_residual_to_feature=recon_diag["sigma_residual_to_feature"],
            max_sigma_residual_to_feature=float(max_sigma_residual_to_feature),
            host_sigma_fraction=recon_diag["pred_host_sigma_fraction"],
            min_host_sigma_fraction=float(min_host_sigma_fraction),
            absorption_depth=recon_diag["host_absorption_depth_proxy"],
            min_absorption_depth=float(min_absorption_depth),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            risk_expected_abs_error=risk_expected_abs_error,
            risk_p_abs_error_le_40=risk_p_abs_error_le_40,
            risk_p_abs_error_le_60=risk_p_abs_error_le_60,
            risk_gold_max_expected_error=float(risk_gold_max_expected_error),
            risk_silver_max_expected_error=float(risk_silver_max_expected_error),
        )
        high_confidence = bool(quality_report["high_confidence"])
        science_grade = str(quality_report["science_grade"])
        confidence = {"A": "high", "B": "intermediate", "C": "low"}.get(science_grade, "low")

        sdss_name = row.get("sdss_name") or Path(path).stem
        out = {
            "sdss_name": sdss_name,
            "local_path": str(path),
            "redshift": float(redshift),
            "pred_sigma_raw_kms": pred_sigma_raw,
            "pred_sigma_affine_kms": pred_sigma_affine,
            "pred_sigma_kms": pred_sigma,
            "central_calibration": args.central_calibration,
            "pred_sigma_err_kms": pred_sigma_err,
            "pred_sigma_err_raw_kms": pred_sigma_err_raw,
            "stat_calibrated_sigma_err_kms": stat_calibrated_err,
            "systematic_sigma_floor_kms": systematic_floor,
            "pre_risk_sigma_err_kms": pred_sigma_err_pre_risk,
            "risk_sigma_floor_kms": risk_sigma_floor,
            "risk_expected_abs_error_kms": risk_expected_abs_error,
            "risk_p68_abs_error_kms": risk_p68_abs_error,
            "risk_p90_abs_error_kms": risk_p90_abs_error,
            "risk_p_abs_error_le_40": risk_p_abs_error_le_40,
            "risk_p_abs_error_le_60": risk_p_abs_error_le_60,
            "risk_reliability_score": risk_reliability_score,
            "recommended_sigma_kms": pred_sigma,
            "recommended_sigma_err_kms": pred_sigma_err,
            "pred_sigma_cat_kms": float(region_sigma_values[0]) if region_sigma_values.size > 0 else np.nan,
            "pred_sigma_fe_kms": float(region_sigma_values[1]) if region_sigma_values.size > 1 else np.nan,
            "pred_sigma_cahk_kms": float(region_sigma_values[2]) if region_sigma_values.size > 2 else np.nan,
            "region_weight_cat": float(region_weight_values[0]) if region_weight_values.size > 0 else np.nan,
            "region_weight_fe": float(region_weight_values[1]) if region_weight_values.size > 1 else np.nan,
            "region_weight_cahk": float(region_weight_values[2]) if region_weight_values.size > 2 else np.nan,
            "training_sigma_min_kms": float(sigma_min),
            "training_sigma_max_kms": float(sigma_max),
            "pred_host_fraction": pred_host,
            "median_snr": float(meta["median_snr"]),
            "median_inst_sigma_kms": median_inst_sigma,
            "pred_sigma_resolution_ratio": resolution_ratio,
            "near_resolution_limit": bool(resolution_ratio < float(min_resolution_ratio)),
            "below_sdss_resolution": bool(resolution_ratio < float(min_resolution_ratio)),
            "advisory_near_resolution_limit": bool(resolution_ratio < float(advisory_resolution_ratio)),
            "valid_fraction": float(meta["valid_fraction"]),
            "sigma_mask_fraction": float(meta["sigma_mask_fraction"]),
            "fit_mask_fraction": float(meta["fit_mask_fraction"]),
            "continuum_mask_fraction": float(meta["continuum_mask_fraction"]),
            "ca_hk_mask_fraction": float(meta.get("ca_hk_mask_fraction", 0.0)),
            "fe_mask_fraction": float(meta.get("fe_mask_fraction", 0.0)),
            "cat_mask_fraction": float(meta.get("cat_mask_fraction", 0.0)),
            "common_resolution_degraded": bool(meta.get("common_resolution_degraded", 0.0)),
            **recon_diag,
            **quality_report,
            "science_grade": science_grade,
            "confidence": confidence,
            "high_confidence": bool(high_confidence),
        }
        if sdss_name in literature:
            lit = literature[sdss_name]
            out["sigma_lit_kms"] = lit["sigma_lit_kms"]
            out["sigma_lit_err_kms"] = lit["sigma_lit_err_kms"]
            out["delta_lit_kms"] = pred_sigma - lit["sigma_lit_kms"]
            out["delta_lit_raw_kms"] = pred_sigma_raw - lit["sigma_lit_kms"]
            out["delta_lit_affine_kms"] = pred_sigma_affine - lit["sigma_lit_kms"]
        output_rows.append(out)
    return output_rows, {
        "device": str(device),
        "channels": list(channels),
        "calibration": calibration,
        "confidence_policy": (
            "Use high-confidence measurements for the strictest sample, include intermediate-confidence "
            "measurements when the larger uncertainty is acceptable, and inspect low-confidence spectra individually."
        ),
    }


def write_csv(rows: list[dict[str, float | str | bool]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | str | bool]], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray([float(row["pred_sigma_kms"]) for row in rows])
    err = np.asarray([float(row["pred_sigma_err_kms"]) for row in rows])
    host = np.asarray([float(row["pred_host_fraction"]) for row in rows])
    snr = np.asarray([float(row["median_snr"]) for row in rows])
    high = np.asarray([bool(row["high_confidence"]) for row in rows])
    redshift = np.asarray([float(row["redshift"]) for row in rows])
    has_lit = np.asarray(["sigma_lit_kms" in row and row.get("sigma_lit_kms") not in ("", None) for row in rows])

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
        axes[0, 0].errorbar(redshift, pred, yerr=err, fmt="o", ms=4, alpha=0.75, color="0.35", ecolor="0.65")
        axes[0, 0].scatter(redshift[high], pred[high], s=30, color="tab:blue", label="high confidence")
        axes[0, 0].set_xlabel("SDSS redshift")
        axes[0, 0].set_ylabel("Predicted sigma [km/s]")
        axes[0, 0].legend(frameon=False, fontsize=8)

        axes[0, 1].scatter(err, pred, c=host, s=28, alpha=0.85, cmap="viridis_r", edgecolors="none")
        axes[0, 1].set_xlabel("Calibrated sigma error [km/s]")
        axes[0, 1].set_ylabel("Predicted sigma [km/s]")

        axes[1, 0].hist(pred, bins=18, color="0.45", alpha=0.8)
        axes[1, 0].hist(pred[high], bins=18, color="tab:blue", alpha=0.65)
        axes[1, 0].set_xlabel("Predicted sigma [km/s]")
        axes[1, 0].set_ylabel("Number")

        axes[1, 1].scatter(snr, err, c=host, s=28, alpha=0.85, cmap="viridis_r", edgecolors="none")
        axes[1, 1].set_xlabel("Median rest-frame S/N proxy")
        axes[1, 1].set_ylabel("Calibrated sigma error [km/s]")
        fig.suptitle(f"SDSS inference summary: N={len(rows)}, high-confidence={int(high.sum())}", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        if np.any(has_lit):
            lit = np.asarray([float(row.get("sigma_lit_kms", np.nan)) for row in rows])
            delta = pred - lit
            finite = np.isfinite(lit) & np.isfinite(pred)
            lo = float(np.nanmin(np.concatenate([lit[finite], pred[finite]])))
            hi = float(np.nanmax(np.concatenate([lit[finite], pred[finite]])))
            pad = 0.06 * max(hi - lo, 1.0)
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
            axes[0].errorbar(lit[finite], pred[finite], yerr=err[finite], fmt="o", ms=4, color="0.45", ecolor="0.7", alpha=0.65)
            axes[0].scatter(lit[finite & high], pred[finite & high], s=34, color="tab:blue", label="high confidence")
            axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.2", lw=1)
            axes[0].set_xlabel("Greene & Ho literature sigma [km/s]")
            axes[0].set_ylabel("Predicted sigma [km/s]")
            axes[0].legend(frameon=False, fontsize=8)
            axes[1].axhline(0.0, color="0.2", lw=1)
            axes[1].scatter(lit[finite], delta[finite], c=err[finite], s=28, alpha=0.8, cmap="magma_r", edgecolors="none")
            axes[1].set_xlabel("Greene & Ho literature sigma [km/s]")
            axes[1].set_ylabel("Prediction - literature [km/s]")
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, meta = infer_rows(args)
    write_csv(rows, args.output_csv)
    calibration = meta.get("calibration", {})
    error_calibration = calibration.get("error_calibration", {})
    calibration_summary = {
        "loaded": bool(calibration),
        "sigma_affine_slope": calibration.get("sigma_affine_slope", calibration.get("calibration_slope")),
        "sigma_affine_intercept": calibration.get("sigma_affine_intercept", calibration.get("calibration_intercept")),
        "sigma_error_scale": calibration.get("sigma_error_scale", error_calibration.get("scale")),
        "sigma_error_floor_kms": calibration.get("sigma_error_floor_kms", error_calibration.get("floor_kms")),
        "quality_cuts": calibration.get("quality_cuts", {}),
    }
    print(
        json.dumps(
            {
                "n": len(rows),
                "device": meta.get("device"),
                "channels": meta.get("channels"),
                "calibration": calibration_summary,
            },
            indent=2,
        )
    )
    print(f"Saved {args.output_csv}")
    if args.output_pdf:
        plot_summary(rows, args.output_pdf)
        print(f"Saved {args.output_pdf}")


if __name__ == "__main__":
    main()
