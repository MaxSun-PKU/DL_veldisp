"""
Generate SDSS-like quasar host-galaxy mock spectra.

The generator is designed around two references already in this project:

- Greene & Ho 2006: use SDSS instrumental resolution, AGN dilution ranges,
  and stellar-absorption windows for Ca H+K, the 5250-5820 A Fe region, and CaT.
- Fe II contamination is drawn from a template bank. By default this includes
  Park et al. 2022, and it automatically picks up optional UV/optical templates
  placed in scripts/fe_template or reference/feii_templates, e.g.
  VW01/Tsuzuki/PyQSOFit files.
- AGN nuisance parameters can be sampled independently or with an approximate
  EV1/Baldwin covariance prior tying Fe II, [O III], Balmer-line width/EW, and
  a mock optical luminosity.
- If downloaded SDSS spectra are supplied, their ivar/coverage/bad-pixel
  structure is used as an empirical noise reference for the mocks. Their WDISP
  resolution values are also used to sample SDSS-like instrumental broadening.

The output HDF5 is backward compatible with data_loader_v2.MockAGNDataset, while
adding ivar/fit_mask/sigma_mask and nuisance labels needed for SDSS transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from feii_templates import DEFAULT_TEMPLATE_SPECS, FeIITemplateSpec, load_feii_template_bank, specs_from_files


C_KMS = 299792.458
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GREENE_SIGMA_WINDOWS = {
    "ca_hk": (3910.0, 4060.0),
    "fe_5250_5820": (5250.0, 5820.0),
    "cat": (8470.0, 8700.0),
}

GREENE_INTERNAL_MASKS = (
    (3955.0, 3985.0),  # Ca H contaminated by H epsilon.
    (5710.0, 5735.0),  # [Fe VII] 5721 inside the 5250-5820 Fe region.
)

PARK_CONTINUUM_WINDOWS = (
    (4170.0, 4260.0),
    (4430.0, 4720.0),
    (5080.0, 5500.0),
)

AGN_EMISSION_MASKS = (
    (3869.0, 18.0),   # [Ne III]
    (4102.9, 35.0),   # H delta
    (4341.6, 45.0),   # H gamma
    (4686.0, 35.0),   # He II
    (4862.7, 90.0),   # H beta
    (4959.0, 35.0),   # [O III]
    (5007.0, 40.0),   # [O III]
    (5721.0, 20.0),   # [Fe VII]
)

SDSS_SKY_LINES = (5577.0, 5890.0, 6300.0, 6363.0, 7246.0, 8430.0, 8919.0)
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
DEFAULT_SCIENCE_SIGMA_BIN_EDGES = (30.0, 60.0, 80.0, 120.0, 180.0, 250.0, 400.0, 700.0)
DEFAULT_SCIENCE_SIGMA_BIN_PROBS = (0.07, 0.09, 0.19, 0.21, 0.18, 0.15, 0.11)


@dataclass
class MockConfig:
    wave_min: float = 3800.0
    wave_max: float = 7500.0
    n_pixels: int = 3700
    n_z: int = 10
    n_age: int = 20
    logzsol_min: float = -2.0
    logzsol_max: float = 0.5
    age_min_gyr: float = 0.01
    age_max_gyr: float = 13.5
    sfh_old_bulge_prob: float = 0.62
    sfh_intermediate_bulge_prob: float = 0.28
    sfh_young_burst_prob: float = 0.25
    sfh_young_burst_min: float = 0.005
    sfh_young_burst_max: float = 0.08
    sfh_rare_young_burst_prob: float = 0.05
    sfh_rare_young_burst_max: float = 0.15
    sigma_min: float = 30.0
    sigma_max: float = 700.0
    sigma_sampling_mode: str = "science_bins"
    science_sigma_bin_edges: tuple[float, ...] = DEFAULT_SCIENCE_SIGMA_BIN_EDGES
    science_sigma_bin_probs: tuple[float, ...] = DEFAULT_SCIENCE_SIGMA_BIN_PROBS
    host_fraction_bin_probs: tuple[float, ...] = (0.20, 0.38, 0.42)
    agn_parameter_mode: str = "independent"
    agn_covariant_fraction: float = 0.45
    agn_log_l5100_min: float = 42.0
    agn_log_l5100_max: float = 46.8
    redshift_sampling_mode: str = "uniform"
    redshift_bin_edges: tuple[float, ...] = (0.01, 0.06, 0.15, 0.35, 0.90)
    redshift_bin_probs: tuple[float, ...] = (0.35, 0.25, 0.25, 0.15)
    sdss_sigma_inst: float = 70.0
    sdss_sigma_inst_min: float = 55.0
    sdss_sigma_inst_max: float = 95.0
    template_sigma_inst: float = 0.0
    common_resolution_sigma: float = 0.0
    snr_min: float = 5.0
    snr_max: float = 60.0
    sky_residual_prob: float = 0.55
    sky_residual_scale: float = 1.25
    telluric_residual_prob: float = 0.35
    redshift_min: float = 0.01
    redshift_max: float = 0.90
    obs_wave_min: float = 3800.0
    obs_wave_max: float = 9200.0
    park_template_path: str = "scripts/park_template.txt"
    feii_template_files: tuple[str, ...] = ()
    sdss_reference_paths: tuple[str, ...] = ()
    sdss_reference_min_valid_fraction: float = 0.20
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    hdf5_compression: str = "gzip"


def log_wave_grid(config: MockConfig) -> np.ndarray:
    return np.logspace(np.log10(config.wave_min), np.log10(config.wave_max), config.n_pixels)


def velocity_pixel_scale(wave: np.ndarray) -> float:
    dloglam = np.median(np.diff(np.log(wave)))
    return C_KMS * dloglam


def sps_logzsol_grid(config: MockConfig) -> np.ndarray:
    return np.linspace(config.logzsol_min, config.logzsol_max, config.n_z)


def sps_age_grid(config: MockConfig) -> np.ndarray:
    return np.logspace(np.log10(config.age_min_gyr), np.log10(config.age_max_gyr), config.n_age)


def median_in_window(wave: np.ndarray, flux: np.ndarray, center: float = 5100.0, half_width: float = 50.0) -> float:
    sel = (wave > center - half_width) & (wave < center + half_width) & np.isfinite(flux)
    if np.any(sel):
        value = np.nanmedian(flux[sel])
    else:
        value = np.nanmedian(flux[np.isfinite(flux)])
    return float(value) if np.isfinite(value) and value > 0 else 1.0


def normalize_at_5100(wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
    return flux / median_in_window(wave, flux)


def build_feii_specs(config: MockConfig) -> tuple[FeIITemplateSpec, ...]:
    specs = []
    for spec in DEFAULT_TEMPLATE_SPECS:
        if spec.name == "park2022_mrk493_optical":
            specs.append(
                FeIITemplateSpec(
                    name=spec.name,
                    path=config.park_template_path,
                    source=spec.source,
                    required=True,
                )
            )
        else:
            specs.append(spec)
    specs.extend(specs_from_files(config.feii_template_files))
    return tuple(specs)


def build_sdss_noise_library(config: MockConfig):
    if not config.sdss_reference_paths:
        return None
    from sdss_reference_spectra import SDSSNoiseLibrary

    return SDSSNoiseLibrary.from_paths(config.sdss_reference_paths)


def build_dense_sps_library(wave: np.ndarray, config: MockConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import fsps

    print(
        f"Building FSPS SPS basis ({config.n_z} Z x {config.n_age} age = "
        f"{config.n_z * config.n_age} templates)"
    )
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0)
    ages = sps_age_grid(config)
    zsols = sps_logzsol_grid(config)
    library = []
    start = time.time()

    for z in zsols:
        sp.params["logzsol"] = float(z)
        for age in ages:
            wave_fsps, flux_fsps = sp.get_spectrum(tage=float(age))
            interp = interp1d(wave_fsps, flux_fsps, bounds_error=False, fill_value=0.0)
            flux = normalize_at_5100(wave, interp(wave))
            library.append(flux.astype(np.float32))

    library = np.asarray(library, dtype=np.float32)
    print(f"Finished SPS basis in {time.time() - start:.1f}s, shape={library.shape}")
    return library, zsols.astype(np.float32), ages.astype(np.float32)


def generate_sfh_weights(config: MockConfig, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    z_grid, age_grid = np.meshgrid(np.arange(config.n_z), np.arange(config.n_age), indexing="ij")

    z_center = rng.uniform(0, config.n_z - 1)
    age_index_max = max(config.n_age - 1, 1)
    old_prob = float(np.clip(config.sfh_old_bulge_prob, 0.0, 1.0))
    intermediate_prob = float(np.clip(config.sfh_intermediate_bulge_prob, 0.0, 1.0 - old_prob))
    age_draw = rng.random()
    if age_draw < old_prob:
        age_center = rng.uniform(0.82 * age_index_max, age_index_max)
        sigma_age = rng.uniform(0.7, 2.4)
    elif age_draw < old_prob + intermediate_prob:
        age_center = rng.uniform(0.68 * age_index_max, 0.88 * age_index_max)
        sigma_age = rng.uniform(1.0, 3.2)
    else:
        age_center = rng.uniform(0.55 * age_index_max, age_index_max)
        sigma_age = rng.uniform(2.0, 5.0)
    sigma_z = rng.uniform(0.8, 3.2)
    weights = np.exp(
        -((z_grid - z_center) ** 2 / (2 * sigma_z**2) + (age_grid - age_center) ** 2 / (2 * sigma_age**2))
    )

    young_fraction = 0.0
    if rng.random() < float(np.clip(config.sfh_young_burst_prob, 0.0, 1.0)):
        burst_max = float(config.sfh_young_burst_max)
        if rng.random() < float(np.clip(config.sfh_rare_young_burst_prob, 0.0, 1.0)):
            burst_max = float(config.sfh_rare_young_burst_max)
        burst_min = min(float(config.sfh_young_burst_min), burst_max)
        young_fraction = float(rng.uniform(burst_min, burst_max))
        burst_age = rng.uniform(0, min(0.60 * age_index_max, age_index_max))
        burst_z = np.clip(z_center + rng.normal(0, 1.0), 0, config.n_z - 1)
        burst = np.exp(-((z_grid - burst_z) ** 2 / 2.0 + (age_grid - burst_age) ** 2 / 1.5))
        weights = (1.0 - young_fraction) * weights / weights.sum() + young_fraction * burst / burst.sum()

    weights = weights / weights.sum()
    return weights.reshape(-1).astype(np.float32), young_fraction


def gaussian_broaden_velocity(flux: np.ndarray, sigma_kms: float, dv_pixel: float) -> np.ndarray:
    sigma_pix = max(float(sigma_kms) / float(dv_pixel), 0.25)
    return gaussian_filter1d(flux, sigma_pix, mode="reflect")


def draw_instrument_sigma(config: MockConfig, rng: np.random.Generator, noise_library=None) -> float:
    if config.common_resolution_sigma > 0:
        return float(config.common_resolution_sigma)
    if noise_library is not None and hasattr(noise_library, "finite_median_inst_sigmas"):
        values = noise_library.finite_median_inst_sigmas
        if values.size:
            draw = float(values[int(rng.integers(0, values.size))])
            jittered = draw + float(rng.normal(0.0, 0.08 * draw))
            return float(np.clip(jittered, config.sdss_sigma_inst_min, config.sdss_sigma_inst_max))
    if config.sdss_sigma_inst_max > config.sdss_sigma_inst_min:
        return float(rng.uniform(config.sdss_sigma_inst_min, config.sdss_sigma_inst_max))
    return float(config.sdss_sigma_inst)


def required_velocity_convolution(
    sigma_star: float,
    sigma_inst: float,
    template_sigma_inst: float = 0.0,
) -> tuple[float, float]:
    observed_eff = float(np.sqrt(max(sigma_star, 0.0) ** 2 + max(sigma_inst, 0.0) ** 2))
    conv2 = observed_eff**2 - max(template_sigma_inst, 0.0) ** 2
    return float(np.sqrt(max(conv2, 0.25**2))), observed_eff


def gaussian_line(wave: np.ndarray, center: float, fwhm_kms: float, equivalent_width: float) -> np.ndarray:
    sigma_lambda = center * fwhm_kms / C_KMS / 2.355
    amp = equivalent_width / (np.sqrt(2.0 * np.pi) * sigma_lambda)
    return amp * np.exp(-0.5 * ((wave - center) / sigma_lambda) ** 2)


def draw_host_fraction(config: MockConfig, rng: np.random.Generator, log_l5100: float | None = None, ev1: float = 0.0) -> float:
    """Draw host-light fraction with an explicit science-oriented mixture.

    The low-host tail is still present so the network learns to reject hard
    cases, but v7 gives more weight to host-visible spectra because those are
    the only objects expected to yield publishable stellar velocity dispersions.
    """
    probs = np.asarray(config.host_fraction_bin_probs, dtype=float)
    if probs.size != 3 or not np.isfinite(probs).all() or probs.sum() <= 0:
        probs = np.asarray((0.20, 0.38, 0.42), dtype=float)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    draw = int(rng.choice(np.arange(3), p=probs))
    if draw == 0:
        host_fraction = float(rng.beta(1.5, 4.2) * 0.70 + 0.02)
    elif draw == 1:
        host_fraction = float(rng.uniform(0.25, 0.72))
    else:
        host_fraction = float(rng.uniform(0.55, 0.95))

    if log_l5100 is None or not np.isfinite(log_l5100):
        return host_fraction

    # More luminous AGNs generally dilute the host more strongly.  Keep broad
    # scatter so the network still sees host-visible quasars at high luminosity
    # and AGN-dominated low-luminosity cases.
    h = np.clip(host_fraction, 1e-3, 1.0 - 1e-3)
    logit = np.log(h / (1.0 - h))
    luminosity_shift = -0.55 * (float(log_l5100) - 44.3)
    ev1_shift = 0.08 * float(ev1)
    scatter = float(rng.normal(0.0, 0.28))
    correlated = 1.0 / (1.0 + np.exp(-(logit + luminosity_shift + ev1_shift + scatter)))
    return float(np.clip(correlated, 0.02, 0.96))


def draw_independent_agn_parameters(rng: np.random.Generator) -> dict[str, float]:
    """Draw the original broad, mostly independent AGN nuisance parameters."""
    return {
        "log_l5100": float(rng.uniform(42.0, 46.8)),
        "ev1": 0.0,
        "covariant_agn": 0.0,
        "power_law_alpha": float(rng.uniform(-2.6, 0.4)),
        "r_feii": float(np.clip(rng.gamma(shape=1.4, scale=0.55), 0.0, 3.0)),
        "feii_fwhm": float(10 ** rng.uniform(np.log10(800.0), np.log10(12000.0))),
        "balmer_fwhm": float(10 ** rng.uniform(np.log10(1000.0), np.log10(15000.0))),
        "hbeta_ew": float(rng.uniform(20.0, 140.0)),
        "oiii5007_ew": float(rng.uniform(3.0, 45.0)),
        "oiii4960_ew": float(rng.uniform(1.0, 15.0)),
        "narrow_hbeta_ew": float(rng.uniform(1.0, 12.0)),
        "heii4686_ew": float(rng.uniform(0.0, 15.0)),
        "fevii5721_ew": float(rng.uniform(0.0, 8.0)),
    }


def draw_ev1_baldwin_agn_parameters(config: MockConfig, rng: np.random.Generator) -> dict[str, float]:
    """Draw AGN nuisance parameters with approximate EV1 and Baldwin covariance.

    This is a broad empirical prior, not an object-by-object quasar-emission
    model.  The latent EV1 coordinate makes Fe II stronger, [O III] weaker, and
    broad H-beta somewhat narrower.  The mock optical luminosity imposes a
    Baldwin-like anticorrelation between line equivalent width and luminosity.
    """
    log_l5100 = float(rng.uniform(config.agn_log_l5100_min, config.agn_log_l5100_max))
    log_l_offset = log_l5100 - 45.0
    ev1 = float(np.clip(rng.normal(0.0, 1.0), -2.5, 2.5))

    alpha = float(np.clip(rng.normal(-1.35 + 0.10 * log_l_offset + 0.12 * ev1, 0.35), -2.8, 0.5))
    r_feii = float(np.clip(np.exp(rng.normal(np.log(0.55) + 0.42 * ev1, 0.42)), 0.0, 3.5))
    fe_fwhm = float(np.clip(10 ** rng.normal(3.35 + 0.08 * log_l_offset - 0.08 * ev1, 0.22), 700.0, 13000.0))
    balmer_fwhm = float(np.clip(10 ** rng.normal(3.55 + 0.10 * log_l_offset - 0.14 * ev1, 0.23), 900.0, 16000.0))

    hbeta_ew = float(np.clip(10 ** rng.normal(np.log10(75.0) - 0.10 * log_l_offset + 0.04 * ev1, 0.22), 12.0, 170.0))
    oiii5007_ew = float(np.clip(10 ** rng.normal(np.log10(18.0) - 0.10 * log_l_offset - 0.36 * ev1, 0.30), 0.5, 70.0))
    oiii4960_ew = float(oiii5007_ew / 2.98)
    narrow_hbeta_ew = float(np.clip(10 ** rng.normal(np.log10(0.20 * oiii5007_ew + 2.0), 0.20), 0.2, 18.0))
    heii4686_ew = float(np.clip(10 ** rng.normal(np.log10(4.0) - 0.10 * log_l_offset - 0.08 * ev1, 0.35), 0.0, 25.0))
    fevii5721_ew = float(np.clip(10 ** rng.normal(np.log10(2.0) - 0.08 * log_l_offset - 0.18 * ev1, 0.35), 0.0, 12.0))

    return {
        "log_l5100": log_l5100,
        "ev1": ev1,
        "covariant_agn": 1.0,
        "power_law_alpha": alpha,
        "r_feii": r_feii,
        "feii_fwhm": fe_fwhm,
        "balmer_fwhm": balmer_fwhm,
        "hbeta_ew": hbeta_ew,
        "oiii5007_ew": oiii5007_ew,
        "oiii4960_ew": oiii4960_ew,
        "narrow_hbeta_ew": narrow_hbeta_ew,
        "heii4686_ew": heii4686_ew,
        "fevii5721_ew": fevii5721_ew,
    }


def draw_agn_parameters(config: MockConfig, rng: np.random.Generator) -> dict[str, float]:
    mode = str(config.agn_parameter_mode).lower()
    if mode == "independent":
        return draw_independent_agn_parameters(rng)
    if mode == "ev1_baldwin":
        return draw_ev1_baldwin_agn_parameters(config, rng)
    if mode in ("mixed_ev1_baldwin", "mixed", "hybrid"):
        if rng.random() < float(np.clip(config.agn_covariant_fraction, 0.0, 1.0)):
            return draw_ev1_baldwin_agn_parameters(config, rng)
        return draw_independent_agn_parameters(rng)
    raise ValueError(f"Unknown agn_parameter_mode={config.agn_parameter_mode!r}")


def add_quasar_components(
    wave: np.ndarray,
    stellar_flux: np.ndarray,
    fe_template_bank: np.ndarray,
    dv_pixel: float,
    rng: np.random.Generator,
    config: MockConfig,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    agn_params = draw_agn_parameters(config, rng)
    covariant_agn = bool(float(agn_params.get("covariant_agn", 0.0)) > 0.5)
    alpha = agn_params["power_law_alpha"]
    host_fraction = draw_host_fraction(
        config,
        rng,
        log_l5100=agn_params["log_l5100"] if covariant_agn else None,
        ev1=agn_params["ev1"],
    )
    r_feii = agn_params["r_feii"]
    fe_fwhm = agn_params["feii_fwhm"]
    balmer_fwhm = agn_params["balmer_fwhm"]
    hbeta_ew = agn_params["hbeta_ew"]
    paschen_fwhm = float(10 ** rng.uniform(np.log10(700.0), np.log10(9000.0)))
    if covariant_agn:
        paschen_ew = float(np.clip(hbeta_ew * rng.uniform(0.02, 0.18), 0.0, 24.0) if rng.random() < 0.45 else 0.0)
        oi8446_ew = float(np.clip(r_feii * rng.uniform(1.0, 5.5), 0.0, 16.0) if rng.random() < 0.30 else 0.0)
        feii8616_ew = float(np.clip(r_feii * rng.uniform(0.8, 4.0), 0.0, 11.0) if rng.random() < 0.25 else 0.0)
    else:
        paschen_ew = float(rng.uniform(0.0, 22.0) if rng.random() < 0.45 else 0.0)
        oi8446_ew = float(rng.uniform(0.0, 14.0) if rng.random() < 0.30 else 0.0)
        feii8616_ew = float(rng.uniform(0.0, 9.0) if rng.random() < 0.25 else 0.0)
    cat_emission_ew = float(rng.uniform(0.0, 5.0) if rng.random() < 0.18 else 0.0)

    power_law = (wave / 5100.0) ** alpha
    fe_sigma = fe_fwhm / 2.355

    n_fe_templates = fe_template_bank.shape[0]
    fe_weights = np.zeros(n_fe_templates, dtype=np.float32)
    primary_idx = int(rng.integers(0, n_fe_templates))
    fe_weights[primary_idx] = 1.0
    if n_fe_templates > 1 and rng.random() < 0.40:
        secondary_idx = int(rng.integers(0, n_fe_templates - 1))
        if secondary_idx >= primary_idx:
            secondary_idx += 1
        secondary_weight = float(rng.uniform(0.10, 0.70))
        fe_weights[primary_idx] = 1.0 - secondary_weight
        fe_weights[secondary_idx] = secondary_weight

    feii = np.zeros_like(wave, dtype=np.float32)
    for weight, template in zip(fe_weights, fe_template_bank):
        if weight > 0:
            feii += weight * gaussian_broaden_velocity(template, fe_sigma, dv_pixel).astype(np.float32)

    balmer = (
        gaussian_line(wave, 4862.66, balmer_fwhm, hbeta_ew)
        + gaussian_line(wave, 4341.62, balmer_fwhm, 0.36 * hbeta_ew)
        + gaussian_line(wave, 4102.89, balmer_fwhm, 0.20 * hbeta_ew)
    )
    narrow = (
        gaussian_line(wave, 5008.24, 350.0, agn_params["oiii5007_ew"])
        + gaussian_line(wave, 4960.30, 350.0, agn_params["oiii4960_ew"])
        + gaussian_line(wave, 4862.66, 350.0, agn_params["narrow_hbeta_ew"])
        + gaussian_line(wave, 4686.0, 900.0, agn_params["heii4686_ew"])
        + gaussian_line(wave, 5721.0, 500.0, agn_params["fevii5721_ew"])
    )
    paschen_cat = paschen_ew * (
        gaussian_line(wave, 8502.48, paschen_fwhm, 0.32)
        + gaussian_line(wave, 8545.38, paschen_fwhm, 0.48)
        + gaussian_line(wave, 8598.39, paschen_fwhm, 0.25)
        + gaussian_line(wave, 8665.02, paschen_fwhm, 0.65)
    )
    cat_nuisance = (
        gaussian_line(wave, 8446.36, 700.0, oi8446_ew)
        + gaussian_line(wave, 8616.95, 500.0, feii8616_ew)
        + cat_emission_ew
        * (
            gaussian_line(wave, 8498.02, 500.0, 0.35)
            + gaussian_line(wave, 8542.09, 500.0, 0.55)
            + gaussian_line(wave, 8662.14, 500.0, 0.45)
        )
    )

    agn_shape = power_law + 0.18 * r_feii * feii + balmer + narrow + paschen_cat + cat_nuisance
    agn_shape = normalize_at_5100(wave, agn_shape)
    stellar_flux = normalize_at_5100(wave, stellar_flux)
    mixed = host_fraction * stellar_flux + (1.0 - host_fraction) * agn_shape

    labels = {
        "host_fraction": host_fraction,
        "agn_fraction": 1.0 - host_fraction,
        "agn_log_l5100": agn_params["log_l5100"],
        "agn_ev1": agn_params["ev1"],
        "agn_covariant_draw": float(agn_params.get("covariant_agn", 0.0)),
        "power_law_alpha": alpha,
        "r_feii": r_feii,
        "feii_fwhm": fe_fwhm,
        "feii_template_index": float(primary_idx),
        "feii_template_mix_entropy": float(-np.sum(fe_weights[fe_weights > 0] * np.log(fe_weights[fe_weights > 0]))),
        "balmer_fwhm": balmer_fwhm,
        "hbeta_ew": hbeta_ew,
        "oiii5007_ew": agn_params["oiii5007_ew"],
        "oiii4960_ew": agn_params["oiii4960_ew"],
        "narrow_hbeta_ew": agn_params["narrow_hbeta_ew"],
        "heii4686_ew": agn_params["heii4686_ew"],
        "fevii5721_ew": agn_params["fevii5721_ew"],
        "paschen_fwhm": paschen_fwhm,
        "paschen_cat_ew": paschen_ew,
        "oi8446_ew": oi8446_ew,
        "feii8616_ew": feii8616_ew,
        "cat_emission_ew": cat_emission_ew,
        "cat_contamination_ew": paschen_ew + oi8446_ew + feii8616_ew + cat_emission_ew,
    }
    return mixed.astype(np.float32), labels, agn_shape.astype(np.float32)


def observed_coverage_mask(wave_rest: np.ndarray, redshift: float, config: MockConfig) -> np.ndarray:
    observed_wave = wave_rest * (1.0 + redshift)
    return (observed_wave >= config.obs_wave_min) & (observed_wave <= config.obs_wave_max)


def inject_observed_frame_residuals(
    observed_wave: np.ndarray,
    noisy_flux: np.ndarray,
    valid: np.ndarray,
    local_scale: float,
    snr_5100: float,
    config: MockConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    noisy = noisy_flux.astype(np.float32).copy()
    valid_bool = valid.astype(bool)
    noise_unit = local_scale / max(float(snr_5100), 1.0)
    sky_strength = 0.0
    telluric_strength = 0.0

    if rng.random() < config.sky_residual_prob:
        for sky in SDSS_SKY_LINES:
            amp = rng.normal(0.0, config.sky_residual_scale * noise_unit)
            width = rng.uniform(1.5, 5.0)
            profile = np.exp(-0.5 * ((observed_wave - sky) / width) ** 2).astype(np.float32)
            noisy[valid_bool] += (amp * profile[valid_bool]).astype(np.float32)
            sky_strength += float(abs(amp) / max(local_scale, 1e-6))

    if rng.random() < config.telluric_residual_prob:
        for center, width in ((6867.0, 18.0), (7605.0, 26.0), (8227.0, 18.0)):
            amp = rng.normal(0.0, 0.55 * config.sky_residual_scale * noise_unit)
            profile = np.exp(-0.5 * ((observed_wave - center) / width) ** 2).astype(np.float32)
            noisy[valid_bool] += (amp * profile[valid_bool]).astype(np.float32)
            telluric_strength += float(abs(amp) / max(local_scale, 1e-6))

    return noisy.astype(np.float32), sky_strength, telluric_strength


def build_fit_masks(
    wave_rest: np.ndarray,
    redshift: float,
    config: MockConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    valid = observed_coverage_mask(wave_rest, redshift, config)

    fit_mask = valid.copy()
    for center, half_width in AGN_EMISSION_MASKS:
        fit_mask &= np.abs(wave_rest - center) > half_width

    continuum_mask = np.zeros_like(valid, dtype=bool)
    for lo, hi in PARK_CONTINUUM_WINDOWS:
        continuum_mask |= (wave_rest >= lo) & (wave_rest <= hi)
    continuum_mask &= fit_mask

    sigma_mask = np.zeros_like(valid, dtype=bool)
    for lo, hi in GREENE_SIGMA_WINDOWS.values():
        sigma_mask |= (wave_rest >= lo) & (wave_rest <= hi)
    sigma_mask &= valid
    for lo, hi in GREENE_INTERNAL_MASKS:
        sigma_mask &= ~((wave_rest >= lo) & (wave_rest <= hi))

    if redshift < 0.05 and wave_rest.max() >= GREENE_SIGMA_WINDOWS["cat"][1]:
        region_code = 0
    elif redshift < 0.76:
        region_code = 1
    else:
        region_code = 2

    return (
        fit_mask.astype(np.float32),
        sigma_mask.astype(np.float32),
        continuum_mask.astype(np.float32),
        valid.astype(np.float32),
        region_code,
    )


def add_sdss_noise(
    wave_rest: np.ndarray,
    flux: np.ndarray,
    redshift: float,
    valid_pixels: np.ndarray,
    config: MockConfig,
    rng: np.random.Generator,
    noise_library=None,
    log_l5100: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float, np.ndarray, float, float]:
    if log_l5100 is not None and np.isfinite(log_l5100):
        percentile = 0.50 + 0.16 * (float(log_l5100) - 44.3) + float(rng.normal(0.0, 0.25))
        percentile = float(np.clip(percentile, 0.0, 1.0))
        snr_5100 = float(config.snr_min + percentile * (config.snr_max - config.snr_min))
    else:
        snr_5100 = float(rng.uniform(config.snr_min, config.snr_max))
    observed_wave = wave_rest * (1.0 + redshift)
    local_scale = median_in_window(wave_rest, flux)

    if noise_library is not None:
        sampled = noise_library.sample_on_observed_grid(
            observed_wave,
            rng,
            min_valid_fraction=config.sdss_reference_min_valid_fraction,
        )
        if sampled is not None:
            noise_shape, reference_valid, reference_inst_sigma, reference_index = sampled
            valid = valid_pixels.astype(bool) & reference_valid
            if np.count_nonzero(valid) > 10:
                noise_shape = noise_shape.astype(np.float32)
                scale = np.nanmedian(noise_shape[valid])
                if np.isfinite(scale) and scale > 0:
                    noise_shape = np.clip(noise_shape / scale, 0.05, 40.0)
                    sigma_noise = np.zeros_like(flux, dtype=np.float32)
                    sigma_noise[valid] = local_scale * noise_shape[valid] / snr_5100

                    ivar = np.zeros_like(flux, dtype=np.float32)
                    ivar[valid] = (1.0 / sigma_noise[valid] ** 2).astype(np.float32)
                    noisy = flux.astype(np.float32).copy()
                    noisy[valid] += rng.normal(0.0, sigma_noise[valid], size=np.count_nonzero(valid)).astype(np.float32)
                    noisy, sky_strength, telluric_strength = inject_observed_frame_residuals(
                        observed_wave,
                        noisy,
                        valid,
                        local_scale,
                        snr_5100,
                        config,
                        rng,
                    )
                    noisy[~valid] = 0.0
                    finite_inst = valid & np.isfinite(reference_inst_sigma) & (reference_inst_sigma > 0)
                    sampled_inst = float(np.nanmedian(reference_inst_sigma[finite_inst])) if np.any(finite_inst) else np.nan
                    return (
                        noisy.astype(np.float32),
                        ivar,
                        valid.astype(np.float32),
                        snr_5100,
                        int(reference_index),
                        sampled_inst,
                        reference_inst_sigma.astype(np.float32),
                        sky_strength,
                        telluric_strength,
                    )

    noise_shape = 1.0 + 0.35 * ((observed_wave - 5500.0) / 2800.0) ** 2
    for sky in SDSS_SKY_LINES:
        noise_shape += rng.uniform(0.05, 0.25) * np.exp(-0.5 * ((observed_wave - sky) / 8.0) ** 2)

    sigma_noise = local_scale * noise_shape / snr_5100
    ivar = np.zeros_like(flux, dtype=np.float32)
    ivar[valid_pixels > 0] = (1.0 / sigma_noise[valid_pixels > 0] ** 2).astype(np.float32)

    noisy = flux + rng.normal(0.0, sigma_noise, size=flux.size).astype(np.float32)
    bad_mask = valid_pixels.astype(bool).copy()
    n_bad = rng.poisson(2.0)
    for _ in range(n_bad):
        start = int(rng.integers(0, flux.size))
        width = int(rng.integers(1, 5))
        bad_mask[start : start + width] = False

    noisy[~valid_pixels.astype(bool)] = 0.0
    noisy, sky_strength, telluric_strength = inject_observed_frame_residuals(
        observed_wave,
        noisy,
        bad_mask,
        local_scale,
        snr_5100,
        config,
        rng,
    )
    noisy[~valid_pixels.astype(bool)] = 0.0
    ivar[~bad_mask] = 0.0
    valid = valid_pixels.astype(np.float32)
    valid[~bad_mask] = 0.0
    return noisy.astype(np.float32), ivar, valid, snr_5100, -1, np.nan, np.full_like(flux, np.nan, dtype=np.float32), sky_strength, telluric_strength


def summarize_mock_quality(
    wave_rest: np.ndarray,
    ivar: np.ndarray,
    fit_mask: np.ndarray,
    sigma_mask: np.ndarray,
    continuum_mask: np.ndarray,
    valid_pixels: np.ndarray,
    host_fraction: float,
    sigma_star: float,
    sigma_inst_kms: float,
    cat_contamination_ew: float = 0.0,
    target_sigma_pixels: float = 350.0,
    host_floor: float = 0.10,
) -> dict[str, float]:
    valid = valid_pixels > 0
    sigma_valid = sigma_mask > 0
    fit_valid = fit_mask > 0
    continuum_valid = continuum_mask > 0
    positive_sigma_ivar = ivar[sigma_valid & np.isfinite(ivar) & (ivar > 0)]
    mean_sigma_ivar = float(np.mean(positive_sigma_ivar)) if positive_sigma_ivar.size else 0.0
    n_sigma_pixels = float(np.count_nonzero(sigma_valid))
    coverage_quality = np.clip(n_sigma_pixels / max(target_sigma_pixels, 1.0), 0.0, 1.0)
    ivar_quality = np.sqrt(mean_sigma_ivar / (mean_sigma_ivar + 1.0)) if mean_sigma_ivar > 0 else 0.0
    host_quality = np.clip((host_fraction - host_floor) / max(0.40 - host_floor, 1e-3), 0.0, 1.0)
    sigma_resolution_ratio = float(sigma_star / max(sigma_inst_kms, 1e-6))
    resolution_core = np.clip((sigma_resolution_ratio - 1.0) / 1.0, 0.0, 1.0)
    resolution_quality = 0.30 + 0.70 * resolution_core
    cat_contamination_quality = 1.0 / (1.0 + max(float(cat_contamination_ew), 0.0) / 15.0)
    quality_proxy = (
        0.25
        + 0.75
        * coverage_quality
        * ivar_quality
        * (0.35 + 0.65 * host_quality)
        * resolution_quality
        * (0.50 + 0.50 * cat_contamination_quality)
    )
    region_fractions = {}
    for name, (lo, hi) in GREENE_SIGMA_WINDOWS.items():
        key = "fe_mask_fraction" if name == "fe_5250_5820" else f"{name}_mask_fraction"
        region = (wave_rest >= lo) & (wave_rest <= hi) & sigma_valid
        region_fractions[key] = float(np.mean(region))
    return {
        "valid_fraction": float(np.mean(valid)),
        "fit_mask_fraction": float(np.mean(fit_valid)),
        "sigma_mask_fraction": float(np.mean(sigma_valid)),
        "continuum_mask_fraction": float(np.mean(continuum_valid)),
        "mean_sigma_ivar": mean_sigma_ivar,
        "sigma_resolution_ratio": sigma_resolution_ratio,
        "resolution_quality": float(resolution_quality),
        "cat_contamination_quality": float(cat_contamination_quality),
        "mock_quality_proxy": float(np.clip(quality_proxy, 0.25, 1.0)),
        **region_fractions,
    }



def parse_float_tuple(text: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one number")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values: {values}")
    return values


def draw_sigma_star(config: MockConfig, rng: np.random.Generator) -> tuple[float, int]:
    mode = str(config.sigma_sampling_mode).lower()
    if mode in ("uniform", "flat"):
        return float(rng.uniform(config.sigma_min, config.sigma_max)), -1
    if mode in ("loguniform", "log"):
        lo = max(float(config.sigma_min), 1e-3)
        hi = max(float(config.sigma_max), lo * 1.001)
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), -1
    if mode not in ("science_bins", "science", "bins"):
        raise ValueError(f"Unknown sigma_sampling_mode={config.sigma_sampling_mode!r}")

    edges = np.asarray(config.science_sigma_bin_edges, dtype=float)
    probs = np.asarray(config.science_sigma_bin_probs, dtype=float)
    edges = np.clip(edges, float(config.sigma_min), float(config.sigma_max))
    keep = np.r_[True, np.diff(edges) > 1e-6]
    edges = edges[keep]
    if edges.size < 2:
        return float(rng.uniform(config.sigma_min, config.sigma_max)), -1
    n_bins = edges.size - 1
    if probs.size != n_bins:
        probs = np.ones(n_bins, dtype=float)
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        probs = np.ones(n_bins, dtype=float)
    probs = probs / probs.sum()
    bin_index = int(rng.choice(np.arange(n_bins), p=probs))
    return float(rng.uniform(edges[bin_index], edges[bin_index + 1])), bin_index


def draw_redshift(config: MockConfig, rng: np.random.Generator) -> tuple[float, int]:
    mode = str(config.redshift_sampling_mode).lower()
    if mode in ("uniform", "flat"):
        return float(rng.uniform(config.redshift_min, config.redshift_max)), -1
    if mode not in ("science_bins", "science", "bins"):
        raise ValueError(f"Unknown redshift_sampling_mode={config.redshift_sampling_mode!r}")

    edges = np.asarray(config.redshift_bin_edges, dtype=float)
    probs = np.asarray(config.redshift_bin_probs, dtype=float)
    edges = np.clip(edges, float(config.redshift_min), float(config.redshift_max))
    keep = np.r_[True, np.diff(edges) > 1e-6]
    edges = edges[keep]
    if edges.size < 2:
        return float(rng.uniform(config.redshift_min, config.redshift_max)), -1
    n_bins = edges.size - 1
    if probs.size != n_bins:
        probs = np.ones(n_bins, dtype=float)
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        probs = np.ones(n_bins, dtype=float)
    probs = probs / probs.sum()
    bin_index = int(rng.choice(np.arange(n_bins), p=probs))
    return float(rng.uniform(edges[bin_index], edges[bin_index + 1])), bin_index


def generate_single_mock(
    wave: np.ndarray,
    template_library: np.ndarray,
    fe_template_bank: np.ndarray,
    config: MockConfig,
    rng: np.random.Generator,
    noise_library=None,
) -> dict[str, np.ndarray | float | int]:
    dv_pixel = velocity_pixel_scale(wave)
    sfh_weights, young_fraction = generate_sfh_weights(config, rng)
    sfh_grid = sfh_weights.reshape(config.n_z, config.n_age)
    mass_weighted_logzsol = float(np.sum(sfh_grid * sps_logzsol_grid(config)[:, None]))
    mass_weighted_age_gyr = float(np.sum(sfh_grid * sps_age_grid(config)[None, :]))
    stellar_base = np.dot(sfh_weights, template_library)

    sigma_star, sigma_sampling_bin = draw_sigma_star(config, rng)
    sigma_inst_kms = draw_instrument_sigma(config, rng, noise_library=noise_library)
    sigma_conv_kms, sigma_observed_eff_kms = required_velocity_convolution(
        sigma_star,
        sigma_inst_kms,
        template_sigma_inst=config.template_sigma_inst,
    )
    stellar_flux = gaussian_broaden_velocity(stellar_base, sigma_conv_kms, dv_pixel)

    clean_flux, labels, agn_shape = add_quasar_components(wave, stellar_flux, fe_template_bank, dv_pixel, rng, config)
    redshift, redshift_sampling_bin = draw_redshift(config, rng)
    fit_mask, sigma_mask, continuum_mask, valid_pixels, region_code = build_fit_masks(wave, redshift, config)
    (
        noisy_flux,
        ivar,
        valid_pixels,
        snr,
        sdss_noise_reference_index,
        sampled_sdss_inst_sigma_kms,
        sampled_inst_sigma_grid,
        sky_residual_strength,
        telluric_residual_strength,
    ) = add_sdss_noise(
        wave,
        clean_flux,
        redshift,
        valid_pixels,
        config,
        rng,
        noise_library=noise_library,
        log_l5100=labels.get("agn_log_l5100") if float(labels.get("agn_covariant_draw", 0.0)) > 0.5 else None,
    )
    if config.common_resolution_sigma > 0:
        inst_sigma_grid = np.full_like(wave, float(config.common_resolution_sigma), dtype=np.float32)
    elif np.any(np.isfinite(sampled_inst_sigma_grid) & (sampled_inst_sigma_grid > 0)):
        inst_sigma_grid = np.where(
            np.isfinite(sampled_inst_sigma_grid) & (sampled_inst_sigma_grid > 0),
            sampled_inst_sigma_grid,
            sigma_inst_kms,
        ).astype(np.float32)
    else:
        inst_sigma_grid = np.full_like(wave, sigma_inst_kms, dtype=np.float32)

    fit_mask = fit_mask * valid_pixels
    sigma_mask = sigma_mask * valid_pixels
    continuum_mask = continuum_mask * valid_pixels
    scaled_agn_flux = labels["agn_fraction"] * agn_shape
    host_flux = clean_flux - scaled_agn_flux
    quality_labels = summarize_mock_quality(
        wave,
        ivar,
        fit_mask,
        sigma_mask,
        continuum_mask,
        valid_pixels,
        labels["host_fraction"],
        sigma_star,
        sigma_inst_kms,
        labels["cat_contamination_ew"],
    )
    return {
        "spectra": noisy_flux,
        "clean_spectra": clean_flux,
        "agn_spectra": agn_shape,
        "scaled_agn_spectra": scaled_agn_flux.astype(np.float32),
        "host_spectra": host_flux.astype(np.float32),
        "ivar": ivar,
        "fit_mask": fit_mask.astype(np.float32),
        "sigma_mask": sigma_mask.astype(np.float32),
        "continuum_mask": continuum_mask.astype(np.float32),
        "valid_pixels": valid_pixels.astype(np.float32),
        "inst_sigma_kms_grid": inst_sigma_grid.astype(np.float32),
        "sfh_weights": sfh_weights,
        "sigma_sampling_bin": float(sigma_sampling_bin),
        "sigma_star": sigma_star,
        "sigma_inst_kms": sigma_inst_kms,
        "template_sigma_inst_kms": float(config.template_sigma_inst),
        "sigma_convolution_kms": sigma_conv_kms,
        "sigma_observed_eff_kms": sigma_observed_eff_kms,
        "sampled_sdss_inst_sigma_kms": sampled_sdss_inst_sigma_kms,
        "sky_residual_strength": sky_residual_strength,
        "telluric_residual_strength": telluric_residual_strength,
        "host_fraction": labels["host_fraction"],
        "agn_fraction": labels["agn_fraction"],
        "snr": snr,
        "redshift": redshift,
        "redshift_sampling_bin": float(redshift_sampling_bin),
        "region_code": region_code,
        "sdss_noise_reference_index": sdss_noise_reference_index,
        "young_fraction": young_fraction,
        "mass_weighted_age_gyr": mass_weighted_age_gyr,
        "mass_weighted_logzsol": mass_weighted_logzsol,
        "agn_log_l5100": labels["agn_log_l5100"],
        "agn_ev1": labels["agn_ev1"],
        "agn_covariant_draw": labels["agn_covariant_draw"],
        "power_law_alpha": labels["power_law_alpha"],
        "r_feii": labels["r_feii"],
        "feii_fwhm": labels["feii_fwhm"],
        "feii_template_index": labels["feii_template_index"],
        "feii_template_mix_entropy": labels["feii_template_mix_entropy"],
        "balmer_fwhm": labels["balmer_fwhm"],
        "hbeta_ew": labels["hbeta_ew"],
        "oiii5007_ew": labels["oiii5007_ew"],
        "oiii4960_ew": labels["oiii4960_ew"],
        "narrow_hbeta_ew": labels["narrow_hbeta_ew"],
        "heii4686_ew": labels["heii4686_ew"],
        "fevii5721_ew": labels["fevii5721_ew"],
        "paschen_fwhm": labels["paschen_fwhm"],
        "paschen_cat_ew": labels["paschen_cat_ew"],
        "oi8446_ew": labels["oi8446_ew"],
        "feii8616_ew": labels["feii8616_ew"],
        "cat_emission_ew": labels["cat_emission_ew"],
        "cat_contamination_ew": labels["cat_contamination_ew"],
        **quality_labels,
    }


def build_split_indices(
    n_samples: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: Union[int, None],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fractions = np.asarray([train_frac, val_frac, test_frac], dtype=float)
    if np.any(fractions < 0) or not np.isfinite(fractions).all() or fractions.sum() <= 0:
        raise ValueError("Split fractions must be finite non-negative numbers with a positive sum.")
    fractions = fractions / fractions.sum()

    n_train = int(round(n_samples * fractions[0]))
    n_val = int(round(n_samples * fractions[1]))
    n_train = min(max(n_train, 0), n_samples)
    n_val = min(max(n_val, 0), n_samples - n_train)
    n_test = n_samples - n_train - n_val

    split_seed = None if seed is None else int(seed) + 104729
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_samples)

    split = np.full(n_samples, SPLIT_CODES["test"], dtype=np.uint8)
    train_indices = np.sort(perm[:n_train]).astype(np.int64)
    val_indices = np.sort(perm[n_train : n_train + n_val]).astype(np.int64)
    test_indices = np.sort(perm[n_train + n_val : n_train + n_val + n_test]).astype(np.int64)
    split[train_indices] = SPLIT_CODES["train"]
    split[val_indices] = SPLIT_CODES["val"]
    split[test_indices] = SPLIT_CODES["test"]
    return split, {"train": train_indices, "val": val_indices, "test": test_indices}


def create_dataset(output_path: Union[str, Path], n_samples: int, config: MockConfig, seed: Union[int, None] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    wave = log_wave_grid(config).astype(np.float32)
    fe_template_bank, feii_manifest = load_feii_template_bank(
        wave,
        specs=build_feii_specs(config),
        root=PROJECT_ROOT,
    )
    noise_library = build_sdss_noise_library(config)
    template_library, zsols, ages = build_dense_sps_library(wave, config)
    split, split_indices = build_split_indices(
        n_samples,
        config.train_frac,
        config.val_frac,
        config.test_frac,
        seed,
    )

    vector_keys = (
        "spectra",
        "clean_spectra",
        "agn_spectra",
        "scaled_agn_spectra",
        "host_spectra",
        "ivar",
        "fit_mask",
        "sigma_mask",
        "continuum_mask",
        "valid_pixels",
        "inst_sigma_kms_grid",
    )
    scalar_keys = (
        "sigma_star",
        "sigma_sampling_bin",
        "host_fraction",
        "agn_fraction",
        "snr",
        "redshift",
        "redshift_sampling_bin",
        "young_fraction",
        "mass_weighted_age_gyr",
        "mass_weighted_logzsol",
        "agn_log_l5100",
        "agn_ev1",
        "agn_covariant_draw",
        "power_law_alpha",
        "r_feii",
        "feii_fwhm",
        "feii_template_index",
        "feii_template_mix_entropy",
        "balmer_fwhm",
        "hbeta_ew",
        "oiii5007_ew",
        "oiii4960_ew",
        "narrow_hbeta_ew",
        "heii4686_ew",
        "fevii5721_ew",
        "valid_fraction",
        "fit_mask_fraction",
        "sigma_mask_fraction",
        "continuum_mask_fraction",
        "mean_sigma_ivar",
        "sigma_inst_kms",
        "template_sigma_inst_kms",
        "sigma_convolution_kms",
        "sigma_observed_eff_kms",
        "sampled_sdss_inst_sigma_kms",
        "sky_residual_strength",
        "telluric_residual_strength",
        "sigma_resolution_ratio",
        "resolution_quality",
        "cat_contamination_quality",
        "ca_hk_mask_fraction",
        "fe_mask_fraction",
        "cat_mask_fraction",
        "paschen_fwhm",
        "paschen_cat_ew",
        "oi8446_ew",
        "feii8616_ew",
        "cat_emission_ew",
        "cat_contamination_ew",
        "mock_quality_proxy",
    )

    with h5py.File(output_path, "w") as hf:
        hf.attrs["description"] = "SDSS-like quasar host mock spectra for velocity-dispersion learning"
        hf.attrs["generation_complete"] = False
        hf.attrs["config_json"] = json.dumps(asdict(config))
        hf.attrs["references"] = (
            "Greene_2006_ApJ_641_117; Park_2022_ApJS_258_38; "
            "Vestergaard_Wilkes_2001; Tsuzuki_2006 when corresponding local templates are present"
        )
        hf.attrs["feii_template_manifest_json"] = json.dumps(feii_manifest)
        hf.attrs["sdss_noise_reference_manifest_json"] = json.dumps(
            noise_library.manifest if noise_library is not None else []
        )
        hf.attrs["split_code_json"] = json.dumps(SPLIT_CODES)
        hf.attrs["split_counts_json"] = json.dumps({name: int(len(indices)) for name, indices in split_indices.items()})

        hf.create_dataset("wave_grid", data=wave)
        hf.create_dataset("template_library", data=template_library, compression=config.hdf5_compression)
        hf.create_dataset("sps_logzsol", data=zsols)
        hf.create_dataset("sps_age_gyr", data=ages)
        hf.create_dataset("feii_template_bank", data=fe_template_bank, compression=config.hdf5_compression)
        hf.create_dataset("park_fe_template", data=fe_template_bank[0])
        hf.create_dataset("split", data=split)
        for split_name, indices in split_indices.items():
            hf.create_dataset(f"{split_name}_indices", data=indices)

        datasets = {
            key: hf.create_dataset(
                key,
                shape=(n_samples, config.n_pixels),
                dtype=np.float32,
                chunks=(min(256, n_samples), config.n_pixels),
                compression=config.hdf5_compression,
                shuffle=True,
            )
            for key in vector_keys
        }
        datasets["sfh_weights"] = hf.create_dataset(
            "sfh_weights",
            shape=(n_samples, config.n_z * config.n_age),
            dtype=np.float32,
            chunks=(min(256, n_samples), config.n_z * config.n_age),
            compression=config.hdf5_compression,
            shuffle=True,
        )
        for key in scalar_keys:
            datasets[key] = hf.create_dataset(key, shape=(n_samples,), dtype=np.float32)
        datasets["region_code"] = hf.create_dataset("region_code", shape=(n_samples,), dtype=np.int16)
        datasets["sdss_noise_reference_index"] = hf.create_dataset(
            "sdss_noise_reference_index",
            shape=(n_samples,),
            dtype=np.int32,
        )

        for i in tqdm(range(n_samples), desc="Generating SDSS-like mocks"):
            record = generate_single_mock(wave, template_library, fe_template_bank, config, rng, noise_library=noise_library)
            for key, dataset in datasets.items():
                dataset[i] = record[key]
        hf.attrs["generation_complete"] = True
        hf.flush()

    print(f"Saved {n_samples} mocks to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./mock_data/mock_sdss_quasar_dataset.h5")
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsps-home", default=None, help="Optional SPS_HOME path, e.g. /path/to/fsps")
    parser.add_argument("--wave-min", type=float, default=3800.0, help="Use 1200 to include UV Fe II/Mg II rest-frame mocks.")
    parser.add_argument("--wave-max", type=float, default=7500.0, help="Use 9000+ to include CaT rest-frame mocks.")
    parser.add_argument("--n-pixels", type=int, default=3700)
    parser.add_argument("--n-z", type=int, default=10)
    parser.add_argument("--n-age", type=int, default=20)
    parser.add_argument("--logzsol-min", type=float, default=-2.0)
    parser.add_argument("--logzsol-max", type=float, default=0.5)
    parser.add_argument("--age-min-gyr", type=float, default=0.01)
    parser.add_argument("--age-max-gyr", type=float, default=13.5)
    parser.add_argument("--sfh-old-bulge-prob", type=float, default=0.62, help="Probability that the dominant host mass component is an old bulge.")
    parser.add_argument("--sfh-intermediate-bulge-prob", type=float, default=0.28, help="Probability that the dominant host mass component is intermediate-age bulge light.")
    parser.add_argument("--sfh-young-burst-prob", type=float, default=0.25, help="Probability of adding a small young/rejuvenated component.")
    parser.add_argument("--sfh-young-burst-min", type=float, default=0.005, help="Minimum mass fraction for the young component when present.")
    parser.add_argument("--sfh-young-burst-max", type=float, default=0.08, help="Typical maximum mass fraction for the young component.")
    parser.add_argument("--sfh-rare-young-burst-prob", type=float, default=0.05, help="Conditional probability of a stronger young component.")
    parser.add_argument("--sfh-rare-young-burst-max", type=float, default=0.15, help="Maximum mass fraction for rare stronger young components.")
    parser.add_argument("--sigma-min", type=float, default=30.0, help="Minimum intrinsic host sigma in km/s. Use <=40 km/s for Greene & Ho-like low-sigma hosts.")
    parser.add_argument("--sigma-max", type=float, default=700.0)
    parser.add_argument("--sigma-sampling-mode", default="science_bins", choices=("science_bins", "uniform", "loguniform"), help="How to draw intrinsic host sigma labels.")
    parser.add_argument("--science-sigma-bin-edges", default="30,60,80,120,180,250,400,700", help="Comma-separated sigma bin edges for science_bins sampling.")
    parser.add_argument("--science-sigma-bin-probs", default="0.07,0.09,0.19,0.21,0.18,0.15,0.11", help="Comma-separated bin probabilities for science_bins sampling.")
    parser.add_argument("--host-fraction-bin-probs", default="0.20,0.38,0.42", help="Comma-separated probabilities for low/mid/high host-fraction draws.")
    parser.add_argument(
        "--agn-parameter-mode",
        default="independent",
        choices=("independent", "ev1_baldwin", "mixed_ev1_baldwin"),
        help=(
            "Draw AGN nuisance parameters independently, with approximate EV1/Baldwin covariance, "
            "or from a mixture of the two priors."
        ),
    )
    parser.add_argument(
        "--agn-covariant-fraction",
        type=float,
        default=0.45,
        help="Fraction of covariant EV1/Baldwin AGN draws when --agn-parameter-mode=mixed_ev1_baldwin.",
    )
    parser.add_argument("--agn-log-l5100-min", type=float, default=42.0, help="Minimum mock log10(lambda L_lambda(5100 A) / erg s^-1).")
    parser.add_argument("--agn-log-l5100-max", type=float, default=46.8, help="Maximum mock log10(lambda L_lambda(5100 A) / erg s^-1).")
    parser.add_argument("--redshift-sampling-mode", default="uniform", choices=("science_bins", "uniform"), help="How to draw mock redshifts.")
    parser.add_argument("--redshift-bin-edges", default="0.01,0.06,0.15,0.35,0.90", help="Comma-separated redshift bin edges for science_bins redshift sampling.")
    parser.add_argument("--redshift-bin-probs", default="0.35,0.25,0.25,0.15", help="Comma-separated redshift bin probabilities.")
    parser.add_argument("--sdss-sigma-inst", type=float, default=70.0, help="Fallback SDSS instrumental Gaussian sigma in km/s.")
    parser.add_argument("--sdss-sigma-inst-min", type=float, default=55.0, help="Lower bound for sampled SDSS instrumental sigma in km/s.")
    parser.add_argument("--sdss-sigma-inst-max", type=float, default=95.0, help="Upper bound for sampled SDSS instrumental sigma in km/s.")
    parser.add_argument(
        "--common-resolution-sigma",
        type=float,
        default=0.0,
        help="If >0, generate mocks at this common instrumental sigma in km/s for common-resolution training.",
    )
    parser.add_argument(
        "--template-sigma-inst",
        type=float,
        default=0.0,
        help=(
            "Effective Gaussian sigma already present in the stellar template/SPS library. "
            "The mock convolution uses sigma_conv^2 = sigma_star^2 + sigma_inst^2 - template_sigma_inst^2."
        ),
    )
    parser.add_argument("--redshift-min", type=float, default=0.01)
    parser.add_argument("--redshift-max", type=float, default=0.90)
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=60.0)
    parser.add_argument("--sky-residual-prob", type=float, default=0.55)
    parser.add_argument("--sky-residual-scale", type=float, default=1.25)
    parser.add_argument("--telluric-residual-prob", type=float, default=0.35)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--park-template", default="scripts/park_template.txt")
    parser.add_argument(
        "--sdss-reference-dir",
        action="append",
        default=[],
        help="Directory containing downloaded SDSS spec*.fits/spec*.fits.gz files; can be passed multiple times.",
    )
    parser.add_argument(
        "--sdss-reference-list",
        action="append",
        default=[],
        help="Text file containing SDSS spectrum paths/globs, one per line; can be passed multiple times.",
    )
    parser.add_argument(
        "--sdss-reference-min-valid-fraction",
        type=float,
        default=0.20,
        help="Minimum fraction of pixels covered by a sampled SDSS ivar reference before falling back to analytic noise.",
    )
    parser.add_argument(
        "--feii-template",
        action="append",
        default=[],
        help="Additional text Fe II template file. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fsps_home:
        os.environ["SPS_HOME"] = args.fsps_home

    config = MockConfig(
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        n_pixels=args.n_pixels,
        n_z=args.n_z,
        n_age=args.n_age,
        logzsol_min=args.logzsol_min,
        logzsol_max=args.logzsol_max,
        age_min_gyr=args.age_min_gyr,
        age_max_gyr=args.age_max_gyr,
        sfh_old_bulge_prob=args.sfh_old_bulge_prob,
        sfh_intermediate_bulge_prob=args.sfh_intermediate_bulge_prob,
        sfh_young_burst_prob=args.sfh_young_burst_prob,
        sfh_young_burst_min=args.sfh_young_burst_min,
        sfh_young_burst_max=args.sfh_young_burst_max,
        sfh_rare_young_burst_prob=args.sfh_rare_young_burst_prob,
        sfh_rare_young_burst_max=args.sfh_rare_young_burst_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_sampling_mode=args.sigma_sampling_mode,
        science_sigma_bin_edges=parse_float_tuple(args.science_sigma_bin_edges, name='science_sigma_bin_edges'),
        science_sigma_bin_probs=parse_float_tuple(args.science_sigma_bin_probs, name='science_sigma_bin_probs'),
        host_fraction_bin_probs=parse_float_tuple(args.host_fraction_bin_probs, name='host_fraction_bin_probs'),
        agn_parameter_mode=args.agn_parameter_mode,
        agn_covariant_fraction=args.agn_covariant_fraction,
        agn_log_l5100_min=args.agn_log_l5100_min,
        agn_log_l5100_max=args.agn_log_l5100_max,
        redshift_sampling_mode=args.redshift_sampling_mode,
        redshift_bin_edges=parse_float_tuple(args.redshift_bin_edges, name='redshift_bin_edges'),
        redshift_bin_probs=parse_float_tuple(args.redshift_bin_probs, name='redshift_bin_probs'),
        sdss_sigma_inst=args.sdss_sigma_inst,
        sdss_sigma_inst_min=args.sdss_sigma_inst_min,
        sdss_sigma_inst_max=args.sdss_sigma_inst_max,
        template_sigma_inst=args.template_sigma_inst,
        common_resolution_sigma=args.common_resolution_sigma,
        redshift_min=args.redshift_min,
        redshift_max=args.redshift_max,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        sky_residual_prob=args.sky_residual_prob,
        sky_residual_scale=args.sky_residual_scale,
        telluric_residual_prob=args.telluric_residual_prob,
        park_template_path=args.park_template,
        feii_template_files=tuple(args.feii_template),
        sdss_reference_paths=tuple(args.sdss_reference_dir + args.sdss_reference_list),
        sdss_reference_min_valid_fraction=args.sdss_reference_min_valid_fraction,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    create_dataset(args.output, args.n_samples, config, seed=args.seed)


if __name__ == "__main__":
    main()
