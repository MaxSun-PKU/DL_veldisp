"""
Utilities for using downloaded SDSS spectra as empirical noise references.

The functions here read common SDSS/BOSS/eBOSS `spec*.fits` files whose
COADD table contains `flux`, `loglam`, and `ivar` columns.  The flux is not
used as a physical component in mocks; the important products are the observed
wavelength grid, inverse-variance profile, and bad-pixel/coverage mask.
When available, the SDSS WDISP/dispersion-resolution column is converted from
Gaussian sigma in log-lambda pixels to km/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np


SPECTRUM_SUFFIXES = (".fits", ".fit", ".fits.gz", ".fit.gz")
C_KMS = 299792.458


@dataclass(frozen=True)
class SDSSReferenceSpectrum:
    path: str
    observed_wave: np.ndarray
    noise_shape: np.ndarray
    valid_mask: np.ndarray
    inst_sigma_kms: np.ndarray
    median_sigma: float
    median_snr: float
    median_inst_sigma_kms: Union[float, None] = None
    redshift: Union[float, None] = None


def is_spectrum_path(path: Union[str, Path]) -> bool:
    name = str(path).lower()
    return any(name.endswith(suffix) for suffix in SPECTRUM_SUFFIXES)


def expand_reference_paths(paths: Sequence[Union[str, Path]]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            for suffix in SPECTRUM_SUFFIXES:
                files.extend(sorted(path.rglob(f"*{suffix}")))
        elif path.is_file() and is_spectrum_path(path):
            files.append(path)
        elif path.is_file():
            base_dir = path.parent
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    listed_path = Path(stripped).expanduser()
                    if not listed_path.is_absolute():
                        listed_path = base_dir / listed_path
                    files.extend(expand_reference_paths([listed_path]))
        else:
            matches = sorted(Path(match) for match in glob(str(path), recursive=True))
            files.extend(match for match in matches if match.is_file() and is_spectrum_path(match))

    unique = []
    seen = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _import_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Reading SDSS FITS reference spectra requires astropy. "
            "Install astropy in the training environment before using --sdss-reference-dir/list."
        ) from exc
    return fits


def _column_lookup(names: Iterable[str]) -> dict[str, str]:
    return {name.lower(): name for name in names}


def velocity_pixel_scale_from_wave(wave: np.ndarray) -> float:
    """Median velocity width of one log-lambda pixel."""
    wave = np.asarray(wave, dtype=np.float64)
    finite = np.isfinite(wave) & (wave > 0)
    if np.count_nonzero(finite) < 3:
        return 69.0
    dloglam = np.diff(np.log(wave[finite]))
    dloglam = dloglam[np.isfinite(dloglam) & (dloglam > 0)]
    if dloglam.size == 0:
        return 69.0
    return float(C_KMS * np.median(dloglam))


def estimate_inst_sigma_kms_from_columns(data, lookup: dict[str, str], observed_wave: np.ndarray) -> np.ndarray:
    """
    Estimate the SDSS instrumental Gaussian sigma in km/s.

    SDSS spec files usually provide WDISP in pixels. On the standard log-lambda
    grid, multiplying WDISP by the local velocity pixel scale gives an
    approximate instrumental sigma. If no resolution column exists, return NaNs
    so callers can fall back to their training default.
    """
    inst_sigma = np.full_like(np.asarray(observed_wave, dtype=np.float32), np.nan, dtype=np.float32)
    resolution_key = "wdisp" if "wdisp" in lookup else "dispres" if "dispres" in lookup else None
    if resolution_key is None:
        return inst_sigma

    wdisp = np.asarray(data[lookup[resolution_key]], dtype=np.float32)
    dv_pixel = velocity_pixel_scale_from_wave(observed_wave)
    candidate = wdisp * float(dv_pixel)
    good = np.isfinite(candidate) & (candidate > 0.0) & (candidate < 300.0)
    inst_sigma[good] = candidate[good].astype(np.float32)
    return inst_sigma


def _find_spectrum_table(hdul):
    for hdu in hdul[1:]:
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if not names:
            continue
        lookup = _column_lookup(names)
        if "flux" in lookup and "ivar" in lookup and ("loglam" in lookup or "wave" in lookup or "wavelength" in lookup):
            return hdu, lookup
    raise ValueError("No table with flux/ivar/loglam columns was found")


def _read_redshift(hdul) -> Union[float, None]:
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


def read_sdss_reference_spectrum(path: Union[str, Path], min_valid_pixels: int = 100) -> SDSSReferenceSpectrum:
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

        sigma = np.zeros_like(ivar, dtype=np.float32)
        sigma[valid] = 1.0 / np.sqrt(ivar[valid])
        finite_sigma = valid & np.isfinite(sigma) & (sigma > 0)
        if int(np.count_nonzero(finite_sigma)) < min_valid_pixels:
            raise ValueError(f"Too few valid ivar pixels in {path}")

        median_sigma = float(np.median(sigma[finite_sigma]))
        noise_shape = np.zeros_like(sigma, dtype=np.float32)
        noise_shape[finite_sigma] = sigma[finite_sigma] / max(median_sigma, 1e-12)

        snr_pix = np.abs(flux[finite_sigma]) * np.sqrt(ivar[finite_sigma])
        median_snr = float(np.median(snr_pix[np.isfinite(snr_pix)])) if snr_pix.size else 0.0
        finite_inst = valid & np.isfinite(inst_sigma_kms) & (inst_sigma_kms > 0)
        median_inst_sigma_kms = float(np.median(inst_sigma_kms[finite_inst])) if np.any(finite_inst) else None
        order = np.argsort(observed_wave)
        return SDSSReferenceSpectrum(
            path=str(path),
            observed_wave=observed_wave[order].astype(np.float32),
            noise_shape=noise_shape[order].astype(np.float32),
            valid_mask=valid[order],
            inst_sigma_kms=inst_sigma_kms[order].astype(np.float32),
            median_sigma=median_sigma,
            median_snr=median_snr,
            median_inst_sigma_kms=median_inst_sigma_kms,
            redshift=_read_redshift(hdul),
        )


class SDSSNoiseLibrary:
    def __init__(self, spectra: Sequence[SDSSReferenceSpectrum]):
        if not spectra:
            raise ValueError("SDSSNoiseLibrary requires at least one valid reference spectrum")
        self.spectra = tuple(spectra)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Union[str, Path]],
        min_valid_pixels: int = 100,
        allow_failures: bool = True,
    ) -> "SDSSNoiseLibrary":
        files = expand_reference_paths(paths)
        if not files:
            raise FileNotFoundError(f"No SDSS FITS spectra found from: {paths}")

        spectra = []
        failures = []
        for file_path in files:
            try:
                spectra.append(read_sdss_reference_spectrum(file_path, min_valid_pixels=min_valid_pixels))
            except Exception as exc:
                failures.append(f"{file_path}: {exc}")
                if not allow_failures:
                    raise

        if not spectra:
            joined = "\n".join(failures[:10])
            raise ValueError(f"No usable SDSS reference spectra were loaded.\n{joined}")
        if failures:
            print(f"[SDSSNoiseLibrary] skipped {len(failures)} unusable reference spectra")
        print(f"[SDSSNoiseLibrary] loaded {len(spectra)} empirical ivar/noise references")
        return cls(spectra)

    @property
    def manifest(self) -> list[dict[str, Union[str, float, None]]]:
        return [
            {
                "index": i,
                "path": spectrum.path,
                "wave_min": float(np.nanmin(spectrum.observed_wave)),
                "wave_max": float(np.nanmax(spectrum.observed_wave)),
                "median_sigma": spectrum.median_sigma,
                "median_snr": spectrum.median_snr,
                "median_inst_sigma_kms": spectrum.median_inst_sigma_kms,
                "redshift": spectrum.redshift,
            }
            for i, spectrum in enumerate(self.spectra)
        ]

    @property
    def finite_median_inst_sigmas(self) -> np.ndarray:
        values = [
            spectrum.median_inst_sigma_kms
            for spectrum in self.spectra
            if spectrum.median_inst_sigma_kms is not None and np.isfinite(spectrum.median_inst_sigma_kms)
        ]
        return np.asarray(values, dtype=np.float32)

    def sample_on_observed_grid(
        self,
        observed_wave: np.ndarray,
        rng: np.random.Generator,
        min_valid_fraction: float = 0.20,
        max_tries: int = 24,
    ) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray, int], None]:
        observed_wave = np.asarray(observed_wave, dtype=np.float32)
        for _ in range(max_tries):
            index = int(rng.integers(0, len(self.spectra)))
            spectrum = self.spectra[index]
            noise_shape = np.interp(
                observed_wave,
                spectrum.observed_wave,
                spectrum.noise_shape,
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)
            valid = (
                np.interp(
                    observed_wave,
                    spectrum.observed_wave,
                    spectrum.valid_mask.astype(np.float32),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
            valid &= np.isfinite(noise_shape) & (noise_shape > 0)
            if np.count_nonzero(valid) / max(valid.size, 1) >= min_valid_fraction:
                inst_sigma_kms = np.interp(
                    observed_wave,
                    spectrum.observed_wave,
                    spectrum.inst_sigma_kms,
                    left=np.nan,
                    right=np.nan,
                ).astype(np.float32)
                return noise_shape, valid, inst_sigma_kms, index
        return None
