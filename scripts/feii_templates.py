"""
Fe II template loading utilities for SDSS/quasar mock generation.

Template files are expected to be plain text with at least two numeric columns:
wavelength [Angstrom] and flux. A third uncertainty column is allowed and ignored.
If the first column looks log-scaled, it is interpreted as log10(wavelength / A).

Recommended local files:
- scripts/park_template.txt
- scripts/fe_template/fe_uv.txt                   # user-collected PyQSOFit/Shen UV composite
- scripts/fe_template/fe_optical.txt              # user-collected PyQSOFit optical template
- reference/feii_templates/fe_uv.txt              # working-copy fallback
- reference/feii_templates/fe_optical.txt         # working-copy fallback
- reference/feii_templates/vestergaard_wilkes_2001.txt
- reference/feii_templates/tsuzuki_2006.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class FeIITemplateSpec:
    name: str
    path: str
    source: str
    required: bool = False


DEFAULT_TEMPLATE_SPECS = (
    FeIITemplateSpec(
        name="park2022_mrk493_optical",
        path="scripts/park_template.txt",
        source="Park et al. 2022 Mrk493STIS optical Fe II, 4000-5600 A",
        required=True,
    ),
    FeIITemplateSpec(
        name="user_pyqsofit_uv_composite",
        path="scripts/fe_template/fe_uv.txt",
        source="User-collected PyQSOFit/Shen UV composite: VW01 + Salviander + Tsuzuki06",
    ),
    FeIITemplateSpec(
        name="user_pyqsofit_optical",
        path="scripts/fe_template/fe_optical.txt",
        source="User-collected PyQSOFit optical Fe II template",
    ),
    FeIITemplateSpec(
        name="reference_pyqsofit_uv_composite",
        path="reference/feii_templates/fe_uv.txt",
        source="Reference fallback PyQSOFit/Shen UV composite: VW01 + Salviander + Tsuzuki06",
    ),
    FeIITemplateSpec(
        name="reference_pyqsofit_optical",
        path="reference/feii_templates/fe_optical.txt",
        source="Reference fallback PyQSOFit optical Fe II template",
    ),
    FeIITemplateSpec(
        name="vw2001_uv",
        path="reference/feii_templates/vestergaard_wilkes_2001.txt",
        source="Vestergaard & Wilkes 2001 UV Fe II/Fe III",
    ),
    FeIITemplateSpec(
        name="tsuzuki2006_uv_optical",
        path="reference/feii_templates/tsuzuki_2006.txt",
        source="Tsuzuki et al. 2006 UV/optical Fe II",
    ),
)


def _as_path(path: Union[str, Path], root: Union[str, Path] = ".") -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def read_numeric_template(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!", "|", "=")):
                continue
            parts = stripped.replace(",", " ").split()
            numeric = []
            for part in parts:
                try:
                    numeric.append(float(part))
                except ValueError:
                    continue
            if len(numeric) >= 2:
                rows.append((numeric[0], numeric[1]))

    if not rows:
        raise ValueError(f"No wavelength/flux rows found in {path}")

    arr = np.asarray(rows, dtype=float)
    wave, flux = arr[:, 0], arr[:, 1]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    if wave.size < 2:
        raise ValueError(f"Template {path} has fewer than two usable rows")
    if np.nanmax(wave) < 10.0:
        wave = 10.0 ** wave
    return wave, flux


def normalize_template(flux: np.ndarray) -> np.ndarray:
    flux = np.clip(np.asarray(flux, dtype=float), 0.0, None)
    positive = flux[flux > 0]
    if positive.size:
        scale = np.percentile(positive, 90)
        if np.isfinite(scale) and scale > 0:
            flux = flux / scale
    return flux.astype(np.float32)


def resample_template(path: Union[str, Path], target_wave: np.ndarray) -> np.ndarray:
    wave, flux = read_numeric_template(path)
    resampled = np.interp(target_wave, wave, flux, left=0.0, right=0.0)
    return normalize_template(resampled)


def is_duplicate_template(template: np.ndarray, templates: Sequence[np.ndarray]) -> bool:
    return any(np.allclose(template, existing, rtol=0.0, atol=1e-6) for existing in templates)


def specs_from_files(paths: Sequence[str], source: str = "user-supplied Fe II template") -> list[FeIITemplateSpec]:
    specs = []
    for path in paths:
        p = Path(path)
        specs.append(FeIITemplateSpec(name=p.stem, path=str(p), source=source))
    return specs


def load_feii_template_bank(
    target_wave: np.ndarray,
    specs: Optional[Iterable[FeIITemplateSpec]] = None,
    root: Union[str, Path] = ".",
) -> tuple[np.ndarray, list[dict[str, str]]]:
    specs = tuple(DEFAULT_TEMPLATE_SPECS if specs is None else specs)
    templates = []
    manifest = []

    for spec in specs:
        path = _as_path(spec.path, root=root)
        if not path.exists():
            if spec.required:
                raise FileNotFoundError(f"Required Fe II template is missing: {path}")
            continue

        template = resample_template(path, target_wave)
        if np.count_nonzero(template) == 0:
            continue
        if is_duplicate_template(template, templates):
            continue

        templates.append(template)
        support = target_wave[template > 0]
        manifest.append(
            {
                "name": spec.name,
                "path": str(path),
                "source": spec.source,
                "wave_min": f"{float(support.min()):.1f}",
                "wave_max": f"{float(support.max()):.1f}",
            }
        )

    if not templates:
        raise ValueError("No Fe II templates could be loaded")
    return np.asarray(templates, dtype=np.float32), manifest
