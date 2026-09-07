"""
Select representative spectra from an SDSS-like mock HDF5 file and plot them.

The selection is label-based so the figure is reproducible and covers a useful
range of cases for a paper: low/high velocity dispersion, host/AGN dominated
continua, young bursts, low metallicity, and Fe II contamination.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REST_LINES = (
    (2798.0, "Mg II"),
    (3934.0, "Ca K"),
    (4862.7, "H beta"),
    (5008.2, "[O III]"),
    (8542.0, "CaT"),
)


CASE_SPECS = (
    {
        "case": "low_sigma_host_old_metal_rich",
        "title": "Low sigma, low-z CaT coverage",
        "targets": {
            "sigma_star": 110.0,
            "host_fraction": 0.75,
            "mass_weighted_age_gyr": 8.5,
            "mass_weighted_logzsol": 0.15,
            "redshift": 0.035,
        },
        "weights": {
            "sigma_star": 1 / 80.0,
            "host_fraction": 4.0,
            "mass_weighted_age_gyr": 0.18,
            "mass_weighted_logzsol": 1.4,
            "redshift": 12.0,
        },
    },
    {
        "case": "intermediate_sigma_feii_strong",
        "title": "Intermediate sigma, Fe II strong",
        "targets": {"sigma_star": 260.0, "host_fraction": 0.50, "r_feii": 2.4, "feii_fwhm": 4500.0},
        "weights": {"sigma_star": 1 / 120.0, "host_fraction": 3.0, "r_feii": 1.3, "feii_fwhm": 1 / 3500.0},
    },
    {
        "case": "high_sigma_agn_dominated_low_snr",
        "title": "High sigma, AGN dominated",
        "targets": {"sigma_star": 520.0, "host_fraction": 0.18, "snr": 10.0, "balmer_fwhm": 9000.0},
        "weights": {"sigma_star": 1 / 160.0, "host_fraction": 5.0, "snr": 0.07, "balmer_fwhm": 1 / 5000.0},
    },
    {
        "case": "extreme_sigma_old_metal_poor",
        "title": "Extreme sigma, old metal poor",
        "targets": {"sigma_star": 660.0, "host_fraction": 0.65, "mass_weighted_age_gyr": 10.0, "mass_weighted_logzsol": -1.4},
        "weights": {"sigma_star": 1 / 120.0, "host_fraction": 3.0, "mass_weighted_age_gyr": 0.15, "mass_weighted_logzsol": 1.0},
    },
    {
        "case": "young_burst_mixed",
        "title": "Young component, mixed light",
        "targets": {"sigma_star": 180.0, "host_fraction": 0.45, "young_fraction": 0.25, "mass_weighted_age_gyr": 1.2},
        "weights": {"sigma_star": 1 / 120.0, "host_fraction": 3.0, "young_fraction": 5.0, "mass_weighted_age_gyr": 0.4},
    },
)


def read_label(hf: h5py.File, key: str, fallback: float = 0.0) -> np.ndarray:
    if key in hf:
        return hf[key][:].astype(float)
    return np.full(len(hf["spectra"]), fallback, dtype=float)


def select_examples(hf: h5py.File) -> list[dict[str, float | int | str]]:
    labels = {
        key: read_label(hf, key)
        for key in (
            "sigma_star",
            "host_fraction",
            "snr",
            "r_feii",
            "feii_fwhm",
            "balmer_fwhm",
            "young_fraction",
            "mass_weighted_age_gyr",
            "mass_weighted_logzsol",
            "redshift",
        )
    }

    selected = []
    used = set()
    for spec in CASE_SPECS:
        score = np.zeros(len(hf["spectra"]), dtype=float)
        for key, target in spec["targets"].items():
            weight = spec["weights"].get(key, 1.0)
            score += (weight * (labels[key] - target)) ** 2
        for index in np.argsort(score):
            index = int(index)
            if index not in used:
                used.add(index)
                row = {"case": spec["case"], "title": spec["title"], "index": index, "score": float(score[index])}
                row.update({key: float(value[index]) for key, value in labels.items()})
                selected.append(row)
                break
    return selected


def contiguous_segments(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, cuts)


def finite_values(*arrays: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = []
    for array in arrays:
        finite = mask & np.isfinite(array)
        if np.any(finite):
            values.append(array[finite])
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values).astype(float)


def set_display_limits(
    ax,
    wave: np.ndarray,
    valid: np.ndarray,
    noisy: np.ndarray,
    clean: np.ndarray,
    host_component: np.ndarray,
    agn_component: np.ndarray,
) -> None:
    good = valid & np.isfinite(noisy) & np.isfinite(clean)
    if not np.any(good):
        return

    valid_wave = wave[good]
    x_pad = 0.015 * (valid_wave.max() - valid_wave.min())
    ax.set_xlim(valid_wave.min() - x_pad, valid_wave.max() + x_pad)

    all_values = finite_values(noisy, clean, host_component, agn_component, mask=good)
    if all_values.size == 0:
        return

    y_lo = float(np.nanmin(all_values))
    y_hi = float(np.nanmax(all_values))
    span = max(y_hi - y_lo, 0.30)
    y_lo -= 0.08 * span
    y_hi += 0.12 * span
    ax.set_ylim(y_lo, y_hi)


def plot_case(
    ax,
    hf: h5py.File,
    row: dict[str, float | int | str],
    wave: np.ndarray,
    panel_label: str | None = None,
) -> None:
    index = int(row["index"])
    noisy = hf["spectra"][index]
    clean = hf["clean_spectra"][index]
    agn = hf["agn_spectra"][index]
    valid = hf["valid_pixels"][index] > 0
    host_fraction = float(row["host_fraction"])
    host_component = clean - (1.0 - host_fraction) * agn
    agn_component = (1.0 - host_fraction) * agn

    for segment in contiguous_segments(valid):
        ax.plot(wave[segment], noisy[segment], color="0.12", lw=0.7, alpha=0.70)
        ax.plot(wave[segment], clean[segment], color="#1f77b4", lw=1.1)
        ax.plot(wave[segment], host_component[segment], color="#16883a", lw=1.35, ls="--", alpha=0.95)
        ax.plot(wave[segment], agn_component[segment], color="#d62728", lw=0.9, ls=":", alpha=0.85)

    set_display_limits(ax, wave, valid, noisy, clean, host_component, agn_component)

    for line_wave, label in REST_LINES:
        near_line = valid & (np.abs(wave - line_wave) < 25.0)
        if np.any(near_line):
            ax.axvline(line_wave, color="0.70", lw=0.6, alpha=0.6)
            label_y = 0.86 if label == "H beta" else 0.98
            ax.text(line_wave, label_y, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color="0.35")

    prefix = f"({panel_label}) " if panel_label else ""
    subtitle = (
        f"{prefix}{row['title']}: "
        rf"$\sigma_\star={row['sigma_star']:.0f}$ km s$^{{-1}}$, "
        rf"$f_{{\rm host}}={row['host_fraction']:.2f}$, S/N={row['snr']:.0f}"
    )
    ax.set_title(subtitle, loc="left", fontsize=8.5, pad=3.5)
    ax.set_ylabel("Normalized flux", fontsize=8.5)
    ax.tick_params(labelsize=7.5)


def write_selection_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output-dir", default="figures/mock_examples")
    parser.add_argument("--prefix", default="mock_spectra_examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as hf:
        wave = hf["wave_grid"][:]
        selected = select_examples(hf)
        write_selection_csv(selected, output_dir / f"{args.prefix}_selection.csv")

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "axes.linewidth": 0.8,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "pdf.fonttype": 42,
            }
        )
        fig, axes = plt.subplots(len(selected), 1, figsize=(7.2, 7.1), sharex=False)
        if len(selected) == 1:
            axes = [axes]
        for panel_index, (ax, row) in enumerate(zip(axes, selected)):
            plot_case(ax, hf, row, wave, panel_label=chr(ord("a") + panel_index))
        axes[-1].set_xlabel(r"Rest wavelength [$\AA$]", fontsize=8.5)
        legend_handles = [
            axes[0].plot([], [], color="0.12", lw=0.8, label="Noisy mock")[0],
            axes[0].plot([], [], color="#1f77b4", lw=1.1, label="Clean total")[0],
            axes[0].plot([], [], color="#16883a", lw=1.35, ls="--", label="Host contribution")[0],
            axes[0].plot([], [], color="#d62728", lw=0.9, ls=":", label="AGN contribution")[0],
        ]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=0.55)

        pdf_path = output_dir / f"{args.prefix}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {pdf_path}")

        for row in selected:
            fig, ax = plt.subplots(figsize=(10.0, 3.0), constrained_layout=True)
            plot_case(ax, hf, row, wave)
            ax.set_xlabel("Rest wavelength [Angstrom]")
            single_pdf = output_dir / f"{args.prefix}_{row['case']}.pdf"
            fig.savefig(single_pdf)
            plt.close(fig)
            print(f"Saved {single_pdf}")


if __name__ == "__main__":
    main()
