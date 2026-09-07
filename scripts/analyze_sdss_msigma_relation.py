"""
Analyze whether the DL velocity-dispersion measurements recover M-sigma.

Inputs are the manifest written by select_sdss_msigma_sample.py and the real
SDSS inference CSV written by infer_sdss_quasar_sigma.py.  The script makes a
diagnostic M-sigma figure and writes summary tables for the full, science-grade,
and golden subsamples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--inference-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sigma-column", default="recommended_sigma_kms")
    parser.add_argument("--sigma-error-column", default="recommended_sigma_err_kms")
    parser.add_argument("--mbh-column", default="log_bh_hb_vp06")
    parser.add_argument("--max-error-science", type=float, default=75.0)
    parser.add_argument("--min-host-science", type=float, default=0.35)
    parser.add_argument("--min-snr-science", type=float, default=10.0)
    parser.add_argument("--min-resolution-ratio", type=float, default=1.15)
    parser.add_argument("--max-fit-recon-mae", type=float, default=0.30)
    parser.add_argument("--max-sigma-recon-mae", type=float, default=0.25)
    parser.add_argument("--plot-title", default="SDSS DR7 low-z Hbeta AGN pilot")
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def local_relation(log_sigma200: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha + beta * log_sigma200


def robust_scatter(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def summarize(name: str, table: pd.DataFrame, sigma_col: str, mbh_col: str) -> dict[str, float | str]:
    finite = np.isfinite(table[sigma_col]) & np.isfinite(table[mbh_col]) & (table[sigma_col] > 0)
    sub = table.loc[finite].copy()
    if len(sub) == 0:
        return {"sample": name, "n": 0}
    log_sigma200 = np.log10(sub[sigma_col].to_numpy(float) / 200.0)
    log_mbh = sub[mbh_col].to_numpy(float)
    tremaine = local_relation(log_sigma200, 8.13, 4.02)
    gultekin = local_relation(log_sigma200, 8.12, 4.24)
    residual_t02 = log_mbh - tremaine
    residual_g09 = log_mbh - gultekin
    corr = (
        float(np.corrcoef(log_sigma200, log_mbh)[0, 1])
        if np.nanstd(log_sigma200) > 0 and np.nanstd(log_mbh) > 0
        else float("nan")
    )
    if len(sub) >= 3 and np.nanstd(log_sigma200) > 0:
        slope, intercept = np.polyfit(log_sigma200, log_mbh, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "sample": name,
        "n": int(len(sub)),
        "median_sigma_kms": float(np.nanmedian(sub[sigma_col])),
        "median_log_mbh": float(np.nanmedian(log_mbh)),
        "corr_logsigma_logmbh": corr,
        "fit_intercept_at_200": float(intercept),
        "fit_slope": float(slope),
        "median_delta_tremaine02_dex": float(np.nanmedian(residual_t02)),
        "scatter_delta_tremaine02_dex": robust_scatter(residual_t02),
        "median_delta_gultekin09_dex": float(np.nanmedian(residual_g09)),
        "scatter_delta_gultekin09_dex": robust_scatter(residual_g09),
    }


def build_quality_masks(table: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, pd.Series, pd.Series]:
    finite = (
        np.isfinite(table[args.sigma_column])
        & np.isfinite(table[args.mbh_column])
        & (table[args.sigma_column] > 0)
    )
    science_ready_native = as_bool(table["science_ready"]) if "science_ready" in table else pd.Series(False, index=table.index)
    golden_native = as_bool(table["golden_sample"]) if "golden_sample" in table else pd.Series(False, index=table.index)
    relaxed_science = (
        finite
        & np.isfinite(table[args.sigma_error_column])
        & (table[args.sigma_error_column] <= args.max_error_science)
        & np.isfinite(table["pred_host_fraction"])
        & (table["pred_host_fraction"] >= args.min_host_science)
        & np.isfinite(table["median_snr"])
        & (table["median_snr"] >= args.min_snr_science)
        & np.isfinite(table["pred_sigma_resolution_ratio"])
        & (table["pred_sigma_resolution_ratio"] >= args.min_resolution_ratio)
        & np.isfinite(table["fit_recon_mae"])
        & (table["fit_recon_mae"] <= args.max_fit_recon_mae)
        & np.isfinite(table["sigma_recon_mae"])
        & (table["sigma_recon_mae"] <= args.max_sigma_recon_mae)
    )
    science = finite & (science_ready_native | relaxed_science)
    golden = finite & (golden_native | (science & (table[args.sigma_error_column] <= 50.0) & (table["pred_host_fraction"] >= 0.45)))
    return finite, science, golden


def load_and_merge(manifest_csv: Path, inference_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    inference = pd.read_csv(inference_csv)
    if "sdss_name" not in manifest.columns or "sdss_name" not in inference.columns:
        raise ValueError("Both manifest and inference CSV must contain sdss_name.")
    merged = inference.merge(
        manifest.drop(columns=[col for col in ("local_path", "z") if col in manifest.columns], errors="ignore"),
        on="sdss_name",
        how="left",
        suffixes=("", "_manifest"),
    )
    return merged


def plot_msigma(table: pd.DataFrame, finite: pd.Series, science: pd.Series, golden: pd.Series, args: argparse.Namespace) -> Path:
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_prefix.with_name(output_prefix.name + "_msigma_relation.pdf")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "font.size": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2), constrained_layout=True)
    ax = axes[0, 0]

    sigma_grid = np.logspace(np.log10(45.0), np.log10(720.0), 300)
    log_sigma200 = np.log10(sigma_grid / 200.0)
    ax.fill_between(
        sigma_grid,
        local_relation(log_sigma200, 8.13, 4.02) - 0.5,
        local_relation(log_sigma200, 8.13, 4.02) + 0.5,
        color="0.88",
        lw=0,
        label=r"T02 $\pm$0.5 dex",
    )
    ax.plot(sigma_grid, local_relation(log_sigma200, 8.13, 4.02), color="0.12", lw=1.8, label="Tremaine+02")
    ax.plot(sigma_grid, local_relation(log_sigma200, 8.12, 4.24), color="#5f6f7a", lw=1.5, ls="--", label="Gultekin+09")

    all_mask = finite & ~science
    ax.scatter(
        table.loc[all_mask, args.sigma_column],
        table.loc[all_mask, args.mbh_column],
        s=11,
        c="0.72",
        alpha=0.45,
        edgecolors="none",
        label="low confidence",
    )
    ax.errorbar(
        table.loc[science & ~golden, args.sigma_column],
        table.loc[science & ~golden, args.mbh_column],
        xerr=table.loc[science & ~golden, args.sigma_error_column],
        fmt="o",
        ms=4.0,
        lw=0.7,
        color="#2f7bbd",
        ecolor="#9abbe0",
        alpha=0.82,
        label="science grade",
    )
    ax.errorbar(
        table.loc[golden, args.sigma_column],
        table.loc[golden, args.mbh_column],
        xerr=table.loc[golden, args.sigma_error_column],
        fmt="*",
        ms=8.5,
        lw=0.8,
        color="#d18b00",
        ecolor="#edc567",
        mec="0.25",
        mew=0.4,
        label="golden",
    )
    ax.set_xscale("log")
    ax.set_xlim(45, 720)
    ax.set_ylim(6.4, 10.2)
    ax.set_xlabel(r"$\sigma_\star$ from DL model [km s$^{-1}$]")
    ax.set_ylabel(r"$\log(M_{\rm BH}/M_\odot)$ from H$\beta$")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title("(a) M--sigma plane", fontsize=11)

    ax = axes[0, 1]
    log_sigma200_obs = np.log10(table.loc[finite, args.sigma_column].to_numpy(float) / 200.0)
    delta = table.loc[finite, args.mbh_column].to_numpy(float) - local_relation(log_sigma200_obs, 8.13, 4.02)
    scatter = ax.scatter(
        table.loc[finite, "pred_host_fraction"],
        delta,
        c=table.loc[finite, args.sigma_error_column],
        s=12,
        cmap="viridis_r",
        alpha=0.70,
        edgecolors="none",
        vmin=20,
        vmax=120,
    )
    ax.axhline(0, color="0.25", lw=1.1)
    ax.axhspan(-0.5, 0.5, color="0.90", zorder=-1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Predicted host fraction")
    ax.set_ylabel(r"$\Delta\log M_{\rm BH}$ from T02")
    ax.set_title("(b) Offset versus host fraction", fontsize=11)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"$\sigma_\star$ error [km s$^{-1}$]")

    ax = axes[1, 0]
    bins = np.linspace(-2.0, 2.0, 41)
    ax.hist(delta, bins=bins, histtype="stepfilled", color="0.75", alpha=0.55, label="all finite")
    if np.any(science):
        d_s = table.loc[science, args.mbh_column].to_numpy(float) - local_relation(
            np.log10(table.loc[science, args.sigma_column].to_numpy(float) / 200.0), 8.13, 4.02
        )
        ax.hist(d_s, bins=bins, histtype="step", lw=1.6, color="#2f7bbd", label="science grade")
    if np.any(golden):
        d_g = table.loc[golden, args.mbh_column].to_numpy(float) - local_relation(
            np.log10(table.loc[golden, args.sigma_column].to_numpy(float) / 200.0), 8.13, 4.02
        )
        ax.hist(d_g, bins=bins, histtype="step", lw=1.8, color="#d18b00", label="golden")
    ax.axvline(0, color="0.25", lw=1.1)
    ax.set_xlabel(r"$\Delta\log M_{\rm BH}$ from T02")
    ax.set_ylabel("Number")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("(c) Offset distribution", fontsize=11)

    ax = axes[1, 1]
    ax.scatter(
        table.loc[finite, "median_snr"],
        table.loc[finite, args.sigma_error_column],
        c=np.where(golden.loc[finite], "#d18b00", np.where(science.loc[finite], "#2f7bbd", "0.65")),
        s=13,
        alpha=0.72,
        edgecolors="none",
    )
    ax.axhline(50, color="#d18b00", lw=1.0, ls="--", alpha=0.85)
    ax.axhline(75, color="#2f7bbd", lw=1.0, ls=":", alpha=0.85)
    ax.set_xlim(0, max(85, np.nanpercentile(table.loc[finite, "median_snr"], 98) * 1.05))
    ax.set_ylim(0, max(150, np.nanpercentile(table.loc[finite, args.sigma_error_column], 98) * 1.08))
    ax.set_xlabel("SDSS spectrum S/N")
    ax.set_ylabel(r"Recommended $\sigma_\star$ error [km s$^{-1}$]")
    ax.set_title("(d) Measurement precision", fontsize=11)

    fig.suptitle(args.plot_title, fontsize=12)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    args.output_prefix = str(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(Path(args.manifest_csv), Path(args.inference_csv))
    if args.sigma_column not in merged.columns:
        raise ValueError(f"Missing sigma column {args.sigma_column!r}")
    if args.sigma_error_column not in merged.columns:
        raise ValueError(f"Missing sigma error column {args.sigma_error_column!r}")
    if args.mbh_column not in merged.columns:
        raise ValueError(f"Missing MBH column {args.mbh_column!r}")

    finite, science, golden = build_quality_masks(merged, args)
    merged["msigma_finite"] = finite
    merged["msigma_science_grade"] = science
    merged["msigma_golden"] = golden
    log_sigma200 = np.log10(merged[args.sigma_column].clip(lower=1e-6) / 200.0)
    merged["delta_logmbh_tremaine02"] = merged[args.mbh_column] - local_relation(log_sigma200, 8.13, 4.02)
    merged["delta_logmbh_gultekin09"] = merged[args.mbh_column] - local_relation(log_sigma200, 8.12, 4.24)

    summary_rows = [
        summarize("all_finite", merged.loc[finite], args.sigma_column, args.mbh_column),
        summarize("science_grade", merged.loc[science], args.sigma_column, args.mbh_column),
        summarize("golden", merged.loc[golden], args.sigma_column, args.mbh_column),
    ]
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    table_path = output_prefix.with_name(output_prefix.name + "_merged_results.csv")
    science_path = output_prefix.with_name(output_prefix.name + "_science_sample.csv")
    pdf_path = plot_msigma(merged, finite, science, golden, args)

    merged.to_csv(table_path, index=False)
    merged.loc[science].sort_values(args.sigma_error_column).to_csv(science_path, index=False)
    summary = {
        "relations": {
            "tremaine2002": "log(M_BH/Msun)=8.13+4.02*log10(sigma/200 km/s)",
            "gultekin2009_all": "log(M_BH/Msun)=8.12+4.24*log10(sigma/200 km/s)",
        },
        "quality_definition": {
            "science_grade": (
                "native science_ready OR relaxed cuts on sigma error, host fraction, S/N, "
                "resolution ratio, and reconstruction residual"
            ),
            "golden": "native golden_sample OR science_grade with sigma error <=50 km/s and host fraction >=0.45",
        },
        "samples": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {science_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
