"""Draw the black hole mass--velocity dispersion application figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import odr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--sigma-column", default="recommended_sigma_kms")
    parser.add_argument("--sigma-error-column", default="recommended_sigma_err_kms")
    parser.add_argument("--mbh-column", default="log_bh_ha")
    parser.add_argument("--title", default=r"Low-redshift SDSS type 1 AGNs: $M_{\rm BH}({\rm H}\alpha)\!-\!\sigma_\star$")
    parser.add_argument("--odr-mask", choices=("science_ready", "grade_b", "golden", "all"), default="science_ready")
    parser.add_argument("--virial-mass-error-dex", type=float, default=0.40)
    return parser.parse_args()


def bool_column(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table:
        return pd.Series(False, index=table.index)
    series = table[name]
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


def summarize(name: str, table: pd.DataFrame, sigma_col: str, mbh_col: str) -> dict[str, float | str | int]:
    finite = np.isfinite(table[sigma_col]) & np.isfinite(table[mbh_col]) & (table[sigma_col] > 0)
    sub = table.loc[finite].copy()
    if len(sub) == 0:
        return {"sample": name, "n": 0}
    x = np.log10(sub[sigma_col].to_numpy(float) / 200.0)
    y = sub[mbh_col].to_numpy(float)
    residual_t02 = y - local_relation(x, 8.13, 4.02)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(sub) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0 else float("nan")
    if len(sub) >= 3 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "sample": name,
        "n": int(len(sub)),
        "median_sigma_kms": float(np.nanmedian(sub[sigma_col])),
        "median_log_mbh": float(np.nanmedian(y)),
        "corr_logsigma_logmbh": corr,
        "fit_intercept_at_200": float(intercept),
        "fit_slope": float(slope),
        "median_delta_tremaine02_dex": float(np.nanmedian(residual_t02)),
        "scatter_delta_tremaine02_dex": robust_scatter(residual_t02),
    }


def bin_medians(table: pd.DataFrame, mask: pd.Series, sigma_col: str, mbh_col: str) -> pd.DataFrame:
    bins = np.array([30.0, 60.0, 80.0, 120.0, 180.0, 250.0, 400.0])
    sub = table.loc[mask & np.isfinite(table[sigma_col]) & np.isfinite(table[mbh_col])].copy()
    sub["sigma_bin"] = pd.cut(sub[sigma_col], bins=bins)
    rows = []
    for interval, group in sub.groupby("sigma_bin", observed=False):
        if len(group) < 5:
            continue
        rows.append(
            {
                "sigma": float(np.nanmedian(group[sigma_col])),
                "log_mbh": float(np.nanmedian(group[mbh_col])),
                "log_mbh_err": robust_scatter(group[mbh_col].to_numpy(float)) / max(np.sqrt(len(group)), 1.0),
                "n": int(len(group)),
                "label": str(interval),
            }
        )
    return pd.DataFrame(rows)


def odr_fit(
    table: pd.DataFrame,
    mask: pd.Series,
    sigma_col: str,
    sigma_err_col: str,
    mbh_col: str,
    virial_mass_error_dex: float,
) -> dict[str, float | int | str]:
    finite = (
        mask
        & np.isfinite(table[sigma_col])
        & np.isfinite(table[sigma_err_col])
        & np.isfinite(table[mbh_col])
        & (table[sigma_col] > 0)
        & (table[sigma_err_col] > 0)
    )
    sub = table.loc[finite].copy()
    if len(sub) < 5:
        return {"n": int(len(sub)), "ok": 0, "reason": "fewer than five finite points"}

    sigma = sub[sigma_col].to_numpy(float)
    sigma_err = sub[sigma_err_col].to_numpy(float)
    x = np.log10(sigma / 200.0)
    y = sub[mbh_col].to_numpy(float)
    sx = np.clip(sigma_err / (sigma * np.log(10.0)), 0.03, 0.50)
    sy = np.full_like(y, float(virial_mass_error_dex))
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return {"n": int(len(sub)), "ok": 0, "reason": "zero dynamic range"}

    def line(beta, values):
        return beta[0] + beta[1] * values

    beta0 = np.polyfit(x, y, 1)[::-1]
    result = odr.ODR(odr.RealData(x, y, sx=sx, sy=sy), odr.Model(line), beta0=beta0, maxit=1000).run()
    return {
        "n": int(len(sub)),
        "ok": 1,
        "alpha": float(result.beta[0]),
        "beta": float(result.beta[1]),
        "alpha_err": float(result.sd_beta[0]),
        "beta_err": float(result.sd_beta[1]),
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "sigma_min": float(np.nanmin(sigma)),
        "sigma_max": float(np.nanmax(sigma)),
        "stopreason": "; ".join(result.stopreason),
    }


def format_sigma_axis(ax) -> None:
    ax.set_xticks([40, 60, 100, 200, 300])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):d}" if value >= 10 else f"{value:g}"))


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_summary) if args.output_summary else output_pdf.with_suffix(".summary.json")

    table = pd.read_csv(input_csv)
    sigma_col = args.sigma_column
    mbh_col = args.mbh_column
    err_col = args.sigma_error_column
    finite = np.isfinite(table[sigma_col]) & np.isfinite(table[mbh_col]) & (table[sigma_col] > 0)
    golden = bool_column(table, "golden_sample") | bool_column(table, "high_confidence")
    science = bool_column(table, "science_ready")
    grade = table.get("science_grade", pd.Series("C", index=table.index)).astype(str)
    grade_b = science & ~golden & grade.eq("B")
    low_conf = finite & ~science
    science_all = finite & science
    odr_masks = {
        "science_ready": science_all,
        "grade_b": grade_b,
        "golden": golden,
        "all": finite,
    }
    summaries = [
        summarize("all_finite", table.loc[finite], sigma_col, mbh_col),
        summarize("quality_selected_AB", table.loc[science_all], sigma_col, mbh_col),
        summarize("intermediate_quality_B", table.loc[grade_b], sigma_col, mbh_col),
        summarize("golden_A", table.loc[golden], sigma_col, mbh_col),
        summarize("low_confidence_C", table.loc[low_conf], sigma_col, mbh_col),
    ]
    odr_result = odr_fit(
        table,
        odr_masks[args.odr_mask],
        sigma_col,
        err_col,
        mbh_col,
        args.virial_mass_error_dex,
    )
    odr_golden = odr_fit(table, golden, sigma_col, err_col, mbh_col, args.virial_mass_error_dex)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "font.size": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8))
    fig.subplots_adjust(left=0.085, right=0.90, bottom=0.08, top=0.80, wspace=0.23, hspace=0.29)
    fig.suptitle(args.title, fontsize=12.0, y=0.975)

    xlim = (36.0, 310.0)
    ylim = (5.55, 8.55)
    sigma_grid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 300)
    x_grid = np.log10(sigma_grid / 200.0)

    ax = axes[0, 0]
    ax.fill_between(
        sigma_grid,
        local_relation(x_grid, 8.13, 4.02) - 0.5,
        local_relation(x_grid, 8.13, 4.02) + 0.5,
        color="0.91",
        lw=0,
        label=r"Tremaine+02 $\pm$0.5 dex",
    )
    ax.plot(sigma_grid, local_relation(x_grid, 8.13, 4.02), color="0.14", lw=1.6, label="Tremaine+02")
    ax.plot(sigma_grid, local_relation(x_grid, 8.12, 4.24), color="#526979", lw=1.3, ls="--", label="Gultekin+09")
    if odr_result.get("ok"):
        odr_x_grid = x_grid
        odr_y_grid = float(odr_result["alpha"]) + float(odr_result["beta"]) * odr_x_grid
        ax.plot(
            sigma_grid,
            odr_y_grid,
            color="#b13a2e",
            lw=1.8,
            ls="-.",
            label="ODR fit",
        )
    ax.scatter(table.loc[low_conf, sigma_col], table.loc[low_conf, mbh_col], s=12, c="0.72", alpha=0.34, edgecolors="none", label="low confidence")
    ax.errorbar(
        table.loc[grade_b, sigma_col],
        table.loc[grade_b, mbh_col],
        xerr=table.loc[grade_b, err_col],
        fmt="o",
        ms=3.8,
        color="#2f7bbd",
        ecolor="#a9c7e8",
        elinewidth=0.42,
        alpha=0.76,
        label="intermediate confidence",
    )
    ax.errorbar(
        table.loc[golden, sigma_col],
        table.loc[golden, mbh_col],
        xerr=table.loc[golden, err_col],
        fmt="*",
        ms=8.5,
        color="#d28b00",
        ecolor="#e8c26a",
        elinewidth=0.48,
        mec="0.20",
        mew=0.45,
        label="high confidence",
    )
    med = bin_medians(table, finite, sigma_col, mbh_col)
    if len(med):
        ax.errorbar(
            med["sigma"],
            med["log_mbh"],
            yerr=med["log_mbh_err"],
            fmt="s",
            ms=5.0,
            color="black",
            ecolor="black",
            elinewidth=0.8,
            label="median bins",
        )
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    format_sigma_axis(ax)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"Inferred $\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"$\log(M_{\rm BH}/M_\odot)$")
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.set_title(r"(a) $M_{\rm BH}\!-\!\sigma_\star$ relation", fontsize=10.5)

    ax = axes[0, 1]
    residual = table[mbh_col] - local_relation(np.log10(table[sigma_col].clip(lower=1e-6) / 200.0), 8.13, 4.02)
    sc = ax.scatter(
        table.loc[finite, sigma_col],
        residual.loc[finite],
        c=table.loc[finite, "pred_host_fraction"],
        s=np.where(golden.loc[finite], 34, np.where(science.loc[finite], 18, 12)),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.76,
        edgecolors="none",
    )
    ax.axhspan(-0.5, 0.5, color="0.91", zorder=-1)
    ax.axhline(0, color="0.15", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    format_sigma_axis(ax)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel(r"Inferred $\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"Residual from Tremaine+02 [dex]")
    ax.set_title("(b) Residuals", fontsize=10.5)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Predicted host fraction")

    ax = axes[1, 0]
    bins = np.linspace(5.0, 9.4, 30)
    ax.hist(table.loc[low_conf, mbh_col], bins=bins, color="0.75", alpha=0.55, label="low confidence")
    ax.hist(table.loc[grade_b, mbh_col], bins=bins, histtype="step", lw=1.7, color="#2f7bbd", label="intermediate confidence")
    ax.hist(table.loc[golden, mbh_col], bins=bins, histtype="step", lw=1.8, color="#d28b00", label="high confidence")
    ax.set_xlabel(r"$\log(M_{\rm BH}/M_\odot)$")
    ax.set_ylabel("Number")
    ax.set_title("(c) Virial-mass distribution", fontsize=10.5)

    ax = axes[1, 1]
    ax.scatter(table.loc[low_conf, "median_snr"], table.loc[low_conf, err_col], s=14, color="0.70", alpha=0.45, edgecolors="none", label="low confidence")
    ax.scatter(table.loc[grade_b, "median_snr"], table.loc[grade_b, err_col], s=18, color="#2f7bbd", alpha=0.78, edgecolors="none", label="intermediate confidence")
    ax.scatter(table.loc[golden, "median_snr"], table.loc[golden, err_col], s=42, marker="*", color="#d28b00", edgecolors="0.20", linewidths=0.35, label="high confidence")
    ax.axhline(50, color="#d28b00", lw=1.0, ls="--", alpha=0.85)
    ax.axhline(75, color="#2f7bbd", lw=1.0, ls=":", alpha=0.85)
    ax.set_xlim(0, max(75.0, float(np.nanpercentile(table.loc[finite, "median_snr"], 98)) * 1.08))
    ax.set_ylim(0, max(150.0, float(np.nanpercentile(table.loc[finite, err_col], 98)) * 1.06))
    ax.set_xlabel("SDSS spectrum S/N")
    ax.set_ylabel(r"Calibrated $\sigma_\star$ uncertainty [km s$^{-1}$]")
    ax.set_title("(d) Measurement precision", fontsize=10.5)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.925),
        ncol=4,
        frameon=False,
        fontsize=9.0,
        handlelength=2.0,
        columnspacing=1.3,
    )
    fig.savefig(output_pdf)
    plt.close(fig)

    summary = {
        "input_csv": str(input_csv),
        "output_pdf": str(output_pdf),
        "quality_definition": {
            "high_confidence": "golden_sample OR high_confidence from infer_sdss_quasar_sigma.py",
            "confidence_selected": "internal science_ready flag from infer_sdss_quasar_sigma.py",
            "low_confidence": "finite spectra not marked confidence-selected",
        },
        "relations": {
            "tremaine2002": "log(M_BH/Msun)=8.13+4.02*log10(sigma/200 km/s)",
            "gultekin2009": "log(M_BH/Msun)=8.12+4.24*log10(sigma/200 km/s)",
        },
        "samples": summaries,
        "odr_fit": {
            "mask": args.odr_mask,
            "virial_mass_error_dex": float(args.virial_mass_error_dex),
            **odr_result,
        },
        "odr_fit_native_golden_for_diagnostic": {
            "virial_mass_error_dex": float(args.virial_mass_error_dex),
            **odr_golden,
        },
        "bin_medians_all": med.to_dict(orient="records") if len(med) else [],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_pdf}")
    print(f"Saved {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
