from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_TAG = os.environ.get(
    "DL_VELDISP_RUN_TAG",
    "sdss_greene2006_science_run_v12_native_res_risk_lineshape_aug",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the ApJ-style Greene & Ho external-validation figure.")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-pdf", default=None)
    return parser.parse_args()


def finite_metric(df: pd.DataFrame, pred_col: str = "recommended_sigma_kms") -> dict[str, float]:
    sub = df[["sigma_lit_kms", pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan}
    delta = sub[pred_col] - sub["sigma_lit_kms"]
    corr = np.corrcoef(sub["sigma_lit_kms"], sub[pred_col])[0, 1] if len(sub) > 1 else np.nan
    return {
        "n": int(len(sub)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "corr": float(corr),
    }


def setup_apj_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def draw(run_tag: str = DEFAULT_RUN_TAG, input_csv: str | None = None, output_pdf: str | None = None) -> None:
    setup_apj_style()
    csv_path = Path(input_csv) if input_csv else ROOT / "figures" / f"{run_tag}_greene2006_external_validation.csv"
    pdf_path = Path(output_pdf) if output_pdf else ROOT / "figures" / f"{run_tag}_greene2006_external_validation_apj.pdf"
    df = pd.read_csv(csv_path)
    pred_col = "recommended_sigma_kms" if "recommended_sigma_kms" in df else "pred_sigma_kms"
    err_col = "recommended_sigma_err_kms" if "recommended_sigma_err_kms" in df else "pred_sigma_err_kms"
    measured = df["sigma_lit_kms"].notna() & df[pred_col].notna()
    plot_df = df.loc[measured].copy()
    plot_df["delta"] = plot_df[pred_col] - plot_df["sigma_lit_kms"]

    high_conf = (
        plot_df.get("golden_sample", False).astype(bool)
        | plot_df.get("publication_ready", False).astype(bool)
        | (plot_df.get("measurement_tier", "") == "gold")
        | plot_df.get("high_confidence", False).astype(bool)
    )
    # Triangles denote spectra below the adopted SDSS resolution threshold.
    # The remaining points follow the three confidence levels defined in the paper.
    resolution_limited = plot_df.get("near_resolution_limit", False).astype(bool) & ~high_conf
    tier = plot_df.get("measurement_tier", "").astype(str)
    intermediate_conf = (tier == "silver") & ~resolution_limited & ~high_conf
    low_conf = ~(high_conf | intermediate_conf | resolution_limited)

    all_m = finite_metric(plot_df, pred_col)
    hc_m = finite_metric(plot_df[high_conf], pred_col)

    c_diag = "0.72"
    c_science = "0.42"
    c_intermediate = "#4c91c7"
    c_high = "#d99a21"
    c_line = "0.18"

    fig, (ax, axr) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.26},
    )

    def errplot(axis, mask, color, alpha=0.72, lw=0.75, cap=1.2):
        if not np.any(mask):
            return
        xerr = plot_df.loc[mask, "sigma_lit_err_kms"] if "sigma_lit_err_kms" in plot_df else None
        yerr = plot_df.loc[mask, err_col] if err_col in plot_df else None
        axis.errorbar(
            plot_df.loc[mask, "sigma_lit_kms"],
            plot_df.loc[mask, pred_col],
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=lw,
            capsize=cap,
            capthick=lw,
            alpha=alpha,
            zorder=1,
        )

    def errplot_resid(axis, mask, color, alpha=0.72, lw=0.75, cap=1.2):
        if not np.any(mask):
            return
        xerr = plot_df.loc[mask, "sigma_lit_err_kms"] if "sigma_lit_err_kms" in plot_df else None
        yerr = plot_df.loc[mask, err_col] if err_col in plot_df else None
        axis.errorbar(
            plot_df.loc[mask, "sigma_lit_kms"],
            plot_df.loc[mask, "delta"],
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=lw,
            capsize=cap,
            capthick=lw,
            alpha=alpha,
            zorder=1,
        )

    # Panel A: direct comparison.
    for mask, color, alpha, lw, cap in [
        (resolution_limited, "0.80", 0.55, 0.65, 0.0),
        (low_conf, "0.76", 0.62, 0.70, 0.0),
        (intermediate_conf, "#8bb6d5", 0.72, 0.75, 0.0),
        (high_conf, "#d8a03b", 0.95, 0.9, 1.6),
    ]:
        errplot(ax, mask, color, alpha=alpha, lw=lw, cap=cap)

    ax.scatter(
        plot_df.loc[resolution_limited, "sigma_lit_kms"],
        plot_df.loc[resolution_limited, pred_col],
        s=27,
        marker="^",
        facecolor="white",
        edgecolor=c_diag,
        linewidth=0.8,
        alpha=0.95,
        zorder=2,
    )
    ax.scatter(
        plot_df.loc[low_conf, "sigma_lit_kms"],
        plot_df.loc[low_conf, pred_col],
        s=25,
        marker="o",
        facecolor=c_science,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=3,
    )
    ax.scatter(
        plot_df.loc[intermediate_conf, "sigma_lit_kms"],
        plot_df.loc[intermediate_conf, pred_col],
        s=27,
        marker="o",
        facecolor=c_intermediate,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.90,
        zorder=3,
    )
    ax.scatter(
        plot_df.loc[high_conf, "sigma_lit_kms"],
        plot_df.loc[high_conf, pred_col],
        s=58,
        marker="*",
        facecolor=c_high,
        edgecolor="0.15",
        linewidth=0.45,
        alpha=1.0,
        zorder=4,
    )

    one_to_one, = ax.plot([20, 350], [20, 350], color=c_line, lw=1.0, ls="-", label="1:1 relation", zorder=0)
    ax.set_xlim(20, 350)
    yerr_vals = plot_df[err_col].to_numpy(float) if err_col in plot_df else np.zeros(len(plot_df))
    ytop = max(350, float(np.nanmax(plot_df[pred_col].to_numpy(float) + yerr_vals)) * 1.04)
    ytop = min(420, math.ceil(ytop / 50) * 50)
    ax.set_ylim(0, ytop)
    ax.set_xlabel(r"Greene & Ho (2006) $\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"Pipeline $\sigma_\star$ [km s$^{-1}$]")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9.5)
    handles = [
        one_to_one,
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="white", markeredgecolor=c_diag, markersize=5.3, label="below SDSS res."),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=c_science, markeredgecolor="white", markersize=5.2, label="low confidence"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=c_intermediate, markeredgecolor="white", markersize=5.2, label="intermediate confidence"),
    ]
    handles.append(
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor=c_high, markeredgecolor="0.15", markersize=8.2, label="high confidence")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.015),
        ncol=5,
        frameon=False,
        borderpad=0.0,
        handlelength=1.4,
        handletextpad=0.45,
        labelspacing=0.35,
    )

    # Panel B: residuals.
    for mask, color, alpha, lw, cap in [
        (resolution_limited, "0.82", 0.50, 0.65, 0.0),
        (low_conf, "0.78", 0.60, 0.70, 0.0),
        (intermediate_conf, "#8bb6d5", 0.72, 0.75, 0.0),
        (high_conf, "#d8a03b", 0.95, 0.9, 1.6),
    ]:
        errplot_resid(axr, mask, color, alpha=alpha, lw=lw, cap=cap)
    axr.scatter(plot_df.loc[resolution_limited, "sigma_lit_kms"], plot_df.loc[resolution_limited, "delta"], s=27, marker="^", facecolor="white", edgecolor=c_diag, linewidth=0.8, alpha=0.95, zorder=2)
    axr.scatter(plot_df.loc[low_conf, "sigma_lit_kms"], plot_df.loc[low_conf, "delta"], s=25, marker="o", facecolor=c_science, edgecolor="white", linewidth=0.35, alpha=0.82, zorder=3)
    axr.scatter(plot_df.loc[intermediate_conf, "sigma_lit_kms"], plot_df.loc[intermediate_conf, "delta"], s=27, marker="o", facecolor=c_intermediate, edgecolor="white", linewidth=0.35, alpha=0.90, zorder=3)
    axr.scatter(plot_df.loc[high_conf, "sigma_lit_kms"], plot_df.loc[high_conf, "delta"], s=58, marker="*", facecolor=c_high, edgecolor="0.15", linewidth=0.45, alpha=1.0, zorder=4)
    axr.axhline(0.0, color=c_line, lw=0.9, zorder=0)
    axr.set_xlim(20, 350)
    residual_extent = np.nanmax(np.abs(plot_df["delta"].to_numpy(float)) + np.nan_to_num(yerr_vals, nan=0.0))
    rlim = max(140, min(260, math.ceil(float(residual_extent) * 1.04 / 25) * 25))
    axr.set_ylim(-rlim, rlim)
    axr.set_xlabel(r"Greene & Ho (2006) $\sigma_\star$ [km s$^{-1}$]")
    axr.set_ylabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    axr.text(0.03, 0.97, "(b)", transform=axr.transAxes, ha="left", va="top", fontsize=9.5)
    axr.text(
        0.965,
        0.055,
        rf"All-sample bias={all_m['bias']:.0f} km s$^{{-1}}$" "\n"
        rf"High-confidence bias={hc_m['bias']:.0f} km s$^{{-1}}$",
        transform=axr.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="0.15",
    )

    for axis in (ax, axr):
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.yaxis.set_minor_locator(AutoMinorLocator(5))
        axis.tick_params(which="major", length=4.0, width=0.8)
        axis.tick_params(which="minor", length=2.0, width=0.6)

    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved: {pdf_path}")
    print(f"all: n={all_m['n']} rmse={all_m['rmse']:.2f} bias={all_m['bias']:.2f} corr={all_m['corr']:.3f}")
    print(f"high-confidence: n={hc_m['n']} rmse={hc_m['rmse']:.2f} bias={hc_m['bias']:.2f} corr={hc_m['corr']:.3f}")


if __name__ == "__main__":
    args = parse_args()
    draw(run_tag=args.run_tag, input_csv=args.input_csv, output_pdf=args.output_pdf)
