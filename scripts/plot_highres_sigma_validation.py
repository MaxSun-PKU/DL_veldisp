#!/usr/bin/env python
"""Plot the high-resolution sigma truth validation for the v12 SDSS model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def metrics(table: pd.DataFrame, pred_col: str = "pred_sigma_kms") -> dict[str, float | int]:
    finite = (
        np.isfinite(table[pred_col])
        & np.isfinite(table["sigma_truth_kms"])
        & (table[pred_col] > 0)
        & (table["sigma_truth_kms"] > 0)
    )
    sub = table.loc[finite].copy()
    if len(sub) == 0:
        return {"n": 0}
    delta = sub[pred_col].to_numpy(float) - sub["sigma_truth_kms"].to_numpy(float)
    truth = sub["sigma_truth_kms"].to_numpy(float)
    pred = sub[pred_col].to_numpy(float)
    corr = float(np.corrcoef(truth, pred)[0, 1]) if len(sub) > 1 and np.nanstd(pred) > 0 and np.nanstd(truth) > 0 else np.nan
    return {
        "n": int(len(sub)),
        "rmse_kms": float(np.sqrt(np.nanmean(delta**2))),
        "mae_kms": float(np.nanmean(np.abs(delta))),
        "bias_kms": float(np.nanmean(delta)),
        "median_delta_kms": float(np.nanmedian(delta)),
        "corr": corr,
        "truth_min_kms": float(np.nanmin(truth)),
        "truth_max_kms": float(np.nanmax(truth)),
        "pred_min_kms": float(np.nanmin(pred)),
        "pred_max_kms": float(np.nanmax(pred)),
    }


def prepare_table(manifest_csv: Path, inference_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    inference = pd.read_csv(inference_csv)
    keep = [
        "sdss_name",
        "literature_id",
        "candidate_group",
        "validation_priority",
        "validation_set",
        "z_lit",
        "sigma_truth_kms",
        "sigma_truth_err_kms",
        "truth_instrument",
        "truth_aperture",
        "n_sigma_windows_usable",
        "sdss_plate",
        "sdss_mjd",
        "sdss_fiberid",
        "sdss_sep_arcsec",
        "lamp_esi_note",
    ]
    keep = [col for col in keep if col in manifest.columns]
    merged = inference.merge(manifest[keep], on="sdss_name", how="left", suffixes=("", "_manifest"))
    if "sigma_lit_kms" in merged:
        # The inference script already carries the same truth values as sigma_lit_kms.
        merged["sigma_truth_kms"] = merged["sigma_truth_kms"].fillna(merged["sigma_lit_kms"])
    if "sigma_lit_err_kms" in merged:
        merged["sigma_truth_err_kms"] = merged["sigma_truth_err_kms"].fillna(merged["sigma_lit_err_kms"])
    return merged


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def sample_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    group = table["candidate_group"].fillna("")
    return {
        "Bennert local Keck/LRIS": group.eq("local_bennert2010_keck_lris"),
        "Woo high-z Keck/LRIS": group.eq("highz_woo06_woo08_keck_lris"),
        "Woo LAMP Keck/ESI": group.eq("lamp_woo2010_keck_esi_caveat"),
    }


def plot_validation(main: pd.DataFrame, esi: pd.DataFrame, output_pdf: Path) -> None:
    setup_style()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([main, esi], ignore_index=True, sort=False)
    combined["is_esi_caveat"] = combined["candidate_group"].fillna("").eq("lamp_woo2010_keck_esi_caveat")
    combined["science_ready_bool"] = as_bool(combined.get("science_ready", pd.Series(False, index=combined.index)))
    combined["gold_bool"] = as_bool(combined.get("golden_sample", pd.Series(False, index=combined.index)))

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7.5))
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.08, top=0.78, wspace=0.28, hspace=0.28)
    ax = axes[0, 0]

    styles = {
        "Bennert local Keck/LRIS": dict(marker="o", color="#2c7fb8", label="Bennert local Keck/LRIS"),
        "Woo high-z Keck/LRIS": dict(marker="s", color="#41ab5d", label="Woo high-z Keck/LRIS"),
        "Woo LAMP Keck/ESI": dict(marker="D", color="#d95f0e", label="LAMP Keck/ESI"),
    }
    masks = sample_masks(combined)
    for label, mask in masks.items():
        sub = combined.loc[mask].copy()
        if len(sub) == 0:
            continue
        style = styles[label]
        ax.errorbar(
            sub["sigma_truth_kms"],
            sub["pred_sigma_kms"],
            xerr=sub["sigma_truth_err_kms"],
            yerr=sub["pred_sigma_err_kms"],
            fmt=style["marker"],
            ms=4.3 if "ESI" not in label else 5.0,
            mfc=style["color"],
            mec="white",
            mew=0.35,
            ecolor=style["color"],
            elinewidth=0.55,
            capsize=1.5,
            alpha=0.80 if "ESI" not in label else 0.62,
            label=style["label"],
        )

    science = combined["science_ready_bool"] & ~combined["is_esi_caveat"]
    if science.any():
        ax.scatter(
            combined.loc[science, "sigma_truth_kms"],
            combined.loc[science, "pred_sigma_kms"],
            s=54,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="confidence-selected",
            zorder=4,
        )
    lo, hi = 35.0, 400.0
    ax.plot([lo, hi], [lo, hi], color="0.15", lw=1.1, label="1:1 relation")

    # Preserve visibility of predictions above the requested display range.
    clipped_high = combined["pred_sigma_kms"] > hi
    if clipped_high.any():
        clipped = combined.loc[clipped_high]
        ax.scatter(
            clipped["sigma_truth_kms"],
            np.full(len(clipped), hi - 2.5),
            marker="^",
            s=35,
            color="#41ab5d",
            edgecolors="white",
            linewidths=0.35,
            clip_on=False,
            zorder=5,
        )
        first = clipped.iloc[0]
        ax.annotate(
            rf"{first['literature_id']}: {first['pred_sigma_kms']:.0f} km s$^{{-1}}$",
            xy=(first["sigma_truth_kms"], hi - 2.5),
            xytext=(7, -15),
            textcoords="offset points",
            fontsize=7.2,
            color="0.25",
            ha="left",
            va="top",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\sigma_\star$ measured from high-resolution spectra [km s$^{-1}$]")
    ax.set_ylabel(r"$\sigma_\star$ predicted from SDSS [km s$^{-1}$]")
    ax.set_title("(a) SDSS prediction versus high-resolution measurement")
    sample_handles, sample_labels = ax.get_legend_handles_labels()

    ax = axes[0, 1]
    for label, mask in masks.items():
        sub = combined.loc[mask].copy()
        if len(sub) == 0:
            continue
        style = styles[label]
        delta = sub["pred_sigma_kms"] - sub["sigma_truth_kms"]
        ax.errorbar(
            sub["sigma_truth_kms"],
            delta,
            yerr=sub["pred_sigma_err_kms"],
            fmt=style["marker"],
            ms=4.3,
            mfc=style["color"],
            mec="white",
            mew=0.35,
            ecolor=style["color"],
            elinewidth=0.55,
            capsize=1.5,
            alpha=0.80 if "ESI" not in label else 0.62,
        )
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.axhspan(-50.0, 50.0, color="0.92", zorder=-1)
    ax.set_xlim(lo, 330)
    ax.set_ylim(-230, 455)
    ax.set_xlabel(r"$\sigma_\star$ measured from high-resolution spectra [km s$^{-1}$]")
    ax.set_ylabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.set_title("(b) Residuals")

    ax = axes[1, 0]
    main_delta = main["pred_sigma_kms"] - main["sigma_truth_kms"]
    bennert_delta = main.loc[main["candidate_group"].eq("local_bennert2010_keck_lris"), "pred_sigma_kms"] - main.loc[
        main["candidate_group"].eq("local_bennert2010_keck_lris"), "sigma_truth_kms"
    ]
    woo_delta = main.loc[main["candidate_group"].eq("highz_woo06_woo08_keck_lris"), "pred_sigma_kms"] - main.loc[
        main["candidate_group"].eq("highz_woo06_woo08_keck_lris"), "sigma_truth_kms"
    ]
    bins = np.arange(-240, 461, 35)
    ax.hist(main_delta, bins=bins, histtype="stepfilled", color="0.78", alpha=0.6, label="all LRIS")
    ax.hist(bennert_delta, bins=bins, histtype="step", color="#2c7fb8", lw=1.4, label="nearby LRIS")
    ax.hist(woo_delta, bins=bins, histtype="step", color="#41ab5d", lw=1.4, label="higher-$z$ LRIS")
    ax.axvline(0.0, color="0.15", lw=1.0)
    ax.set_xlabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel("Number")
    ax.set_title("(c) Residual distribution")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    scatter = ax.scatter(
        main["pred_host_fraction"],
        main["pred_sigma_kms"] - main["sigma_truth_kms"],
        c=main["median_snr"],
        s=34,
        cmap="viridis",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.axhspan(-50.0, 50.0, color="0.92", zorder=-1)
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(-230, 455)
    ax.set_xlabel("Predicted host fraction")
    ax.set_ylabel(r"$\Delta\sigma_\star$ [km s$^{-1}$]")
    ax.set_title("(d) Residual versus host fraction")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("SDSS median S/N")

    fig.legend(
        sample_handles,
        sample_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.965),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.3,
    )
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-manifest", required=True)
    parser.add_argument("--main-inference", required=True)
    parser.add_argument("--esi-manifest", required=True)
    parser.add_argument("--esi-inference", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    main_table = prepare_table(Path(args.main_manifest), Path(args.main_inference))
    esi_table = prepare_table(Path(args.esi_manifest), Path(args.esi_inference))
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(output_prefix.with_name(output_prefix.name + "_main_merged.csv"), index=False)
    esi_table.to_csv(output_prefix.with_name(output_prefix.name + "_lamp_esi_merged.csv"), index=False)

    affine_table = main_table.copy()
    affine_table["pred_sigma_kms"] = affine_table["pred_sigma_affine_kms"]
    summary = {
        "central_value": "network sigma without the mock-only linear correction",
        "main_all": metrics(main_table),
        "main_affine_diagnostic": metrics(affine_table),
        "bennert_local_keck_lris": metrics(main_table.loc[main_table["candidate_group"].eq("local_bennert2010_keck_lris")]),
        "woo_highz_keck_lris": metrics(main_table.loc[main_table["candidate_group"].eq("highz_woo06_woo08_keck_lris")]),
        "science_ready_main": metrics(main_table.loc[as_bool(main_table["science_ready"])]),
        "low_confidence_main": metrics(main_table.loc[as_bool(main_table["low_confidence"])]),
        "lamp_esi_caveat": metrics(esi_table),
        "counts": {
            "main_n": int(len(main_table)),
            "main_science_ready_n": int(as_bool(main_table["science_ready"]).sum()),
            "main_golden_n": int(as_bool(main_table["golden_sample"]).sum()),
            "esi_n": int(len(esi_table)),
            "esi_science_ready_n": int(as_bool(esi_table["science_ready"]).sum()),
        },
    }
    with open(output_prefix.with_name(output_prefix.name + "_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_validation(main_table, esi_table, output_prefix.with_name(output_prefix.name + "_comparison.pdf"))
    print(json.dumps(summary, indent=2))
    print("Saved", output_prefix.with_name(output_prefix.name + "_comparison.pdf"))


if __name__ == "__main__":
    main()
