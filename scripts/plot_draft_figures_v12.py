from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = "sdss_greene2006_science_run_v12_native_res_risk_lineshape_aug"
H5_PATH = ROOT / "mock_data" / "mock_sdss_greene2006_science_run_v12_native_res_risk_220k.h5"
HISTORY_JSON = ROOT / "models" / f"{RUN_TAG}_training_history.json"
DRAFT_FIG_DIR = ROOT / "draft" / "DL_veldisp" / "figures"
ROOT_FIG_DIR = ROOT / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.85,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def save_figure(fig: plt.Figure, name: str, *, also_root_name: str | None = None) -> None:
    DRAFT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = DRAFT_FIG_DIR / name
    fig.savefig(out)
    print(f"saved: {out}")
    if also_root_name:
        ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        root_out = ROOT_FIG_DIR / also_root_name
        fig.savefig(root_out)
        print(f"saved: {root_out}")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, *, fc, ec="#2f455c", bold=False, fontsize=8.2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        linewidth=1.05,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        linespacing=1.03,
        zorder=4,
    )
    return box


def arrow(ax, start, end, *, color="#6b7785", rad=0.0, linestyle="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=0.95,
        linestyle=linestyle,
        color=color,
        shrinkA=2.2,
        shrinkB=2.2,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(patch)


def orthogonal_arrow(ax, points, *, color="#6b7785"):
    """Draw a right-angle connector whose arrowhead appears only at the target."""
    if len(points) < 2:
        raise ValueError("An orthogonal connector needs at least two points")
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(
            xs,
            ys,
            color=color,
            lw=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
    arrow(ax, points[-2], points[-1], color=color)


def connector_bus(ax, source, bus_x, targets, *, color="#6b7785"):
    """Connect one source to several targets through a vertical routing lane."""
    target_y = [point[1] for point in targets]
    arrow(ax, source, (bus_x, source[1]), color=color)
    ax.plot(
        [bus_x, bus_x],
        [source[1], max([source[1], *target_y])],
        color=color,
        lw=0.95,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    for target in targets:
        arrow(ax, (bus_x, target[1]), target, color=color)


def plot_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = {
        "mock": "#edf5ea",
        "input": "#e8f0f7",
        "model": "#fff3cf",
        "output": "#f5eaf3",
        "training": "#f0f2f4",
    }
    process = "#667684"
    training = "#8a6f62"

    ax.text(0.02, 0.94, "Training with mock spectra", ha="left", va="center", fontsize=9.4, fontweight="bold")
    top_y, top_h = 0.69, 0.15
    top_boxes = [
        (0.02, 0.20, "mock generation\nstellar host + AGN\nSDSS observing effects", cols["mock"]),
        (0.28, 0.24, "shared preprocessing and inputs\nrest-frame grid + normalization\nflux errors + masks\nresolution + regions + $L_{5100}$", cols["input"]),
        (0.59, 0.16, "neural network\nFigure 2", cols["model"]),
        (0.81, 0.17, "predictions and\nhost + AGN\nreconstruction", cols["output"]),
    ]
    for x, width, label, color in top_boxes:
        rounded_box(ax, (x, top_y), width, top_h, label, fc=color, bold=("network" in label))
    for first, second in zip(top_boxes[:-1], top_boxes[1:]):
        arrow(
            ax,
            (first[0] + first[1], top_y + top_h / 2),
            (second[0], top_y + top_h / 2),
            color=process,
        )

    rounded_box(
        ax,
        (0.28, 0.46),
        0.30,
        0.11,
        "validation mocks\nadopted epoch + uncertainty calibration",
        fc=cols["training"],
        ec=process,
        fontsize=7.8,
    )
    rounded_box(
        ax,
        (0.72, 0.46),
        0.26,
        0.11,
        "mock targets + outputs $\\rightarrow \\mathcal{L}_{\\rm train}$\nbackpropagation updates weights",
        fc=cols["training"],
        ec=training,
        fontsize=7.55,
    )
    arrow(ax, (0.895, top_y), (0.895, 0.57), color=training)
    arrow(ax, (0.74, 0.52), (0.67, top_y), color=training, rad=-0.25, linestyle="--")
    arrow(ax, (0.64, top_y), (0.56, 0.57), color=process, rad=0.18)

    ax.text(0.02, 0.36, "Application to observed spectra", ha="left", va="center", fontsize=9.4, fontweight="bold")
    bottom_y, bottom_h = 0.10, 0.14
    bottom_boxes = [
        (0.02, 0.20, "observed SDSS\nspectrum", cols["input"]),
        (0.28, 0.24, "same preprocessing\nand input construction", cols["input"]),
        (0.59, 0.16, "fixed trained model\nforward pass", cols["model"]),
        (0.81, 0.17, "$\\sigma_\\star$, uncertainty,\nconfidence category,\nand reconstruction", cols["output"]),
    ]
    for x, width, label, color in bottom_boxes:
        rounded_box(ax, (x, bottom_y), width, bottom_h, label, fc=color, bold=("fixed" in label))
    for first, second in zip(bottom_boxes[:-1], bottom_boxes[1:]):
        arrow(
            ax,
            (first[0] + first[1], bottom_y + bottom_h / 2),
            (second[0], bottom_y + bottom_h / 2),
            color=process,
        )
    arrow(ax, (0.54, 0.46), (0.64, bottom_y + bottom_h), color=process, rad=-0.08)
    ax.text(
        0.575,
        0.35,
        "adopt epoch and calibrate",
        fontsize=7.1,
        color="#4f5d6b",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        zorder=6,
    )

    save_figure(fig, "dl_pipeline_overview_v12.pdf")


def plot_network_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = {
        "input": "#e8f0f7",
        "encoder": "#fff3cf",
        "feature": "#edf5ea",
        "head": "#f5eaf3",
    }
    process = "#667684"

    ax.text(0.105, 0.94, "Inputs", ha="center", va="center", fontsize=9.4, fontweight="bold")
    ax.text(0.385, 0.94, "Feature extraction", ha="center", va="center", fontsize=9.4, fontweight="bold")
    ax.text(0.615, 0.94, "Feature vectors", ha="center", va="center", fontsize=9.4, fontweight="bold")
    ax.text(0.855, 0.94, "Output layers", ha="center", va="center", fontsize=9.4, fontweight="bold")

    input_x, input_y, input_w, input_h = 0.02, 0.30, 0.17, 0.42
    rounded_box(
        ax,
        (input_x, input_y),
        input_w,
        input_h,
        "wavelength-aligned\ninput arrays (10)\n\nnormalized flux\nflux-error weight + masks\nwindow-normalized flux\ninstrumental resolution\nregion indicators + $L_{5100}$",
        fc=cols["input"],
        fontsize=7.55,
    )

    enc_x, enc_w, enc_h = 0.25, 0.28, 0.18
    encoders = [
        (0.69, "full-spectrum encoder\nmulti-scale 1D convolutions\nkernels: 3, 11, 31\nattention + average pooling"),
        (0.41, "absorption-window encoder\nusable CaT, Fe/metal,\nand Ca II H+K wavelengths\n1D convolution kernels: 9, 15, 31\nmean/max/std pooling"),
        (0.13, "line-shape encoder\nflux + first/second differences\n1D convolution kernels: 9, 15, 31\nmean/max/std pooling"),
    ]
    for y, label in encoders:
        rounded_box(ax, (enc_x, y), enc_w, enc_h, label, fc=cols["encoder"], fontsize=7.15)

    centers = [y + enc_h / 2 for y, _ in encoders]
    bus_x = 0.22
    arrow(ax, (input_x + input_w, input_y + input_h / 2), (bus_x, input_y + input_h / 2), color=process)
    ax.plot([bus_x, bus_x], [min(centers), max(centers)], color=process, lw=0.95, zorder=1)
    for center in centers:
        arrow(ax, (bus_x, center), (enc_x, center), color=process)

    feat_x, feat_w, feat_h = 0.57, 0.10, 0.10
    features = [
        (0.73, "$\\mathbf{f}_{\\rm full}$\n128"),
        (0.45, "$\\mathbf{f}_{\\rm win}$\n192"),
        (0.17, "$\\mathbf{f}_{\\rm shape}$\n192"),
    ]
    for (enc_y, _), (feat_y, label) in zip(encoders, features):
        rounded_box(ax, (feat_x, feat_y), feat_w, feat_h, label, fc=cols["feature"], bold=True, fontsize=7.8)
        arrow(
            ax,
            (enc_x + enc_w, enc_y + enc_h / 2),
            (feat_x, feat_y + feat_h / 2),
            color=process,
        )

    head_x, head_w = 0.74, 0.24
    rounded_box(
        ax,
        (head_x, 0.47),
        head_w,
        0.34,
        "output modules\n\nkinematics\nbase $\\sigma_\\star$ + window/line\ncorrections + region weighting\n$\\rightarrow \\hat{\\sigma}_\\star$\n\nother outputs\nlog variance, host fraction,\nstellar-population weights,\nand seven-interval classifier",
        fc=cols["head"],
        fontsize=7.15,
    )

    merge_x = 0.705
    feature_centers = [y + feat_h / 2 for y, _ in features]
    for center in feature_centers:
        arrow(ax, (feat_x + feat_w, center), (merge_x, center), color=process)
    ax.plot([merge_x, merge_x], [min(feature_centers), max(feature_centers)], color=process, lw=0.95, zorder=1)
    arrow(ax, (merge_x, 0.64), (head_x, 0.64), color=process)

    rounded_box(
        ax,
        (head_x, 0.13),
        head_w,
        0.20,
        "physical decoders\nstellar-population weights + $\\hat{\\sigma}_\\star$\n$\\rightarrow$ broadened host\nfull features $\\rightarrow$ AGN\nhost fraction $\\rightarrow$ reconstruction",
        fc=cols["head"],
        fontsize=7.15,
    )
    arrow(ax, (head_x + head_w / 2, 0.47), (head_x + head_w / 2, 0.33), color=process)

    save_figure(fig, "dl_network_architecture_v12.pdf")


def finite_values(handle: h5py.File, key: str, sample: np.ndarray | None = None) -> np.ndarray:
    values = handle[key][sample] if sample is not None else handle[key][:]
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def histogram_panel(ax, values, bins, title, xlabel, color, *, logx=False, logy=False, vlines=None):
    ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
    if vlines:
        for x, kwargs in vlines:
            ax.axvline(x, **kwargs)
    ax.set_title(title, loc="left", pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    if not logy:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(which="major", length=3.5, width=0.8)
    ax.tick_params(which="minor", length=2.0, width=0.6)


def plot_parameter_coverage() -> None:
    with h5py.File(H5_PATH, "r") as handle:
        sample = None
        fields = {
            "sigma": finite_values(handle, "sigma_star", sample),
            "host": finite_values(handle, "host_fraction", sample),
            "snr": finite_values(handle, "snr", sample),
            "redshift": finite_values(handle, "redshift", sample),
            "age": finite_values(handle, "mass_weighted_age_gyr", sample),
            "metal": finite_values(handle, "mass_weighted_logzsol", sample),
            "feii": finite_values(handle, "r_feii", sample),
            "inst": finite_values(handle, "sampled_sdss_inst_sigma_kms", sample),
            "coverage": finite_values(handle, "sigma_mask_fraction", sample),
        }

    fig, axes = plt.subplots(3, 3, figsize=(7.35, 6.2), constrained_layout=True)
    palette = ["#486a9a", "#6b9e78", "#b9854b", "#7b6ca8", "#c45f5f", "#647d8e", "#ba6d9a", "#8f8f55", "#4c968f"]
    science_edges = [30, 60, 80, 120, 180, 250, 400, 700]
    panels = [
        (fields["sigma"], science_edges, r"(a) input $\sigma_\star$", r"$\sigma_\star$ [km s$^{-1}$]", palette[0], False, False),
        (fields["host"], np.linspace(0, 1, 36), r"(b) host fraction", r"$f_{\rm host}$", palette[1], False, False),
        (fields["snr"], np.linspace(5, 80, 36), r"(c) median S/N", "S/N", palette[2], False, False),
        (fields["redshift"], np.linspace(0.01, 0.9, 36), r"(d) redshift", "$z$", palette[3], False, False),
        (fields["age"], np.linspace(0, 13.5, 36), r"(e) stellar age", "mass-weighted age [Gyr]", palette[4], False, False),
        (fields["metal"], np.linspace(-2.0, 0.5, 36), r"(f) metallicity", r"$\log Z/Z_\odot$", palette[5], False, False),
        (fields["feii"], np.linspace(np.nanpercentile(fields["feii"], 0.5), np.nanpercentile(fields["feii"], 99.5), 36), r"(g) Fe II template amplitude", r"$A_{\rm FeII}$", palette[6], False, False),
        (fields["inst"], np.linspace(50, 100, 32), r"(h) SDSS resolution", r"$\sigma_{\rm inst}$ [km s$^{-1}$]", palette[7], False, False),
        (fields["coverage"], np.linspace(0, max(0.09, np.nanpercentile(fields["coverage"], 99.5)), 34), r"(i) usable stellar-window coverage", r"$f_{\rm win}=N_{\rm valid,win}/N_{\rm grid}$", palette[8], False, False),
    ]
    for ax, (values, bins, title, xlabel, color, logx, logy) in zip(axes.flat, panels):
        vlines = None
        if "input" in title:
            vlines = [(edge, {"color": "0.55", "lw": 0.6, "alpha": 0.55}) for edge in science_edges[1:-1]]
        histogram_panel(ax, values, bins, title, xlabel, color, logx=logx, logy=logy, vlines=vlines)

    save_figure(
        fig,
        "parameter_coverage_v12.pdf",
        also_root_name=f"{RUN_TAG}_parameter_coverage_apj.pdf",
    )


def padded_ylim(values, *, frac=0.18, floor=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo = min(float(np.nanmin(values)), floor)
    hi = float(np.nanmax(values))
    pad = max((hi - lo) * frac, 1e-3)
    return lo - pad * 0.12, hi + pad


def plot_training_history() -> None:
    payload = json.loads(HISTORY_JSON.read_text())
    history = payload["history"]
    epochs = np.asarray([row["epoch"] for row in history], dtype=float)
    best_epoch = int(payload.get("best_epoch", 20))
    full = np.asarray([row.get("validation_mode", "full") == "full" or row["epoch"] <= 3 for row in history], dtype=bool)

    # Historical runs stored the common train/validation loss under *_total.
    # New runs also record the complete optimization loss separately.
    train_main = np.asarray([row.get("train_main_loss", row["train_total"]) for row in history], dtype=float)
    val_main = np.asarray([row.get("val_main_loss", row["val_total"]) for row in history], dtype=float)
    train_sigma = np.asarray([row["train_sigma"] for row in history], dtype=float)
    val_sigma = np.asarray([row["val_sigma_loss"] for row in history], dtype=float)
    train_recon = np.asarray([row["train_recon"] for row in history], dtype=float)
    val_recon = np.asarray([row["val_recon"] for row in history], dtype=float)
    rmse = np.asarray([row["val_sigma_rmse"] for row in history], dtype=float)
    bin_rmse = np.asarray([row["val_science_bin_weighted_rmse"] for row in history], dtype=float)
    lowmid_rmse = np.asarray([row["val_science_low_mid_rmse"] for row in history], dtype=float)
    score = np.asarray([row["selection_score"] for row in history], dtype=float)
    corr = np.asarray([row["val_sigma_corr"] for row in history], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, train_main, color="0.20", lw=1.5, label="training")
    ax.plot(epochs, val_main, color="#c9433f", lw=1.3, label="validation")
    ax.axvspan(0.5, 3.5, color="0.92", zorder=-1)
    ax.axvline(best_epoch, color="0.15", ls=":", lw=0.9)
    ax.set_ylabel("Main loss")
    ax.set_ylim(*padded_ylim(np.r_[train_main, val_main], frac=0.25))
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.92, "(a)", transform=ax.transAxes)
    ax.text(0.60, 0.18, f"adopted\nepoch {best_epoch}", transform=ax.transAxes, fontsize=7.4)

    ax = axes[0, 1]
    ax.plot(epochs, train_sigma, color="#2f6da3", lw=1.4, label=r"train $\sigma_\star$")
    ax.plot(epochs, val_sigma, color="#8a67b8", lw=1.4, label=r"val. $\sigma_\star$")
    ax.plot(epochs, train_recon, color="#d7982d", lw=1.25, label="train recon.")
    ax.plot(epochs, val_recon, color="#b9652a", lw=1.25, label="val. recon.")
    ax.axvspan(0.5, 3.5, color="0.92", zorder=-1)
    ax.axvline(best_epoch, color="0.15", ls=":", lw=0.9)
    ax.set_ylabel("Loss component")
    ax.set_ylim(*padded_ylim(np.r_[train_sigma, val_sigma, train_recon, val_recon], frac=0.24))
    ax.legend(frameon=False, loc="upper right", ncol=1)
    ax.text(0.04, 0.92, "(b)", transform=ax.transAxes)

    ax = axes[1, 0]
    ax.plot(epochs, rmse, color="#2f6da3", lw=1.4, label="RMSE")
    ax.plot(epochs, bin_rmse, color="#62a45d", lw=1.35, label="equal-bin RMSE")
    ax.plot(epochs, lowmid_rmse, color="#e69f00", lw=1.35, label=r"$\sigma_\star<250$ RMSE")
    ax.scatter(epochs[full], rmse[full], s=13, color="#2f6da3", zorder=3)
    ax.axvspan(0.5, 3.5, color="0.92", zorder=-1)
    ax.axvline(best_epoch, color="0.15", ls=":", lw=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation metric [km s$^{-1}$]")
    ax.set_ylim(40, max(122, float(np.nanmax(rmse)) + 10))
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.92, "(c)", transform=ax.transAxes)

    ax = axes[1, 1]
    ax.plot(epochs, score, color="0.25", lw=1.45, label="validation score")
    ax2 = ax.twinx()
    ax2.plot(epochs, corr, color="#c9433f", lw=1.35, label="correlation")
    ax.axvspan(0.5, 3.5, color="0.92", zorder=-1)
    ax.axvline(best_epoch, color="0.15", ls=":", lw=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax2.set_ylabel(r"Validation $r$")
    ax.set_ylim(*padded_ylim(score, frac=0.22, floor=145))
    ax2.set_ylim(min(0.74, float(np.nanmin(corr)) - 0.02), max(0.89, float(np.nanmax(corr)) + 0.02))
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    ax.text(0.04, 0.92, "(d)", transform=ax.transAxes)

    for ax in axes.flat:
        ax.set_xlim(0.5, max(epochs) + 0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.tick_params(which="major", length=3.5, width=0.8)
        ax.tick_params(which="minor", length=2.0, width=0.6)

    save_figure(fig, "training_loss_trends_v12.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate APJ draft method and training figures.")
    parser.add_argument(
        "--which",
        choices=("all", "pipeline", "network", "coverage", "history"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    if args.which in {"all", "pipeline"}:
        plot_pipeline_overview()
    if args.which in {"all", "network"}:
        plot_network_architecture()
    if args.which in {"all", "coverage"}:
        plot_parameter_coverage()
    if args.which in {"all", "history"}:
        plot_training_history()


if __name__ == "__main__":
    main()
