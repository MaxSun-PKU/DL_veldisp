"""Export the small subset of a mock HDF5 needed for model inference.

The full training data can be tens of gigabytes. Real-spectrum inference only
needs the wavelength grid, SPS template basis, Fe II basis, and a few model
configuration values. This script writes those items to a compact HDF5 file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "wave_grid",
    "template_library",
    "sps_logzsol",
    "sps_age_gyr",
)

OPTIONAL_DATASETS = (
    "feii_template_bank",
    "park_fe_template",
)


def export_inference_assets(source_path: Path, output_path: Path, overwrite: bool = False) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
            missing = [name for name in REQUIRED_DATASETS if name not in source]
            if missing:
                raise KeyError(f"Training HDF5 is missing required datasets: {missing}")

            for name in REQUIRED_DATASETS + OPTIONAL_DATASETS:
                if name in source:
                    source.copy(name, output)

            config_json = source.attrs.get("config_json", "{}")
            config = json.loads(config_json)
            output.attrs["config_json"] = json.dumps(config, sort_keys=True)
            output.attrs["description"] = "Compact assets for HybridAGNNet inference; no mock spectra or labels."
            output.attrs["asset_only"] = True
            output.attrs["source_description"] = source.attrs.get("description", "")

            if "sigma_inst_kms" in source:
                sigma_inst = source["sigma_inst_kms"][:]
                valid = sigma_inst[np.isfinite(sigma_inst) & (sigma_inst > 0)]
                value = float(np.median(valid)) if valid.size else float(config.get("sdss_sigma_inst", 70.0))
            else:
                value = float(config.get("sdss_sigma_inst", 70.0))
            output.create_dataset("sigma_inst_kms", data=np.asarray([value], dtype=np.float32))

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"Wrote {output_path} ({size_mb:.1f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Full mock-training HDF5.")
    parser.add_argument("--output", required=True, type=Path, help="Compact inference-assets HDF5.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_inference_assets(args.source, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
