"""
Select and optionally download GH06-like low-redshift SDSS broad-line AGNs.

The default mother catalog is Liu et al. (2019, ApJS, 243, 21; CDS
J/ApJS/243/21), a homogeneous SDSS DR7 broad-line AGN catalog selected mainly
from broad H-alpha.  This is closer to the Greene & Ho (2006; GH06) external
validation sample than a luminous quasar catalog, because it includes many
low-redshift Seyfert-like sources with stronger host-galaxy light.

The default selection prioritizes spectra for which all three GH06 stellar
velocity-dispersion windows are observable in SDSS legacy spectroscopy:
Ca H+K, the 5250-5820 A Fe/metal window, and the Ca II triplet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CDS_LIU2019_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/243/21"
SDSS_DR18_LEGACY_SPEC_BASE = "https://dr18.sdss.org/sas/dr18/spectro/sdss/redux/26"
SDSS_LEGACY_OBS_MIN = 3800.0
SDSS_LEGACY_OBS_MAX = 9200.0

SIGMA_WINDOWS = {
    "cahk": (3910.0, 4060.0),
    "fe": (5250.0, 5820.0),
    "cat": (8480.0, 8700.0),
}

TABLE1_FIELDS = {
    "seq": (0, 5, int),
    "sdss_name": (6, 25, str),
    "ra": (26, 38, float),
    "dec": (39, 53, float),
    "z": (54, 64, float),
    "plate": (65, 69, int),
    "mjd": (70, 75, int),
    "fiber": (76, 79, int),
    "n_spec": (80, 81, int),
}

TABLE2_FIELDS = {
    "seq": (0, 5, int),
    "log_bha": (48, 68, float),
    "e_log_bha": (69, 94, float),
    "fwhm_bha": (95, 115, float),
    "e_fwhm_bha": (116, 146, float),
    "ew_bha": (147, 168, float),
    "e_ew_bha": (169, 193, float),
    "log_bhb": (807, 827, float),
    "e_log_bhb": (828, 855, float),
    "fwhm_bhb": (856, 876, float),
    "e_fwhm_bhb": (877, 909, float),
    "ew_bhb": (910, 932, float),
    "log_oiii5007": (1124, 1144, float),
    "fwhm_oiii5007": (1172, 1191, float),
    "log_feii_4570": (1265, 1285, float),
    "ew_feii_4570": (1312, 1335, float),
    "log_l5100": (1360, 1378, float),
    "e_log_l5100": (1379, 1403, float),
    "multipeak_flag": (1404, 1405, int),
    "log_bh_hb": (1406, 1427, float),
    "log_bh_ha": (1428, 1449, float),
    "log_bh_fid": (1450, 1469, float),
    "log_edd": (1470, 1493, float),
}

TABLE3_FIELDS = {
    "seq": (0, 5, int),
    "psf_g": (46, 64, float),
    "psf_r": (86, 104, float),
    "psf_i": (127, 145, float),
    "petro_g": (249, 267, float),
    "petro_r": (292, 310, float),
    "petro_i": (333, 351, float),
}


@dataclass
class DownloadRecord:
    source_catalog: str
    sdss_name: str
    ra: float
    dec: float
    z: float
    plate: int
    mjd: int
    fiberID: int
    run2d: str
    url: str
    local_path: str
    download_status: str
    coverage_mode: str
    n_sigma_windows_observable: int
    cahk_observable: int
    fe_observable: int
    cat_observable: int
    log_bh_fid: float
    log_bh_ha: float
    log_bh_hb: float
    log_l5100: float
    log_bha: float
    fwhm_bha: float
    e_fwhm_bha: float
    ew_bha: float
    multipeak_flag: int
    host_proxy_delta_r: float
    preselection_score: float
    mass_bin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sdss_msigma_sample/liu2019_gh06_like_three_window")
    parser.add_argument("--catalog-base-url", default=CDS_LIU2019_BASE)
    parser.add_argument("--max-download", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--overwrite-catalog", action="store_true")
    parser.add_argument("--overwrite-spectra", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--full-spectra", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)

    parser.add_argument("--coverage-mode", choices=("three_window", "at_least_two"), default="three_window")
    parser.add_argument("--min-z", type=float, default=0.010)
    parser.add_argument("--max-z-three-window", type=float, default=0.055)
    parser.add_argument("--max-z-two-window", type=float, default=0.350)
    parser.add_argument("--obs-wave-min", type=float, default=SDSS_LEGACY_OBS_MIN)
    parser.add_argument("--obs-wave-max", type=float, default=SDSS_LEGACY_OBS_MAX)

    parser.add_argument("--min-fwhm-bha", type=float, default=1000.0)
    parser.add_argument("--max-fwhm-bha", type=float, default=15000.0)
    parser.add_argument("--max-fwhm-frac-err", type=float, default=0.35)
    parser.add_argument("--min-log-l5100", type=float, default=40.8)
    parser.add_argument("--max-log-l5100", type=float, default=44.0)
    parser.add_argument("--max-log-bha", type=float, default=42.9)
    parser.add_argument("--min-log-mbh", type=float, default=5.0)
    parser.add_argument("--max-log-mbh", type=float, default=9.5)
    parser.add_argument("--min-host-proxy-delta-r", type=float, default=0.10)
    parser.add_argument("--allow-multipeak", action="store_true")
    return parser.parse_args()


def _as_abs_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_value(text: str, dtype):
    text = text.strip()
    if not text:
        return "" if dtype is str else np.nan
    try:
        if dtype is str:
            return text
        if dtype is int:
            value = int(text)
            return np.nan if value <= -900 else value
        value = float(text)
        return np.nan if value <= -900.0 else value
    except ValueError:
        return "" if dtype is str else np.nan


def download_cds_file(output_dir: Path, base_url: str, filename: str, overwrite: bool, timeout: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path
    url = f"{base_url.rstrip('/')}/{filename}"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        gz_response = requests.get(f"{url}.gz", timeout=timeout)
        gz_response.raise_for_status()
        gz_path = output_dir / f"{filename}.gz"
        gz_path.write_bytes(gz_response.content)
        with gzip.open(gz_path, "rb") as fin, open(path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return path
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def parse_fixed_width(path: Path, fields: dict[str, tuple[int, int, type]], force_csv: Path | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            rows.append({name: _parse_value(line[start:end], dtype) for name, (start, end, dtype) in fields.items()})
    table = pd.DataFrame(rows)
    if force_csv is not None:
        table.to_csv(force_csv, index=False)
    return table


def load_liu2019_catalog(output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    table1_path = download_cds_file(output_dir, args.catalog_base_url, "table1.dat", args.overwrite_catalog, args.timeout)
    table2_path = download_cds_file(output_dir, args.catalog_base_url, "table2.dat", args.overwrite_catalog, args.timeout)
    table3_path = download_cds_file(output_dir, args.catalog_base_url, "table3.dat", args.overwrite_catalog, args.timeout)

    parsed_csv = output_dir / "liu2019_dr7_broadline_key_columns.csv"
    if parsed_csv.exists() and not args.overwrite_catalog:
        return pd.read_csv(parsed_csv)

    t1 = parse_fixed_width(table1_path, TABLE1_FIELDS)
    t2 = parse_fixed_width(table2_path, TABLE2_FIELDS)
    t3 = parse_fixed_width(table3_path, TABLE3_FIELDS)
    table = t1.merge(t2, on="seq", how="inner").merge(t3, on="seq", how="left")
    table["host_proxy_delta_r"] = table["psf_r"] - table["petro_r"]
    table["fwhm_bha_frac_err"] = table["e_fwhm_bha"] / table["fwhm_bha"].replace(0, np.nan)
    table["source_catalog"] = "Liu et al. 2019, ApJS, 243, 21; CDS J/ApJS/243/21"
    table.to_csv(parsed_csv, index=False)
    return table


def finite(table: pd.DataFrame, column: str) -> pd.Series:
    return np.isfinite(pd.to_numeric(table[column], errors="coerce"))


def add_window_coverage(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    table = table.copy()
    rest_min = args.obs_wave_min / (1.0 + table["z"].to_numpy(float))
    rest_max = args.obs_wave_max / (1.0 + table["z"].to_numpy(float))
    n_windows = np.zeros(len(table), dtype=int)
    for name, (lo, hi) in SIGMA_WINDOWS.items():
        observable = np.isfinite(rest_min) & np.isfinite(rest_max) & (rest_min <= lo) & (rest_max >= hi)
        table[f"{name}_observable"] = observable.astype(int)
        n_windows += observable.astype(int)
    table["n_sigma_windows_observable"] = n_windows
    return table


def apply_preselection(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    table = add_window_coverage(table, args)
    max_z = args.max_z_three_window if args.coverage_mode == "three_window" else args.max_z_two_window
    min_windows = 3 if args.coverage_mode == "three_window" else 2
    mask = (
        finite(table, "z")
        & (table["z"] >= args.min_z)
        & (table["z"] <= max_z)
        & (table["n_sigma_windows_observable"] >= min_windows)
        & finite(table, "plate")
        & finite(table, "mjd")
        & finite(table, "fiber")
        & finite(table, "log_bh_fid")
        & (table["log_bh_fid"] >= args.min_log_mbh)
        & (table["log_bh_fid"] <= args.max_log_mbh)
        & finite(table, "log_l5100")
        & (table["log_l5100"] >= args.min_log_l5100)
        & (table["log_l5100"] <= args.max_log_l5100)
        & finite(table, "log_bha")
        & (table["log_bha"] <= args.max_log_bha)
        & finite(table, "fwhm_bha")
        & (table["fwhm_bha"] >= args.min_fwhm_bha)
        & (table["fwhm_bha"] <= args.max_fwhm_bha)
        & finite(table, "fwhm_bha_frac_err")
        & (table["fwhm_bha_frac_err"] <= args.max_fwhm_frac_err)
        & finite(table, "host_proxy_delta_r")
        & (table["host_proxy_delta_r"] >= args.min_host_proxy_delta_r)
    )
    if not args.allow_multipeak:
        mask &= table["multipeak_flag"].fillna(-1).astype(int) == 0

    selected = table.loc[mask].copy()
    selected["preselection_score"] = (
        1.20 * selected["host_proxy_delta_r"].to_numpy(float)
        + 0.55 * selected["n_sigma_windows_observable"].to_numpy(float)
        + 0.18 * np.clip(selected["ew_bha"].to_numpy(float), 0.0, 500.0) / 100.0
        - 0.55 * np.maximum(selected["log_l5100"].to_numpy(float) - 43.0, 0.0)
        - 0.45 * np.maximum(selected["log_bha"].to_numpy(float) - 42.0, 0.0)
        - 0.30 * selected["z"].to_numpy(float) / max(max_z, 1e-6)
        - 1.00 * selected["fwhm_bha_frac_err"].to_numpy(float)
    )
    selected["mass_bin"] = pd.cut(
        selected["log_bh_fid"],
        bins=[5.0, 6.5, 7.2, 7.9, 8.6, 9.5],
        include_lowest=True,
    ).astype(str)
    selected = selected.sort_values("preselection_score", ascending=False)
    return selected


def stratified_top_sample(candidates: pd.DataFrame, max_download: int, seed: int) -> pd.DataFrame:
    if max_download <= 0 or len(candidates) <= max_download:
        return candidates.copy()
    rng = np.random.default_rng(seed)
    chosen_indices: list[int] = []
    groups = list(candidates.groupby("mass_bin", sort=True))
    quota = max(1, int(np.ceil(max_download / max(len(groups), 1))))
    for _, group in groups:
        jitter = rng.normal(0.0, 1e-4, len(group))
        ranked = group.assign(_rank_score=group["preselection_score"].to_numpy(float) + jitter)
        chosen_indices.extend(ranked.sort_values("_rank_score", ascending=False).head(quota).index.tolist())
    if len(chosen_indices) < max_download:
        remaining = candidates.loc[~candidates.index.isin(chosen_indices)].copy()
        jitter = rng.normal(0.0, 1e-4, len(remaining))
        remaining = remaining.assign(_rank_score=remaining["preselection_score"].to_numpy(float) + jitter)
        chosen_indices.extend(
            remaining.sort_values("_rank_score", ascending=False)
            .head(max_download - len(chosen_indices))
            .index.tolist()
        )
    return candidates.loc[chosen_indices[:max_download]].sort_values("preselection_score", ascending=False).copy()


def spectrum_url(row: pd.Series, full_spectra: bool) -> str:
    plate = int(row["plate"])
    mjd = int(row["mjd"])
    fiber = int(row["fiber"])
    spectra_dir = "spectra" if full_spectra else "spectra/lite"
    return f"{SDSS_DR18_LEGACY_SPEC_BASE}/{spectra_dir}/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits"


def download_file(session: requests.Session, url: str, path: Path, timeout: float, overwrite: bool) -> str:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as response:
        if response.status_code == 404:
            return "missing_404"
        response.raise_for_status()
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(path)
    return "downloaded"


def build_records(
    selected: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[DownloadRecord]:
    spec_dir = output_dir / ("spectra_full" if args.full_spectra else "spectra_lite")
    session = requests.Session()
    records: list[DownloadRecord] = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiber = int(row["fiber"])
        local_path = spec_dir / f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        url = spectrum_url(row, args.full_spectra)
        status = "not_downloaded" if args.no_download else download_file(session, url, local_path, args.timeout, args.overwrite_spectra)
        local_path_text = "" if status == "missing_404" else str(local_path)
        records.append(
            DownloadRecord(
                source_catalog=str(row["source_catalog"]),
                sdss_name=str(row["sdss_name"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                z=float(row["z"]),
                plate=plate,
                mjd=mjd,
                fiberID=fiber,
                run2d="26",
                url=url,
                local_path=local_path_text,
                download_status=status,
                coverage_mode=args.coverage_mode,
                n_sigma_windows_observable=int(row["n_sigma_windows_observable"]),
                cahk_observable=int(row["cahk_observable"]),
                fe_observable=int(row["fe_observable"]),
                cat_observable=int(row["cat_observable"]),
                log_bh_fid=float(row["log_bh_fid"]),
                log_bh_ha=float(row["log_bh_ha"]),
                log_bh_hb=float(row["log_bh_hb"]),
                log_l5100=float(row["log_l5100"]),
                log_bha=float(row["log_bha"]),
                fwhm_bha=float(row["fwhm_bha"]),
                e_fwhm_bha=float(row["e_fwhm_bha"]),
                ew_bha=float(row["ew_bha"]),
                multipeak_flag=int(row["multipeak_flag"]),
                host_proxy_delta_r=float(row["host_proxy_delta_r"]),
                preselection_score=float(row["preselection_score"]),
                mass_bin=str(row["mass_bin"]),
            )
        )
        print(
            f"[{i:04d}/{len(selected):04d}] {row['sdss_name']} "
            f"z={row['z']:.4f} logM={row['log_bh_fid']:.2f} "
            f"logL5100={row['log_l5100']:.2f} windows={int(row['n_sigma_windows_observable'])} {status}"
        )
    return records


def write_outputs(candidates: pd.DataFrame, selected: pd.DataFrame, records: list[DownloadRecord], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "liu2019_gh06_like_preselected_candidates.csv", index=False)
    selected.to_csv(output_dir / "liu2019_gh06_like_selected_catalog.csv", index=False)
    records_as_dicts = [asdict(record) for record in records]
    manifest_path = output_dir / "liu2019_gh06_like_sdss_manifest.csv"
    downloaded_manifest_path = output_dir / "liu2019_gh06_like_sdss_manifest_downloaded.csv"
    if records_as_dicts:
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records_as_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(records_as_dicts)
        downloaded_records = [
            row
            for row in records_as_dicts
            if row["download_status"] in {"downloaded", "exists"} and row.get("local_path")
        ]
        with open(downloaded_manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records_as_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(downloaded_records)
    (output_dir / "liu2019_gh06_like_sdss_manifest.json").write_text(
        json.dumps(records_as_dicts, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "mother_catalog": "Liu et al. 2019, ApJS, 243, 21; CDS J/ApJS/243/21",
        "science_motivation": (
            "GH06-like selection: low-redshift broad-Halpha/type-1 AGNs with "
            "observable Ca H+K, Fe/metal, and CaT dispersion windows when coverage_mode=three_window."
        ),
        "selection": {
            "coverage_mode": args.coverage_mode,
            "redshift": {
                "min_z": args.min_z,
                "max_z_three_window": args.max_z_three_window,
                "max_z_two_window": args.max_z_two_window,
            },
            "fwhm_bha": [args.min_fwhm_bha, args.max_fwhm_bha],
            "max_fwhm_bha_frac_err": args.max_fwhm_frac_err,
            "log_l5100": [args.min_log_l5100, args.max_log_l5100],
            "max_log_bha": args.max_log_bha,
            "log_bh_fid": [args.min_log_mbh, args.max_log_mbh],
            "min_host_proxy_delta_r": args.min_host_proxy_delta_r,
            "multipeak": "excluded unless --allow-multipeak is passed",
        },
        "n_preselected": int(len(candidates)),
        "n_selected": int(len(selected)),
        "n_downloaded_or_existing": int(sum(record.download_status in {"downloaded", "exists"} for record in records)),
        "download_status_counts": pd.Series([record.download_status for record in records]).value_counts().to_dict()
        if records
        else {},
        "mass_bin_counts_selected": selected["mass_bin"].value_counts().sort_index().to_dict() if len(selected) else {},
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote downloaded manifest: {downloaded_manifest_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = _as_abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_liu2019_catalog(output_dir, args)
    candidates = apply_preselection(table, args)
    selected = stratified_top_sample(candidates, args.max_download, args.seed)
    records = build_records(selected, output_dir, args)
    write_outputs(candidates, selected, records, output_dir, args)


if __name__ == "__main__":
    main()
