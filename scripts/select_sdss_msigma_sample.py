"""
Select and download a low-redshift SDSS type-1 AGN sample for an M-sigma pilot.

The mother catalog is the Shen et al. (2011) SDSS DR7 quasar property catalog
from CDS/VizieR (J/ApJS/194/45).  We use the Shen DR7 plate-MJD-fiber spectra
so that the downloaded spectrum is the same spectrum used for the Hbeta
single-epoch black-hole mass measurement.
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
CDS_SHEN_CATALOG_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/194/45/catalog.dat.gz"
SDSS_DR18_LEGACY_SPEC_BASE = "https://dr18.sdss.org/sas/dr18/spectro/sdss/redux/26"


FIELD_SPECS = {
    "shen_sdss_id": (0, 18, str),
    "ra": (19, 28, float),
    "dec": (29, 38, float),
    "z_dr7": (39, 47, float),
    "plate": (48, 52, int),
    "fiber": (53, 56, int),
    "mjd": (57, 62, int),
    "target_flag": (63, 72, int),
    "n_spec": (73, 74, int),
    "uniform_flag": (75, 76, int),
    "i_abs_mag": (77, 83, float),
    "log_lbol": (84, 90, float),
    "e_log_lbol": (91, 97, float),
    "bal_flag": (98, 99, int),
    "log_l5100": (134, 140, float),
    "e_log_l5100": (141, 147, float),
    "fwhm_bhb": (392, 399, float),
    "e_fwhm_bhb": (400, 407, float),
    "ew_bhb": (408, 415, float),
    "e_ew_bhb": (416, 424, float),
    "ew_fe_hb": (523, 530, float),
    "sn_hb": (556, 561, float),
    "chi2_hb": (562, 569, float),
    "log_bh_hb_vp06": (896, 901, float),
    "e_log_bh_hb_vp06": (902, 907, float),
    "log_bh_fid": (956, 961, float),
    "e_log_bh_fid": (962, 967, float),
    "log_edd": (968, 975, float),
    "special_flag": (976, 977, int),
    "z_hw": (978, 986, float),
    "e_z_hw": (987, 995, float),
    "gmag": (1003, 1009, float),
    "rmag": (1010, 1016, float),
    "imag": (1017, 1023, float),
}


@dataclass
class DownloadRecord:
    sdss_name: str
    shen_sdss_id: str
    ra: float
    dec: float
    z: float
    z_dr7: float
    z_hw: float
    redshift_source: str
    plate: int
    mjd: int
    fiberID: int
    run2d: str
    url: str
    local_path: str
    download_status: str
    log_bh_hb_vp06: float
    e_log_bh_hb_vp06: float
    log_l5100: float
    e_log_l5100: float
    fwhm_bhb: float
    e_fwhm_bhb: float
    sn_hb: float
    chi2_hb: float
    ew_bhb: float
    ew_fe_hb: float
    log_edd: float
    bal_flag: int
    special_flag: int
    preselection_score: float
    mass_bin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sdss_msigma_sample/shen2011_lowz_hbeta")
    parser.add_argument("--catalog-url", default=CDS_SHEN_CATALOG_URL)
    parser.add_argument("--max-download", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--overwrite-catalog", action="store_true")
    parser.add_argument("--overwrite-spectra", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="Write selected tables but do not download spectra.")
    parser.add_argument("--full-spectra", action="store_true", help="Download full spec files instead of spectra/lite.")
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--min-z", type=float, default=0.06)
    parser.add_argument("--max-z", type=float, default=0.35)
    parser.add_argument("--min-sn-hb", type=float, default=20.0)
    parser.add_argument("--max-chi2-hb", type=float, default=5.0)
    parser.add_argument("--min-fwhm-hb", type=float, default=1000.0)
    parser.add_argument("--max-fwhm-hb", type=float, default=15000.0)
    parser.add_argument("--max-fwhm-frac-err", type=float, default=0.50)
    parser.add_argument("--min-log-l5100", type=float, default=42.0)
    parser.add_argument("--max-log-l5100", type=float, default=45.0)
    parser.add_argument("--min-log-mbh", type=float, default=5.5)
    parser.add_argument("--max-log-mbh", type=float, default=10.2)
    parser.add_argument("--allow-bal", action="store_true")
    parser.add_argument("--allow-special-interest", action="store_true")
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
            return int(text)
        return float(text)
    except ValueError:
        return "" if dtype is str else np.nan


def download_catalog(output_dir: Path, url: str, overwrite: bool, timeout: float) -> Path:
    gz_path = output_dir / "shen2011_dr7_quasar_properties_catalog.dat.gz"
    dat_path = output_dir / "shen2011_dr7_quasar_properties_catalog.dat"
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite or not gz_path.exists():
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        gz_path.write_bytes(response.content)
    if overwrite or not dat_path.exists():
        with gzip.open(gz_path, "rb") as fin, open(dat_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return dat_path


def parse_catalog(dat_path: Path, parsed_csv: Path, force: bool = False) -> pd.DataFrame:
    if parsed_csv.exists() and not force:
        return pd.read_csv(parsed_csv)

    rows = []
    with open(dat_path, "r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            rows.append(
                {
                    name: _parse_value(line[start:end], dtype)
                    for name, (start, end, dtype) in FIELD_SPECS.items()
                }
            )
    table = pd.DataFrame(rows)
    table["sdss_name"] = "J" + table["shen_sdss_id"].astype(str)
    z_hw_valid = np.isfinite(table["z_hw"]) & (table["z_hw"] > 0)
    table["z"] = np.where(z_hw_valid, table["z_hw"], table["z_dr7"])
    table["redshift_source"] = np.where(z_hw_valid, "HewettWild2010", "SDSS_DR7_QSO_catalog")
    table.to_csv(parsed_csv, index=False)
    return table


def finite(table: pd.DataFrame, column: str) -> pd.Series:
    return np.isfinite(pd.to_numeric(table[column], errors="coerce"))


def apply_catalog_preselection(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    fwhm_frac_err = table["e_fwhm_bhb"] / table["fwhm_bhb"].replace(0, np.nan)
    mask = (
        finite(table, "z")
        & (table["z"] >= args.min_z)
        & (table["z"] <= args.max_z)
        & finite(table, "log_bh_hb_vp06")
        & (table["log_bh_hb_vp06"] >= args.min_log_mbh)
        & (table["log_bh_hb_vp06"] <= args.max_log_mbh)
        & finite(table, "log_l5100")
        & (table["log_l5100"] >= args.min_log_l5100)
        & (table["log_l5100"] <= args.max_log_l5100)
        & finite(table, "fwhm_bhb")
        & (table["fwhm_bhb"] >= args.min_fwhm_hb)
        & (table["fwhm_bhb"] <= args.max_fwhm_hb)
        & finite(table, "e_fwhm_bhb")
        & (table["e_fwhm_bhb"] > 0)
        & finite(table, "sn_hb")
        & (table["sn_hb"] >= args.min_sn_hb)
        & finite(table, "chi2_hb")
        & (table["chi2_hb"] > 0)
        & (table["chi2_hb"] <= args.max_chi2_hb)
        & finite(table, "plate")
        & finite(table, "mjd")
        & finite(table, "fiber")
    )
    mask &= np.isfinite(fwhm_frac_err) & (fwhm_frac_err <= args.max_fwhm_frac_err)
    if not args.allow_bal:
        mask &= table["bal_flag"].fillna(-1).astype(int) == 0
    if not args.allow_special_interest:
        mask &= table["special_flag"].fillna(-1).astype(int) == 0

    selected = table.loc[mask].copy()
    selected["fwhm_bhb_frac_err"] = fwhm_frac_err.loc[selected.index]
    selected["preselection_score"] = (
        2.0 * np.log10(np.clip(selected["sn_hb"].to_numpy(float), 1.0, None))
        - 0.16 * selected["chi2_hb"].to_numpy(float)
        - 0.35 * np.maximum(selected["log_l5100"].to_numpy(float) - 44.2, 0.0)
        - 0.60 * selected["z"].to_numpy(float)
        - 0.50 * selected["fwhm_bhb_frac_err"].to_numpy(float)
    )
    selected["mass_bin"] = pd.cut(
        selected["log_bh_hb_vp06"],
        bins=[5.5, 7.5, 8.0, 8.5, 9.0, 10.2],
        include_lowest=True,
    ).astype(str)
    selected = selected.sort_values("preselection_score", ascending=False)
    return selected


def stratified_top_sample(candidates: pd.DataFrame, max_download: int, seed: int) -> pd.DataFrame:
    if max_download <= 0 or len(candidates) <= max_download:
        return candidates.copy()

    rng = np.random.default_rng(seed)
    mass_bins = list(candidates["mass_bin"].dropna().unique())
    quota = max(1, int(np.ceil(max_download / max(len(mass_bins), 1))))
    chosen_indices: list[int] = []
    for _, group in candidates.groupby("mass_bin", sort=False):
        jitter = rng.normal(0.0, 1e-4, len(group))
        group = group.assign(_rank_score=group["preselection_score"].to_numpy(float) + jitter)
        chosen_indices.extend(group.sort_values("_rank_score", ascending=False).head(quota).index.tolist())

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
    *,
    no_download: bool,
    full_spectra: bool,
    timeout: float,
    overwrite: bool,
) -> list[DownloadRecord]:
    spec_dir = output_dir / ("spectra_full" if full_spectra else "spectra_lite")
    session = requests.Session()
    records: list[DownloadRecord] = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiber = int(row["fiber"])
        local_path = spec_dir / f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        url = spectrum_url(row, full_spectra)
        status = "not_downloaded" if no_download else download_file(session, url, local_path, timeout, overwrite)
        if status == "missing_404":
            local_path_text = ""
        else:
            local_path_text = str(local_path)
        records.append(
            DownloadRecord(
                sdss_name=str(row["sdss_name"]),
                shen_sdss_id=str(row["shen_sdss_id"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                z=float(row["z"]),
                z_dr7=float(row["z_dr7"]),
                z_hw=float(row["z_hw"]),
                redshift_source=str(row["redshift_source"]),
                plate=plate,
                mjd=mjd,
                fiberID=fiber,
                run2d="26",
                url=url,
                local_path=local_path_text,
                download_status=status,
                log_bh_hb_vp06=float(row["log_bh_hb_vp06"]),
                e_log_bh_hb_vp06=float(row["e_log_bh_hb_vp06"]),
                log_l5100=float(row["log_l5100"]),
                e_log_l5100=float(row["e_log_l5100"]),
                fwhm_bhb=float(row["fwhm_bhb"]),
                e_fwhm_bhb=float(row["e_fwhm_bhb"]),
                sn_hb=float(row["sn_hb"]),
                chi2_hb=float(row["chi2_hb"]),
                ew_bhb=float(row["ew_bhb"]),
                ew_fe_hb=float(row["ew_fe_hb"]),
                log_edd=float(row["log_edd"]),
                bal_flag=int(row["bal_flag"]),
                special_flag=int(row["special_flag"]),
                preselection_score=float(row["preselection_score"]),
                mass_bin=str(row["mass_bin"]),
            )
        )
        print(
            f"[{i:04d}/{len(selected):04d}] {row['sdss_name']} "
            f"z={row['z']:.4f} logM={row['log_bh_hb_vp06']:.2f} "
            f"S/N(Hb)={row['sn_hb']:.1f} {status}"
        )
    return records


def write_outputs(candidates: pd.DataFrame, selected: pd.DataFrame, records: list[DownloadRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "shen2011_msigma_preselected_candidates.csv", index=False)
    selected.to_csv(output_dir / "shen2011_msigma_selected_catalog.csv", index=False)
    records_as_dicts = [asdict(record) for record in records]
    manifest_path = output_dir / "shen2011_msigma_sdss_manifest.csv"
    downloaded_manifest_path = output_dir / "shen2011_msigma_sdss_manifest_downloaded.csv"
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
    (output_dir / "shen2011_msigma_sdss_manifest.json").write_text(
        json.dumps(records_as_dicts, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "n_preselected": int(len(candidates)),
        "n_selected": int(len(selected)),
        "n_downloaded_or_existing": int(sum(record.download_status in {"downloaded", "exists"} for record in records)),
        "n_missing": int(sum(record.download_status.startswith("missing") for record in records)),
        "selection": {
            "mother_catalog": "Shen et al. 2011, ApJS, 194, 45; CDS J/ApJS/194/45",
            "redshift": "Hewett & Wild 2010 zHW when available, otherwise SDSS DR7 QSO catalog z",
            "bh_mass": "Hbeta VP06 single-epoch mass column logBHHV",
            "bal": "BAL=0 unless --allow-bal is passed",
            "special_interest": "SpF=0 unless --allow-special-interest is passed",
        },
        "mass_bin_counts_selected": selected["mass_bin"].value_counts().sort_index().to_dict(),
        "download_status_counts": pd.Series([record.download_status for record in records]).value_counts().to_dict()
        if records
        else {},
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote downloaded manifest: {downloaded_manifest_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = _as_abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_path = download_catalog(output_dir, args.catalog_url, args.overwrite_catalog, args.timeout)
    parsed_csv = output_dir / "shen2011_dr7_parsed_key_columns.csv"
    table = parse_catalog(dat_path, parsed_csv, force=args.overwrite_catalog)
    candidates = apply_catalog_preselection(table, args)
    selected = stratified_top_sample(candidates, args.max_download, args.seed)
    records = build_records(
        selected,
        output_dir,
        no_download=args.no_download,
        full_spectra=args.full_spectra,
        timeout=args.timeout,
        overwrite=args.overwrite_spectra,
    )
    write_outputs(candidates, selected, records, output_dir)


if __name__ == "__main__":
    main()
