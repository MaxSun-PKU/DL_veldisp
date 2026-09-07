#!/usr/bin/env python
"""Build a high-resolution sigma truth sample and crossmatch it to SDSS spectra.

The goal is an external validation table for the DL velocity-dispersion
pipeline.  We keep the literature truth definition explicit because the
published measurements use different apertures and observing setups.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "external_validation" / "highres_truth"

SDSS_OBS_MIN = 3800.0
SDSS_OBS_MAX = 9200.0
WINDOWS = {
    "cahk": (3850.0, 4100.0),
    "mgfe": (5100.0, 5350.0),
    "cat": (8480.0, 8700.0),
}


WOO06_SIGMA_ERR = {
    "S01": 8,
    "S04": 8,
    "S05": 5,
    "S06": 14,
    "S07": 13,
    "S08": 11,
    "S09": 15,
    "S11": 9,
    "S12": 22,
    "S23": 8,
    "S24": 10,
    "S26": 8,
    "S28": 10,
}

WOO08_SIGMA_ERR = {
    "W09": 19,
    "W11": 21,
    "W14": 20,
    "W17": 14,
    "W22": 21,
}


def clean_latex(value: str) -> str:
    value = value.split("%")[0]
    value = value.replace("\\nodata", "")
    value = value.replace("\\,", " ")
    value = value.replace("\\", "")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("$", "")
    value = value.replace("pm", "+/-")
    value = value.replace("arcsec", "arcsec")
    value = value.replace("...", "")
    return " ".join(value.strip().split())


def parse_pm(value: str) -> tuple[float, float]:
    value = clean_latex(value)
    m = re.search(r"([0-9.]+)\s*\+/-\s*([0-9.]+)", value)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def coord_from_hms_dms(ra_hms: str, dec_dms: str) -> tuple[float, float]:
    ra = ":".join(clean_latex(ra_hms).split())
    dec = ":".join(clean_latex(dec_dms).split())
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def coord_from_compact_name(name: str) -> tuple[float, float]:
    m = re.match(
        r"(?P<rah>\d{2})(?P<ram>\d{2})(?P<ras>\d{2}\.\d+)"
        r"(?P<sign>[+-])(?P<decd>\d{2})(?P<decm>\d{2})(?P<decs>\d{2}\.\d+)",
        name,
    )
    if not m:
        raise ValueError(f"Cannot parse compact coordinate target: {name}")
    ra = f"{m.group('rah')}:{m.group('ram')}:{m.group('ras')}"
    dec = f"{m.group('sign')}{m.group('decd')}:{m.group('decm')}:{m.group('decs')}"
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def normalize_target_id(target: str) -> str:
    target = clean_latex(target)
    target = target.replace("...........................", "")
    target = target.strip(". ")
    if re.fullmatch(r"W\d", target):
        target = f"W0{target[1:]}"
    return target


def parse_le2023_table() -> pd.DataFrame:
    tex = (BASE / "le2023_source" / "mr3.tex").read_text()
    block = tex.split("\\startdata", 1)[1].split("\\enddata", 1)[0]
    rows = []
    for raw in block.splitlines():
        if "&" not in raw or "\\\\" not in raw:
            continue
        parts = [clean_latex(p) for p in raw.replace("\\\\", "").split("&")]
        if len(parts) < 12:
            continue
        target = normalize_target_id(parts[0])
        try:
            rows.append(
                {
                    "literature_id": target,
                    "z_lit": float(parts[1]),
                    "log_l5100_1e44": float(parts[2]),
                    "hbeta_sigma_kms": float(parts[3]),
                    "log_mbh": float(parts[4]),
                    "sigma_truth_kms": float(parts[5]),
                    "oiii_core_sigma_kms": float(parts[6]),
                    "oiii_out_sigma_kms": float(parts[7]),
                    "vmax_kms": float(parts[8]),
                    "snr_lit": float(parts[11]),
                }
            )
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    df["source_reference"] = "Le+2023 compilation"
    return df


def parse_woo06_coords() -> pd.DataFrame:
    tex = (BASE / "woo2006_mbh_source" / "ms.tex").read_text()
    rows = []
    for raw in tex.splitlines():
        if not re.match(r"\s*S\d+", raw):
            continue
        parts = [clean_latex(p) for p in raw.replace("\\\\", "").split("&")]
        if len(parts) < 4:
            continue
        target = normalize_target_id(parts[0])
        if target not in WOO06_SIGMA_ERR:
            continue
        if not re.match(r"^\d{1,2}\s+\d{1,2}\s+[0-9.]+$", parts[2]):
            continue
        if not re.match(r"^[+-]\d{1,2}\s+\d{1,2}\s+[0-9.]+$", parts[3]):
            continue
        ra, dec = coord_from_hms_dms(parts[2], parts[3])
        rows.append(
            {
                "literature_id": target,
                "ra_lit": ra,
                "dec_lit": dec,
                "z_original": float(re.sub(r"[^0-9.]", "", parts[1])),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("literature_id")


def parse_woo08_table() -> pd.DataFrame:
    tex = (BASE / "woo2008_source" / "ms20astro.tex").read_text()
    rows = []
    for raw in tex.splitlines():
        if not re.match(r"\s*W\d+", raw):
            continue
        parts = [clean_latex(p) for p in raw.replace("\\\\", "").split("&")]
        if len(parts) < 11:
            continue
        target = normalize_target_id(parts[0])
        if target not in WOO08_SIGMA_ERR:
            continue
        ra, dec = coord_from_hms_dms(parts[2], parts[3])
        sigma, sigma_err = parse_pm(parts[10])
        rows.append(
            {
                "literature_id": target,
                "ra_lit": ra,
                "dec_lit": dec,
                "z_original": float(parts[1]),
                "sigma_truth_original_kms": sigma,
                "sigma_truth_err_original_kms": sigma_err,
            }
        )
    return pd.DataFrame(rows)


def build_woo_le_highz() -> pd.DataFrame:
    le = parse_le2023_table()
    woo06 = parse_woo06_coords()
    woo08 = parse_woo08_table()
    woo = pd.concat([woo06, woo08], ignore_index=True)
    df = le.merge(woo, on="literature_id", how="inner")
    df["candidate_group"] = "highz_woo06_woo08_keck_lris"
    df["truth_instrument"] = "Keck/LRIS long-slit"
    df["truth_aperture"] = "published stellar-feature extraction; not SDSS fiber matched"
    df["sigma_truth_err_kms"] = df["literature_id"].map({**WOO06_SIGMA_ERR, **WOO08_SIGMA_ERR})
    df["validation_priority"] = "B"
    df["priority_reason"] = (
        "High-S/N Keck long-slit sigma truth and SDSS counterpart; "
        "z~0.35-0.57 means CaT is outside SDSS, so this is a difficult two-window test."
    )
    return df


def build_sdssrm_supplement() -> pd.DataFrame:
    le = parse_le2023_table()
    mask = le["literature_id"].str.match(r"^\d{6}\.\d+[+-]\d{6}\.\d+$")
    df = le[mask].copy()
    coords = [coord_from_compact_name(t) for t in df["literature_id"]]
    df["ra_lit"] = [x[0] for x in coords]
    df["dec_lit"] = [x[1] for x in coords]
    df["z_original"] = df["z_lit"]
    df["candidate_group"] = "sdss_rm_coadd_supplement"
    df["truth_instrument"] = "SDSS-RM coadded spectrum"
    df["truth_aperture"] = "SDSS-RM coadded fiber spectrum"
    df["sigma_truth_err_kms"] = df["sigma_truth_kms"] * math.log(10.0) * 0.04
    df["validation_priority"] = "C"
    df["priority_reason"] = (
        "Useful supplementary check, but the sigma truth is from coadded SDSS-RM spectra, "
        "not independent high-resolution Keck/ESI data."
    )
    return df


def parse_bennert2010() -> pd.DataFrame:
    tex = (BASE / "bennert2010_source" / "ms.tex").read_text()
    sample_rows = []
    for raw in tex.splitlines():
        if not re.match(r"\s*\d{4}[+-]\d{4}\s*&", raw):
            continue
        parts = [clean_latex(p) for p in raw.replace("\\\\", "").split("&")]
        if len(parts) < 8:
            continue
        obj = parts[0]
        if not parts[1].startswith("SDSSJ"):
            continue
        ra, dec = coord_from_hms_dms(parts[5], parts[6])
        sample_rows.append(
            {
                "literature_id": obj,
                "sdss_name_lit": parts[1],
                "z_lit": float(parts[2]),
                "ra_lit": ra,
                "dec_lit": dec,
                "alternative_name": parts[8] if len(parts) > 8 else "",
            }
        )
    sample = pd.DataFrame(sample_rows).drop_duplicates("literature_id")

    sigma_rows = []
    for raw in tex.splitlines():
        if not re.match(r"\s*\d{4}[+-]\d{4}\s*&", raw):
            continue
        parts = [p.strip() for p in raw.replace("\\\\", "").split("&")]
        if len(parts) < 5:
            continue
        obj = clean_latex(parts[0])
        if not re.search(r"\$\\pm\$", parts[1]):
            continue
        spat, spat_err = parse_pm(parts[1])
        ap_reff, ap_reff_err = parse_pm(parts[2])
        ap_1p5, ap_1p5_err = parse_pm(parts[3])
        sigma_rows.append(
            {
                "literature_id": obj,
                "sigma_spat_reff_kms": spat,
                "sigma_spat_reff_err_kms": spat_err,
                "sigma_ap_reff_kms": ap_reff,
                "sigma_ap_reff_err_kms": ap_reff_err,
                "sigma_truth_kms": ap_1p5,
                "sigma_truth_err_kms": ap_1p5_err,
                "sigma_truth_original_kms": spat,
                "sigma_truth_err_original_kms": spat_err,
            }
        )
    sigma = pd.DataFrame(sigma_rows).drop_duplicates("literature_id")
    df = sample.merge(sigma, on="literature_id", how="inner")
    df["z_original"] = df["z_lit"]
    df["candidate_group"] = "local_bennert2010_keck_lris"
    df["source_reference"] = "Bennert+2010 local SDSS Seyfert 1 sample"
    df["truth_instrument"] = "Keck/LRIS long-slit"
    df["truth_aperture"] = "sigma_ap_1.5arcsec; closest published proxy for SDSS 3 arcsec fiber"
    df["validation_priority"] = "A"
    df["priority_reason"] = (
        "SDSS-selected local type-1 AGN with high-quality Keck/LRIS stellar kinematics; "
        "published sigma_ap,1.5arcsec is explicitly intended for SDSS-fiber comparison."
    )
    return df


def window_coverage_fraction(z: float, rest_min: float, rest_max: float) -> float:
    obs_min = rest_min * (1.0 + z)
    obs_max = rest_max * (1.0 + z)
    overlap = max(0.0, min(obs_max, SDSS_OBS_MAX) - max(obs_min, SDSS_OBS_MIN))
    return overlap / max(obs_max - obs_min, 1.0e-6)


def add_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (wmin, wmax) in WINDOWS.items():
        frac = [window_coverage_fraction(float(z), wmin, wmax) for z in out["z_lit"]]
        out[f"{name}_coverage_fraction"] = frac
        out[f"{name}_usable_in_sdss"] = out[f"{name}_coverage_fraction"] >= 0.7
    out["n_sigma_windows_usable"] = (
        out["cahk_usable_in_sdss"].astype(int)
        + out["mgfe_usable_in_sdss"].astype(int)
        + out["cat_usable_in_sdss"].astype(int)
    )
    out["coverage_note"] = np.where(
        out["n_sigma_windows_usable"] >= 3,
        "CaHK+Mg/Fe+CaT mostly covered by SDSS",
        np.where(
            out["n_sigma_windows_usable"] == 2,
            "Two stellar windows covered by SDSS; CaT usually outside for z>0.1",
            "Limited stellar-window coverage in SDSS",
        ),
    )
    return out


def query_sdss_datalab(ra: float, dec: float, radius_arcsec: float = 2.0) -> pd.DataFrame:
    radius_deg = radius_arcsec / 3600.0
    sql = (
        "select ra,dec,plate,mjd,fiberid,z,class,run2d,specobjid "
        "from sdss_dr17.specobj "
        f"where q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{radius_deg:.10f})"
    )
    url = "https://datalab.noirlab.edu/query/query"
    headers = {
        "X-DL-AuthToken": "anonymous.0.0.anon_access",
        "X-DL-TimeoutRequest": "60",
        "Content-Type": "text/ascii",
        "X-DL-ClientVersion": "manual",
    }
    params = {
        "sql": sql,
        "ofmt": "csv",
        "out": "None",
        "async": "False",
        "drop": "False",
        "profile": "default",
    }
    response = requests.get(url, params=params, headers=headers, timeout=90)
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.lower().startswith("error"):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def nearest_sdss_match(row: pd.Series, max_radius_arcsec: float) -> dict[str, object]:
    for radius in (2.0, max_radius_arcsec):
        if radius <= 2.0 and max_radius_arcsec <= 2.0:
            radii = [2.0]
        try:
            match = query_sdss_datalab(float(row.ra_lit), float(row.dec_lit), radius)
        except Exception as exc:  # keep the catalog build alive for transient service errors
            return {"sdss_query_error": repr(exc), "sdss_matched": False}
        if len(match) == 0:
            if radius < max_radius_arcsec:
                time.sleep(0.05)
                continue
            return {"sdss_matched": False}
        target = SkyCoord(float(row.ra_lit) * u.deg, float(row.dec_lit) * u.deg)
        coords = SkyCoord(match["ra"].to_numpy(float) * u.deg, match["dec"].to_numpy(float) * u.deg)
        sep = target.separation(coords).arcsec
        order = np.argsort(sep)
        best = match.iloc[int(order[0])].to_dict()
        best["sdss_sep_arcsec"] = float(sep[int(order[0])])
        best["sdss_matched"] = True
        best["sdss_match_radius_arcsec"] = radius
        return {f"sdss_{k}": v for k, v in best.items()}
    return {"sdss_matched": False}


def add_sdss_crossmatch(df: pd.DataFrame, max_radius_arcsec: float) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        if i and i % 20 == 0:
            print(f"Crossmatched {i}/{len(df)} candidates")
        info = nearest_sdss_match(row, max_radius_arcsec=max_radius_arcsec)
        rows.append(info)
        time.sleep(0.08)
    match_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), match_df.reset_index(drop=True)], axis=1)
    if "sdss_plate" in out.columns:
        out["sdss_sas_dr17_url"] = out.apply(make_sdss_url, axis=1)
    return out


def query_sdss_skyserver_batch(df: pd.DataFrame, max_radius_arcsec: float) -> pd.DataFrame:
    values = ",".join(
        [
            f"('{row.literature_id}',cast({float(row.ra_lit):.8f} as float),"
            f"cast({float(row.dec_lit):.8f} as float))"
            for _, row in df.iterrows()
        ]
    )
    sql = f"""
select c.literature_id,c.ra0,c.dec0,
       x.ra as sdss_ra,x.dec as sdss_dec,x.plate as sdss_plate,x.mjd as sdss_mjd,
       x.fiberid as sdss_fiberid,x.z as sdss_z,x.objclass as sdss_class,
       x.run2d as sdss_run2d,x.specobjid as sdss_specobjid,x.sep_arcsec as sdss_sep_arcsec
from (values {values}) as c(literature_id,ra0,dec0)
cross apply (
 select top 1 s.ra,s.dec,s.plate,s.mjd,s.fiberID as fiberid,s.z,s.class as objclass,
        s.run2d,s.specObjID as specobjid,
        n.distance*60.0 as sep_arcsec
 from dbo.fGetNearbySpecObjEq(c.ra0,c.dec0,{max_radius_arcsec / 60.0}) n
 join SpecObjAll s on s.specObjID=n.specObjID
 order by n.distance
) x
"""
    url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
    response = requests.get(url, params={"cmd": sql, "format": "csv"}, timeout=90)
    response.raise_for_status()
    result = pd.read_csv(io.StringIO(response.text), comment="#")
    if "error_message" in result.columns:
        raise RuntimeError(result["error_message"].iloc[0])
    return result


def add_sdss_crossmatch_skyserver(
    df: pd.DataFrame,
    max_radius_arcsec: float,
    chunk_size: int = 25,
) -> pd.DataFrame:
    chunks = []
    for start in range(0, len(df), chunk_size):
        stop = min(start + chunk_size, len(df))
        chunk = df.iloc[start:stop]
        print(f"SkyServer crossmatch {start + 1}-{stop}/{len(df)}", flush=True)
        try:
            chunks.append(query_sdss_skyserver_batch(chunk, max_radius_arcsec=max_radius_arcsec))
        except Exception as exc:
            print(f"  batch failed: {exc!r}; falling back to smaller chunks", flush=True)
            if len(chunk) == 1:
                failed = pd.DataFrame(
                    {"literature_id": [chunk["literature_id"].iloc[0]], "sdss_query_error": [repr(exc)]}
                )
                chunks.append(failed)
            else:
                chunks.append(
                    add_sdss_crossmatch_skyserver(
                        chunk,
                        max_radius_arcsec=max_radius_arcsec,
                        chunk_size=max(1, len(chunk) // 2),
                    )[["literature_id"] + [c for c in chunk.columns if c.startswith("sdss_")]]
                )
    match = pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()
    out = df.merge(match, on="literature_id", how="left")
    out["sdss_matched"] = out["sdss_plate"].notna() if "sdss_plate" in out.columns else False
    out["sdss_sas_dr17_url"] = out.apply(make_sdss_url, axis=1)
    return out


def make_sdss_url(row: pd.Series) -> str:
    if not bool(row.get("sdss_matched", row.get("sdss_sdss_matched", False))):
        return ""
    try:
        plate = int(row["sdss_plate"])
        mjd = int(row["sdss_mjd"])
        fiber = int(row.get("sdss_fiberid", row.get("sdss_fiberID")))
        run2d = str(row["sdss_run2d"])
    except Exception:
        return ""
    return (
        "https://data.sdss.org/sas/dr17/sdss/spectro/redux/"
        f"{run2d}/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
    )


def build_catalog(include_supplement: bool) -> pd.DataFrame:
    frames = [parse_bennert2010(), build_woo_le_highz()]
    if include_supplement:
        frames.append(build_sdssrm_supplement())
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = add_coverage_flags(df)
    preferred_cols = [
        "candidate_group",
        "validation_priority",
        "literature_id",
        "sdss_name_lit",
        "alternative_name",
        "ra_lit",
        "dec_lit",
        "z_lit",
        "sigma_truth_kms",
        "sigma_truth_err_kms",
        "truth_instrument",
        "truth_aperture",
        "n_sigma_windows_usable",
        "cahk_coverage_fraction",
        "mgfe_coverage_fraction",
        "cat_coverage_fraction",
        "coverage_note",
        "log_l5100_1e44",
        "log_mbh",
        "snr_lit",
        "source_reference",
        "priority_reason",
    ]
    for col in preferred_cols:
        if col not in df.columns:
            df[col] = np.nan
    other_cols = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + other_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "highres_sigma_sdss_crossmatch_candidates.csv",
    )
    parser.add_argument("--max-radius-arcsec", type=float, default=3.0)
    parser.add_argument("--include-sdssrm-supplement", action="store_true")
    parser.add_argument("--no-query", action="store_true")
    parser.add_argument("--backend", choices=("skyserver", "datalab"), default="skyserver")
    args = parser.parse_args()

    df = build_catalog(include_supplement=args.include_sdssrm_supplement)
    if not args.no_query:
        if args.backend == "skyserver":
            df = add_sdss_crossmatch_skyserver(df, max_radius_arcsec=args.max_radius_arcsec)
        else:
            df = add_sdss_crossmatch(df, max_radius_arcsec=args.max_radius_arcsec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} candidates to {args.output}")
    print(df.groupby(["candidate_group", "validation_priority"]).size())
    if "sdss_matched" in df.columns:
        print("SDSS matches:")
        print(df.groupby(["candidate_group"])["sdss_matched"].sum())


if __name__ == "__main__":
    main()
