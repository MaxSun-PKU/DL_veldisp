"""
Download the 40 SDSS spectra in Greene & Ho (2006), ApJ, 641, 117.

The source list is Table 1 of:
    Greene, J. E. & Ho, L. C. 2006, ApJ, 641, 117
    "Measuring Stellar Velocity Dispersions in Active Galaxies"

The paper used DR3 spectra.  This script resolves each Table 1 SDSS J2000
name to the nearest modern SkyServer spectrum and downloads the matching
plate-MJD-fiber FITS file from the SDSS SAS DR18 legacy SDSS archive.  The
plate-MJD-fiber observation is the useful invariant for our ivar/noise work.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
SAS_BASE_URL = "https://dr18.sdss.org/sas/dr18/spectro/sdss/redux"


GREENE_2006_TABLE1_NAMES = (
    "J000805.62+145023.4",
    "J004236.86-104921.8",
    "J011703.58+000027.3",
    "J020615.99-001729.1",
    "J033013.26-053236.0",
    "J080243.39+310403.3",
    "J080538.66+261005.4",
    "J082510.23+375919.7",
    "J083202.16+461425.7",
    "J083949.64+484701.4",
    "J092438.88+560746.9",
    "J093812.26+074340.0",
    "J094529.37+093610.4",
    "J094838.42+403043.7",
    "J095033.15+441851.7",
    "J101912.57+635802.7",
    "J104451.73+063548.6",
    "J111857.68+580323.5",
    "J112536.16+542257.1",
    "J120114.35-034041.0",
    "J120257.81+045045.0",
    "J121607.09+504930.0",
    "J121754.97+583935.6",
    "J122324.13+024044.4",
    "J123237.48+662452.3",
    "J130620.97+531823.1",
    "J131305.80+012755.9",
    "J134952.84+020445.1",
    "J140018.42+050242.2",
    "J143452.45+483942.7",
    "J144629.97+500130.5",
    "J145706.80+494008.4",
    "J145901.35+611353.5",
    "J150745.00+512710.2",
    "J161156.31+521116.8",
    "J161951.31+405847.2",
    "J162012.75+400906.1",
    "J162345.88+410456.3",
    "J164909.58+361325.8",
    "J232721.96+152437.3",
)


@dataclass
class SpectrumRecord:
    sdss_name: str
    ra: float
    dec: float
    specObjID: int
    plate: int
    mjd: int
    fiberID: int
    run2d: str
    spec_ra: float
    spec_dec: float
    z: float
    spec_class: str
    distance_arcmin: float
    url: str
    local_path: str
    status: str


def parse_sdss_name(name: str) -> tuple[float, float]:
    match = re.fullmatch(r"J(\d{2})(\d{2})(\d{2}\.\d+)([+-])(\d{2})(\d{2})(\d{2}\.\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse SDSS J2000 name: {name}")
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (int(hh) + int(mm) / 60.0 + float(ss) / 3600.0)
    dec = int(dd) + int(dm) / 60.0 + float(ds) / 3600.0
    if sign == "-":
        dec *= -1.0
    return ra, dec


def parse_csv_response(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def query_nearest_spectrum(
    session: requests.Session,
    ra: float,
    dec: float,
    radius_arcmin: float,
    timeout: float,
) -> dict[str, str]:
    sql = (
        "SELECT TOP 1 "
        "s.specObjID,s.plate,s.mjd,s.fiberID,s.run2d,s.ra,s.dec,s.z,s.class,n.distance "
        f"FROM dbo.fGetNearbySpecObjEq({ra:.8f},{dec:.8f},{radius_arcmin:.4f}) n "
        "JOIN SpecObjAll s ON n.specObjID=s.specObjID "
        "ORDER BY n.distance"
    )
    response = session.get(SKYSERVER_SQL_URL, params={"cmd": sql, "format": "csv"}, timeout=timeout)
    response.raise_for_status()
    rows = parse_csv_response(response.text)
    if not rows:
        raise RuntimeError(f"No SDSS spectrum found within {radius_arcmin} arcmin of ra={ra}, dec={dec}")
    return rows[0]


def spectrum_url(row: dict[str, str], lite: bool) -> str:
    plate = int(row["plate"])
    mjd = int(row["mjd"])
    fiber = int(row["fiberID"])
    run2d = row["run2d"]
    spectra_dir = "spectra/lite" if lite else "spectra"
    return f"{SAS_BASE_URL}/{run2d}/{spectra_dir}/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits"


def download_file(session: requests.Session, url: str, path: Path, timeout: float, overwrite: bool) -> str:
    if path.exists() and not overwrite and path.stat().st_size > 0:
        return "exists"

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(path)
    return "downloaded"


def write_manifest(records: Iterable[SpectrumRecord], output_dir: Path) -> None:
    records = list(records)
    json_path = output_dir / "greene2006_sdss_manifest.json"
    csv_path = output_dir / "greene2006_sdss_manifest.csv"
    source_path = output_dir / "greene2006_table1_sources.txt"

    json_path.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n", encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    source_path.write_text("\n".join(record.local_path for record in records) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sdss_reference_spectra/greene2006")
    parser.add_argument("--radius-arcmin", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--full", action="store_true", help="Download full spec FITS files instead of lite spectra.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    records: list[SpectrumRecord] = []
    for i, name in enumerate(GREENE_2006_TABLE1_NAMES, start=1):
        ra, dec = parse_sdss_name(name)
        row = query_nearest_spectrum(session, ra, dec, args.radius_arcmin, args.timeout)
        url = spectrum_url(row, lite=not args.full)
        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiber = int(row["fiberID"])
        local_path = output_dir / f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        status = download_file(session, url, local_path, args.timeout, args.overwrite)

        record = SpectrumRecord(
            sdss_name=name,
            ra=ra,
            dec=dec,
            specObjID=int(row["specObjID"]),
            plate=plate,
            mjd=mjd,
            fiberID=fiber,
            run2d=row["run2d"],
            spec_ra=float(row["ra"]),
            spec_dec=float(row["dec"]),
            z=float(row["z"]),
            spec_class=row["class"],
            distance_arcmin=float(row["distance"]),
            url=url,
            local_path=str(local_path),
            status=status,
        )
        records.append(record)
        print(
            f"[{i:02d}/{len(GREENE_2006_TABLE1_NAMES)}] {name} -> "
            f"plate={plate:04d} mjd={mjd} fiber={fiber:04d} z={record.z:.5f} {status}"
        )

    write_manifest(records, output_dir)
    print(f"Saved manifest and source list to {output_dir}")


if __name__ == "__main__":
    main()
