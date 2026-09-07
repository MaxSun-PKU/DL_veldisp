"""
Download commonly used public Fe II template files into reference/feii_templates.

The UV file distributed with PyQSOFit is a practical composite used in quasar
fitting workflows; its code comments describe the wavelength stitching as
VW01, Salviander et al. 2007, and Tsuzuki et al. 2006.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reference" / "feii_templates"

TEMPLATE_URLS = {
    "fe_uv.txt": "https://raw.githubusercontent.com/legolason/PyQSOFit/master/src/pyqsofit/fe_uv.txt",
    "fe_optical.txt": "https://raw.githubusercontent.com/legolason/PyQSOFit/master/src/pyqsofit/fe_optical.txt",
}


def download(url: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return
    with urlopen(url, timeout=30) as response:
        content = response.read()
    output_path.write_bytes(content)
    print(f"downloaded: {output_path} ({len(content)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, url in TEMPLATE_URLS.items():
        output_path = OUTPUT_DIR / filename
        download(url, output_path, overwrite=args.overwrite)
        manifest[filename] = {
            "url": url,
            "note": "Downloaded by scripts/download_feii_templates.py",
        }

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
