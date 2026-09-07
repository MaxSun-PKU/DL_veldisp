# Third-party data and software

This repository includes or downloads spectral templates and survey data that
should be cited separately from the `DL_veldisp` source code.

## Fe II templates

- `scripts/park_template.txt` represents the optical Fe II template described
  by Park et al. (2022), doi:10.3847/1538-4365/ac3f3e.
- `reference/feii_templates/fe_uv.txt` and `fe_optical.txt` are distributed by
  [PyQSOFit](https://github.com/legolason/PyQSOFit), which is released under
  GPL-3.0. The UV composite incorporates wavelength regions associated with
  Vestergaard & Wilkes (2001) and Tsuzuki et al. (2006). See the manifest and
  `reference/feii_templates/README.md` for provenance.

Users should cite the underlying template papers in scientific work. Adding a
template to a local run does not transfer ownership or change its original
terms of use.

## SDSS spectra

The SDSS FITS spectra are downloaded from the public SDSS Science Archive
Server and are not stored in this repository. Publications should include the
SDSS acknowledgment and the survey/data-release citations appropriate to the
spectra used.

## FSPS

FSPS and python-FSPS are external dependencies. They are not redistributed in
this repository. Cite Conroy, Gunn, & White (2009), Conroy & Gunn (2010), and
the python-FSPS software release as appropriate.
