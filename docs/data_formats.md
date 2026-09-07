# Data formats

## Observed SDSS spectra

`scripts/infer_sdss_quasar_sigma.py` reads SDSS, BOSS, and eBOSS `spec*.fits`
files. It searches the binary tables for these columns:

| Quantity | Required names | Use |
|---|---|---|
| Flux | `flux` | Observed flux density |
| Flux precision | `ivar` | Converted to normalized uncertainty weighting |
| Wavelength | `loglam`, `wave`, or `wavelength` | Shifted to rest frame and interpolated |
| Pixel flags | `and_mask` or `or_mask` | Removes flagged pixels when available |
| Resolution | `wdisp` | Instrumental Gaussian dispersion in SDSS pixels |

The flux unit can remain in the standard SDSS scale because each spectrum and
its uncertainty are divided by the same continuum normalization. The code
uses the SDSS logarithmic pixel scale to convert `WDISP` to km/s.

### Manifest CSV

The recommended manifest columns are:

- `local_path`: path to the FITS file;
- `z`: systemic redshift used to construct the rest-frame spectrum; and
- `sdss_name`: stable object identifier used when joining output tables.

If `z` is absent, the script searches the FITS headers and tables. A catalog
redshift is preferable because the adopted systemic redshift is then explicit.

## Mock-training HDF5

The generator writes arrays with the first dimension equal to the number of
mock spectra. The main groups of datasets are:

- spectral arrays: `spectra`, `ivar`, `fit_mask`, `sigma_mask`,
  `inst_sigma_kms_grid`, `host_spectra`, and `scaled_agn_spectra`;
- physical labels: `sigma_star`, `host_fraction`, `sfh_weights`, stellar age,
  and stellar metallicity;
- AGN labels: continuum slope, luminosity, Fe II strength and width, Balmer
  parameters, narrow-line parameters, and the EV1 latent coordinate;
- observing labels: redshift, S/N, SDSS noise-reference index, instrumental
  velocity dispersion, sky residuals, and telluric residuals; and
- split metadata: `train_indices`, `val_indices`, `test_indices`, and `split`.

The file attribute `config_json` records the generation parameters. Template
and SDSS-reference manifests are stored as additional JSON attributes.

## Compact inference assets

`scripts/export_inference_assets.py` copies only:

- `wave_grid`;
- `template_library`;
- `feii_template_bank` and the Park template when present;
- `sps_logzsol` and `sps_age_gyr`;
- the median instrumental velocity dispersion; and
- `config_json`.

It does not contain mock spectra or labels.

## Inference CSV

The output contains the prediction and the information needed to assess it.
The most important columns are:

| Column | Meaning |
|---|---|
| `pred_sigma_kms` | Reported stellar velocity dispersion |
| `pred_sigma_raw_kms` | Direct network output |
| `pred_sigma_affine_kms` | Validation-mock linear correction, provided for diagnostics |
| `recommended_sigma_err_kms` | Calibrated observational uncertainty |
| `pred_host_fraction` | Predicted host contribution |
| `median_inst_sigma_kms` | Median instrumental velocity dispersion in the fitted regions |
| `pred_sigma_resolution_ratio` | `pred_sigma_raw_kms / median_inst_sigma_kms` |
| `confidence` | High, intermediate, or low confidence |
| `quality_failed_checks` | Failed measurement criteria |
| `quality_advisory_flags` | Conditions that need scientific review |
| `fit_recon_mae` | Reconstruction error over valid fitted pixels |
| `sigma_recon_mae` | Reconstruction error in the stellar-kinematic windows |
| `risk_expected_abs_error_kms` | Local validation-mock estimate of absolute error |

Column names in older calibration files can include `science_grade` or
`golden_sample`. They are retained for backward compatibility. The public
interpretation is the simpler high/intermediate/low confidence category.
