# DL_veldisp

`DL_veldisp` measures the stellar velocity dispersion of type 1 AGN and quasar
host galaxies from SDSS spectra. The model is trained on forward-modeled
host-plus-AGN spectra and returns

- the host stellar velocity dispersion, `pred_sigma_kms`;
- a calibrated uncertainty, `recommended_sigma_err_kms`;
- predicted host and AGN spectral components;
- host-fraction and reconstruction diagnostics; and
- a high-, intermediate-, or low-confidence assessment for each spectrum.

The pipeline combines a one-dimensional convolutional network, channel and
wavelength attention, line-shape branches for three stellar absorption
regions, and an explicit stellar-population plus AGN reconstruction model.

> **Research status**
>
> This repository accompanies an astrophysical analysis in preparation. The
> reference model is supported primarily for nearby SDSS type 1 AGNs with
> detectable host absorption and useful coverage of Ca H+K, the 5250-5820 A
> Fe/metal region, and the Ca II triplet. Read
> [Scientific scope and limitations](docs/scientific_scope.md) before using an
> individual prediction as a physical measurement.

## Quick start: infer from SDSS spectra

### 1. Create the environment

```bash
git clone https://github.com/MaxSun-PKU/DL_veldisp.git
cd DL_veldisp
conda env create -f environment.yml
conda activate dl-veldisp
```

Mock generation requires FSPS and `python-fsps`. Install FSPS following the
[official python-FSPS instructions](https://dfm.io/python-fsps/current/installation/)
and set its location before generating mocks:

```bash
export SPS_HOME=/absolute/path/to/fsps
```

FSPS is not required when applying the bundled reference model to an SDSS
spectrum.

### 2. Prepare spectra

The inference script accepts either a directory of SDSS `spec*.fits` files or
a CSV manifest. A manifest is recommended because it records the adopted
systemic redshift:

```csv
local_path,z,sdss_name
/data/spec-0331-52368-0121.fits,0.01948,J015530.02-085704.0
```

The FITS spectrum must provide `flux`, `ivar`, and `loglam` or wavelength. The
code converts `ivar` to the flux-error weighting used by the model. SDSS
`and_mask` or `or_mask` values are used when available. `WDISP` is converted
to the wavelength-dependent instrumental velocity dispersion. If `WDISP` is
missing, the model configuration supplies a fallback value and the output is
flagged accordingly.

### 3. Run inference

```bash
python scripts/infer_sdss_quasar_sigma.py \
  --assets artifacts/reference_model/inference_assets.h5 \
  --model artifacts/reference_model/model.pth \
  --calibration-json artifacts/reference_model/calibration.json \
  --manifest-csv examples/sdss_manifest.csv \
  --central-calibration raw \
  --output-csv outputs/sdss_sigma.csv \
  --output-pdf outputs/sdss_sigma_diagnostics.pdf
```

Replace the example path in `examples/sdss_manifest.csv` with a local FITS
file. The default reported central value is the direct network prediction.
The validation-derived affine correction is written to a separate column but
is not adopted for observed spectra because it did not improve the GH06
external comparison.

Important output columns are described in
[docs/data_formats.md](docs/data_formats.md). In particular, inspect
`confidence`, `quality_failed_checks`, `quality_advisory_flags`,
`recommended_sigma_err_kms`, `pred_host_fraction`, and
`pred_sigma_resolution_ratio` together with `pred_sigma_kms`.

## Reproduce the training pipeline

The reference configuration is recorded in
[`notebooks/training_run.ipynb`](notebooks/training_run.ipynb). It executes the
same four stages as the command-line workflow:

1. Download or supply SDSS spectra used as empirical noise, mask, coverage,
   and resolution references.
2. Generate forward-modeled host-plus-AGN spectra with fixed train,
   validation, and test indices.
3. Train the network and select a checkpoint using the validation split.
4. Calibrate uncertainties on validation mocks and report final metrics on
   the independent test split.

The full reference run contains 220,000 spectra and occupies tens of
gigabytes. Start with the smoke test below before launching the reference run.

### Download empirical SDSS references

```bash
python scripts/download_greene2006_sdss.py \
  --output-dir sdss_reference_spectra/greene2006
```

These public SDSS spectra are downloaded at run time and are not committed to
the repository.

### Download optional Fe II templates

```bash
python scripts/download_feii_templates.py
```

The generator includes the Park et al. (2022) optical template and can add the
UV and optical templates distributed by PyQSOFit. Additional two-column
wavelength-flux templates can be passed repeatedly with `--feii-template`.

### Generate a small smoke-test data set

```bash
export SPS_HOME=/absolute/path/to/fsps

python scripts/generate_sdss_quasar_mocks.py \
  --output mock_data/smoke_1000.h5 \
  --n-samples 1000 \
  --seed 20260630 \
  --fsps-home "$SPS_HOME" \
  --wave-min 1200 \
  --wave-max 9000 \
  --n-pixels 6000 \
  --sigma-min 30 \
  --sigma-max 700 \
  --sigma-sampling-mode science_bins \
  --science-sigma-bin-edges 30,60,80,120,180,250,400,700 \
  --science-sigma-bin-probs 0.07,0.08,0.18,0.20,0.19,0.16,0.12 \
  --agn-parameter-mode mixed_ev1_baldwin \
  --agn-covariant-fraction 0.45 \
  --redshift-min 0.01 \
  --redshift-max 0.90 \
  --snr-min 5 \
  --snr-max 80 \
  --train-frac 0.80 \
  --val-frac 0.10 \
  --test-frac 0.10 \
  --sdss-reference-dir sdss_reference_spectra/greene2006
```

The output contains explicit `train_indices`, `val_indices`, and
`test_indices`. The test indices must remain untouched until checkpoint
selection and uncertainty calibration are complete.

### Train

```bash
python scripts/train_hybrid_sdss.py \
  --h5 mock_data/smoke_1000.h5 \
  --save models/smoke_model.pth \
  --epochs 3 \
  --batch-size 32 \
  --num-workers 0 \
  --val-num-workers 0 \
  --sigma-warmup-epochs 0 \
  --full-val-every 1 \
  --early-stopping-patience 0 \
  --min-full-training-epochs 1 \
  --channels flux,ivar,mask,sigma_flux,inst_sigma,region_cat,region_fe,region_cahk,agn_log_l5100,sigma_mask \
  --history-json models/smoke_history.json
```

The reference run uses a three-epoch high-host-fraction warmup, line-broadening
augmentation, sigma-interval balancing, early stopping, and periodic full
validation. The notebook records every parameter used for that run. Apple MPS,
CUDA, and CPU are selected automatically in that order when available.

### Evaluate and calibrate uncertainty

```bash
python scripts/evaluate_hybrid_sdss.py \
  --h5 mock_data/smoke_1000.h5 \
  --model models/smoke_model.pth \
  --split test \
  --calibrate-split val \
  --output figures/smoke_test.pdf \
  --metrics-json figures/smoke_test.json \
  --calibration-json-out models/smoke_calibration.json \
  --batch-size 128 \
  --num-workers 0
```

### Export compact inference assets

```bash
python scripts/export_inference_assets.py \
  --source mock_data/smoke_1000.h5 \
  --output models/smoke_inference_assets.h5
```

This file contains the wavelength grid, SPS basis, Fe II basis, and physical
configuration. It excludes mock spectra, labels, and split indices.

## Model inputs

The reference checkpoint uses ten arrays sampled on the same rest-frame
wavelength grid:

| Channel | Meaning |
|---|---|
| `flux` | Spectrum normalized near rest-frame 5100 A |
| `ivar` | Clipped relative precision derived from the SDSS flux uncertainty |
| `mask` | Valid pixels used by the reconstruction model |
| `sigma_flux` | Locally centered and scaled flux inside the stellar-kinematic windows |
| `inst_sigma` | SDSS instrumental velocity dispersion divided by 100 km/s |
| `region_cat` | Indicator that Ca II triplet is the principal covered region |
| `region_fe` | Indicator that the Fe/metal window is the principal covered region |
| `region_cahk` | Indicator that Ca H+K is the principal covered region |
| `agn_log_l5100` | Approximate optical luminosity, standardized around log L5100 = 45 |
| `sigma_mask` | Pixels retained in the three stellar-kinematic regions |

See [docs/model_and_training.md](docs/model_and_training.md) for the network,
loss terms, mock priors, and platform-specific training behavior.

## Reference model

The bundled reference assets are:

| File | Purpose |
|---|---|
| `artifacts/reference_model/model.pth` | Trained PyTorch state dictionary |
| `artifacts/reference_model/inference_assets.h5` | Wavelength grid and spectral bases |
| `artifacts/reference_model/calibration.json` | Validation-based uncertainty and confidence calibration |
| `artifacts/reference_model/checksums.sha256` | SHA-256 checksums for the three files |

The checkpoint expects the ten channels listed above and a 6000-pixel
rest-frame grid spanning 1200-9000 A. Do not mix the checkpoint with assets or
calibration from another training run.

## Repository layout

```text
DL_veldisp/
|-- artifacts/reference_model/   # compact model files used for inference
|-- docs/                        # scientific and technical documentation
|-- examples/                    # input manifest template
|-- notebooks/                   # portable end-to-end training notebook
|-- scripts/                     # generation, training, evaluation, inference
|-- data_loader_v2.py            # lazy HDF5 data loader and channel builder
|-- physmodel.py                 # HybridAGNNet and training losses
|-- environment.yml             # validated conda environment
`-- requirements.txt            # pip dependency list
```

Generated data, SDSS FITS files, figures, paper drafts, and intermediate
checkpoints are intentionally excluded from version control.

## Verify the installation

```bash
python -m unittest discover -s tests -v
```

The tests check that the compact assets contain no training spectra and that
the bundled checkpoint produces finite outputs with the published ten-channel
configuration. GitHub Actions runs the same checks on every push and pull
request.

## Scientific validation

The reference model was evaluated on an independent mock test split, the
Greene & Ho (2006) SDSS sample, and SDSS counterparts of AGNs with published
higher-resolution Keck measurements. A separate application recovered the
population-level black hole mass--stellar velocity-dispersion trend in 300
low-redshift SDSS broad-line AGNs. Numerical results and interpretation are
summarized in [docs/scientific_scope.md](docs/scientific_scope.md).

## Citation

If this code contributes to a publication, cite the accompanying paper after
its bibliographic information becomes available and cite the relevant input
data and spectral models. The current references include:

- Greene & Ho (2006), *Measuring Stellar Velocity Dispersions in Active
  Galaxies*, [doi:10.1086/500353](https://doi.org/10.1086/500353)
- Park et al. (2022), *A New Iron Emission Template for Active Galactic
  Nuclei. I.*, [doi:10.3847/1538-4365/ac3f3e](https://doi.org/10.3847/1538-4365/ac3f3e)
- Vestergaard & Wilkes (2001),
  [doi:10.1086/320357](https://doi.org/10.1086/320357)
- Tsuzuki et al. (2006),
  [doi:10.1086/506376](https://doi.org/10.1086/506376)
- Conroy, Gunn, & White (2009) and Conroy & Gunn (2010) for FSPS
- Paszke et al. (2019) for PyTorch

Machine-readable software citation metadata are provided in `CITATION.cff`.

## License

The source code is released under the GNU General Public License v3.0. Spectral
templates and downloaded survey data retain their original attribution and
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Questions and reproducibility reports

Please open a GitHub issue and include the command, Python and PyTorch
versions, device (`mps`, `cuda`, or `cpu`), and the relevant output flags. Do
not upload proprietary spectra or credentials in an issue.
