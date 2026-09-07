# Model and training

## Forward model

Each mock spectrum is the sum of a broadened host-galaxy spectrum and an AGN
component. The host basis is generated with FSPS across 12 metallicities and
24 stellar ages. The stellar-population prior favors old and intermediate-age
bulges while allowing a smaller young component. Intrinsic host velocity
dispersion spans 30-700 km/s.

The AGN model contains a power-law continuum, broad and narrow emission lines,
and a bank of Fe II templates. A fraction of the mocks follows approximate
Eigenvector 1 and Baldwin-effect correlations among optical luminosity, Fe II,
[O III], and Balmer-line properties. The remaining mocks use independent AGN
draws to preserve broader nuisance variation.

Redshift-dependent wavelength coverage, flux uncertainty, bad-pixel patterns,
and instrumental resolution are sampled from real SDSS reference spectra.
The generator also adds sky and telluric residuals. These operations make the
training domain specific to SDSS observations.

## Network

`HybridAGNNet` has four connected parts:

1. A one-dimensional convolutional backbone uses Inception-style kernels and
   CBAM channel/wavelength attention to summarize the full spectrum.
2. A kinematic branch processes the flux restricted to Ca H+K, the Fe/metal
   region, and CaT.
3. A line-shape branch processes the locally normalized flux and its first and
   second finite differences inside those regions.
4. Physical heads predict velocity dispersion, its preliminary variance,
   stellar-population weights, host fraction, and AGN parameters. The latter
   quantities generate explicit host and AGN reconstructions.

The three regional dispersion estimates are combined according to wavelength
coverage. CaT receives the highest priority, followed by the Fe/metal window
and Ca H+K, following the empirical assessment of Greene & Ho (2006).

## Training losses

The principal loss supervises stellar velocity dispersion. Additional losses
constrain:

- the distribution and interval of predicted dispersions;
- ordering of spectra with separated true dispersions;
- the response to controlled line-broadening augmentation;
- host fraction and stellar-population weights;
- the total, host, and stellar-window reconstructions; and
- the predicted dispersion variance.

Sample weights prevent low-host-fraction, low-S/N, poorly covered, and
resolution-limited spectra from dominating the gradient. Their weights remain
nonzero, so these spectra are still represented during training.

The first three epochs use spectra with host fraction at least 0.3 and S/N at
least 20 to establish sensitivity to line width. Full training then uses the
complete training split. The validation score combines dispersion performance
over the full range, equally weighted dispersion intervals, the range below
250 km/s, reconstruction error, and host-fraction error.

## Apple MPS behavior

The trainer selects Apple MPS before CUDA or CPU when MPS is available. Some
grouped-convolution and reduction-heavy backward paths produced non-finite
gradients with the tested PyTorch MPS backend. The default MPS path keeps the
corresponding reconstruction terms in the reported loss while detaching their
unstable broadening and shared-encoder gradients. Supervised dispersion,
host-fraction, stellar-population, uncertainty, line-broadening, and stable
interval losses continue to train the encoder.

The command-line flags `--allow-mps-reconstruction-grad` and
`--allow-mps-global-sigma-losses` expose the full gradient paths for debugging
or later PyTorch versions. Use them only after checking gradients on the target
hardware.

## Split discipline

- Training indices update model weights.
- Validation indices select the checkpoint, fit the optional linear correction,
  calibrate uncertainty, and build the local quality-risk model.
- Test indices are used once for final mock performance.

Observed validation samples, including GH06 and the higher-resolution Keck
comparisons, do not enter model training or mock uncertainty calibration.
