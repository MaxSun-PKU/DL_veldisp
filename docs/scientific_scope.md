# Scientific scope and limitations

## Reference validation

The reference network was trained on 220,000 SDSS-like mock spectra over
30-700 km/s. The independent 22,000-spectrum mock test set gives an RMSE of
79.5 km/s over the full range. Precision improves strongly when the host
fraction is high, the stellar absorption is resolved, and the predicted
uncertainty is small. The validation-defined quality subset gives an RMSE of
25.0 km/s after the mock-only linear correction.

On 39 SDSS spectra from Greene & Ho (2006), the direct network predictions
give an RMSE of 45.4 km/s and a mean offset of -1.5 km/s. The 18 spectra that
pass the confidence selection give an RMSE of 29.1 km/s. The linear correction
derived from mocks worsens the observational offset and is therefore not used
as the default central value for real spectra.

For 22 nearby AGNs with higher-resolution Keck/LRIS measurements, the SDSS
predictions give an RMSE of 45.8 km/s and a mean offset of -8.3 km/s. A
higher-redshift comparison has larger scatter. Those spectra lack CaT coverage
and differ in physical aperture and AGN luminosity, so the causes cannot be
separated with the current sample.

An application to 300 low-redshift SDSS broad-line AGNs recovers a positive
black hole mass--velocity-dispersion relation. An orthogonal-distance fit to
the 123 confidence-selected spectra gives a slope of 3.62 +/- 0.39, consistent
with local inactive-galaxy relations over the sampled range.

## Supported use

The present validation supports population analyses of nearby SDSS type 1 AGNs
that have detectable host absorption and useful coverage of all three stellar
regions. Individual measurements should be accompanied by their calibrated
uncertainty, confidence category, resolution ratio, host fraction, coverage,
and reconstruction diagnostics.

## Known limitations

- The network tends to overpredict dispersions near the SDSS resolution limit
  and underpredict values above about 250 km/s.
- Confidence selection reduces residuals in the mock and GH06 samples, but it
  does not guarantee that an individual prediction is accurate.
- SDSS fiber measurements are luminosity-weighted line-of-sight dispersions
  over the fiber aperture. They are not automatically equivalent to a
  dispersion measured within one bulge effective radius.
- Stellar abundance patterns, extinction, AGN line profiles, and sky
  subtraction can differ from the mock prior.
- The uncertainty model has broader tails than a Gaussian distribution. The
  local validation-risk term partly accounts for this behavior.
- The three confidence categories were calibrated for the reference SDSS
  domain. A different survey requires instrument-specific mocks and a new
  calibration.

The model should not be used as a substitute for higher-resolution stellar
kinematics when precise measurements of an individual object are required.
