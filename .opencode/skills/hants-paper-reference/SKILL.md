---
name: hants-paper-reference
description: Use when implementing, reviewing, debugging, or optimizing HANTS and needing the paper-faithful meaning of NOF, SF, IDRT, FET, DOD, stopping behavior, or outlier rejection rules from Roerink et al. 2000.
---

# HANTS Paper Reference

## Overview

This skill condenses the HANTS paper into implementation checks. Use it to keep code aligned with Roerink et al. (2000) instead of drifting toward a different robust harmonic fitting algorithm.

Primary source in this repo:

- `docs/reference/Roerink 等 - 2000 - Reconstructing cloudfree NDVI composites using Fourier analysis of time series.pdf`

## When to Use

- Checking whether an HANTS implementation still matches the paper
- Reviewing code that changes `NOF`, `SF`, `IDRT`, `FET`, or `DOD`
- Explaining why HANTS rejects only one outlier direction
- Debugging stopping conditions, minimum observation logic, or unexpected outlier retention
- Optimizing HANTS without silently changing its semantics

Do not use this skill as a generic harmonic regression tutorial. It is for paper-faithful HANTS behavior.

## Quick Reference

| Parameter | Paper meaning | Implementation consequence |
|---|---|---|
| `NOF` | Number of frequencies, including zero frequency (mean) | Curve uses `2 * NOF - 1` parameters |
| `SF` | Hi/Lo suppression flag | Reject high or low outliers, not both unless intentionally changing the algorithm |
| `IDRT` | Invalid data rejection threshold | Values above or below a known invalid threshold are excluded before fitting |
| `FET` | Fit error tolerance | Stop when all remaining observations are within tolerance in the selected suppression direction |
| `DOD` | Degree of overdeterminedness | Final fit must retain at least `(2 * NOF - 1) + DOD` valid observations |

## Paper Constraints

### Core algorithm

- HANTS fits a least-squares curve based on harmonic components (`sines` and `cosines`), not FFT-only processing.
- It is intended for irregularly spaced observations and iterative outlier rejection.
- Outliers are removed by assigning them weight zero, then coefficients are recalculated on the remaining observations.
- The loop repeats until either:
  - the maximum error is acceptable, or
  - the number of remaining points becomes too small.

### Directional rejection matters

- The paper defines `SF` as a one-sided suppression flag: reject either high or low values.
- Error checking is done in the selected Hi/Lo direction.
- The paper's own example states that HANTS cannot reject outliers in the opposite direction of `SF`.

### Minimum observation logic

- The curve has `2 * NOF - 1` parameters.
- The number of valid observations must always be at least that many.
- `DOD` adds extra observations beyond the bare minimum to improve reliability.
- The final retained set should therefore be no smaller than `(2 * NOF - 1) + DOD`.

### Example values from the paper

- `NOF = 3` means zero frequency plus periods of 1 year and 6 months.
- With `NOF = 3`, the output has `5` Fourier coefficients.
- Example configuration in the paper:
  - `SF = low`
  - `IDRT = +0.7`
  - `FET = 0.05 NDVI units`
  - `DOD = 13`
- In that example, `DOD = 13` together with the minimum of `5` observations yields a minimum of `18` points in the final fit.

## Implementation Checks

Use these checks during code review:

1. Does `NOF` include the zero-frequency mean term?
2. Does the code treat model size as `2 * NOF - 1`?
3. Does `SF` control one-sided rejection rather than symmetric residual clipping?
4. Does `IDRT` exclude known-invalid values before or during fitting, instead of acting like a regularization parameter?
5. Does `FET` stop the loop based on the remaining points' absolute error in the selected direction?
6. Does the stopping logic prevent fitting with fewer than `(2 * NOF - 1) + DOD` valid observations?
7. Is the implementation still doing iterative refit-after-rejection, rather than replacing HANTS with a different robust solver?

## What The Paper Does Not Pin Down

- The paper explains that invalid points are assigned weight zero, but it does not fully prescribe one exact low-level data structure for weights versus filtered arrays.
- The paper describes repeated removal/reweighting of bad observations, but does not require one exact loop shape or memory layout.
- The paper explicitly says there are no objective rules for choosing control parameter magnitudes; tuning remains context dependent.

This means implementation details may change, but the semantics above should not.

## Common Mistakes

- Treating `NOF` as the number of non-zero harmonics only.
- Letting `SF` reject both high and low outliers by default.
- Forgetting that HANTS is directional and therefore may keep opposite-direction outliers.
- Using too few retained observations after rejection and still calling the fit paper-faithful.
- Replacing iterative rejection with a different robust regression and still calling it HANTS.
- Confusing `IDRT` with a generic residual threshold instead of a known-invalid value threshold.
