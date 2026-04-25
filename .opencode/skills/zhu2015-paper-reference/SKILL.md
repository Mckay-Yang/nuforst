---
name: zhu2015-paper-reference
description: Use when implementing, reviewing, debugging, or optimizing the Zhu et al. 2015 synthetic Landsat method and needing the paper-faithful model selection, break detection, backup rules, or QA band semantics.
---

# Zhu2015 Paper Reference

## Overview

This skill condenses Zhu et al. (2015) into implementation checks. Use it to keep the repository's `Zhu2015` baseline aligned with the paper instead of drifting into a simplified harmonic regression.

Primary source in this repo:

- `docs/reference/Zhu 等 - 2015 - Generating synthetic Landsat images based on all available Landsat data Predicting Landsat surface.pdf`

## When to Use

- Checking whether the `Zhu2015` implementation is still paper-faithful
- Reviewing code that changes model order selection, break detection, or QA output
- Explaining why Zhu2015 is more than a single LASSO fit
- Debugging `12/18/24` thresholds, `6 consecutive observations`, or temporally adjusted RMSE
- Optimizing Zhu2015 without silently removing paper-required behavior

Do not use this skill for generic Landsat reconstruction advice. It is specifically about Zhu et al. (2015).

## Quick Reference

| Topic | Paper rule | Implementation consequence |
|---|---|---|
| Model family | `simple`, `advanced`, `full` harmonic + trend models | Different coefficient counts and harmonics must be preserved |
| Model selection | Depends on count of clear observations | `12-17 -> simple`, `18-23 -> advanced`, `>=24 -> full` |
| Regression | Use `LASSO`, not OLS, for all time series models | Do not silently swap to OLS as the production path |
| Break detection | Difference larger than `2 * RMSE` for `6 consecutive observations` | Break logic must preserve both threshold and run length |
| Temporally adjusted RMSE | Use nearest `24` observations by day-of-year when clear obs > 24 | RMSE must vary by season, not be one global scalar |
| QA band | Encodes how the model was estimated and how it was used | Preserve both unit digit and tens digit semantics |

## Paper Constraints

### Three model forms

- `simple model`: 4 coefficients
  - overall value
  - annual cosine
  - annual sine
  - linear trend
- `advanced model`: simple model plus bimodal terms
  - adds `cos(4πx/T)` and `sin(4πx/T)`
- `full model`: advanced model plus trimodal terms
  - adds `cos(6πx/T)` and `sin(6πx/T)`

The paper treats these as harmonic models plus a long-term trend component.

### Model selection by clear-observation count

- `12 <= clear_obs < 18` -> `simple`
- `18 <= clear_obs < 24` -> `advanced`
- `clear_obs >= 24` -> `full`

The paper motivates this with the rule that clear observations should exceed three times the number of coefficients for robust estimation.

### LASSO is part of the method

- The paper explicitly switches from OLS to `LASSO` to reduce overfitting, especially for advanced and full models.
- A paper-faithful implementation should therefore keep LASSO as the estimation method for the time series models.

### Break detection

- Breaks are detected by comparing predictions with real Landsat observations.
- A break occurs when the difference is larger than `2 * RMSE` for `6 consecutive observations`.
- In the paper, this difference is the gap between model-predicted reflectance and the real clear observation, and the threshold is the predicted value plus/minus `2 * temporally-adjusted RMSE`.
- The paper states three improvements relative to CCDC:
  - lower threshold: `2 * RMSE` instead of `3 * RMSE`
  - require `6 consecutive observations` instead of `3`
  - exclude the blue band and thermal band from change detection; use Bands `2, 3, 4, 5, 7`

### Temporally adjusted RMSE

- RMSE is adjusted through time because seasonal variance differs.
- When clear observations exceed `24`, the paper uses the nearest `24` observations in day-of-year space to compute RMSE for thresholding.
- This seasonal adjustment is part of the break-detection logic, not an optional visualization detail.

## Backup Algorithms

- If clear observations are between `6` and `11`, use the `simple` model as a backup and do not use it for change detection.
- If clear observations are between `1` and `5`, use the median of all clear observations.
- For perennial snow pixels:
  - estimate snow-covered reflectance using all available unsaturated snow observations with the `simple` model
  - if unsaturated snow observations are fewer than `12`, use value `1` as surface reflectance
- Backup models for `<12` clear observations or perennial snow are for synthetic image generation only, not abrupt change detection.

## QA Band Semantics

The paper defines a two-digit QA code.

### Unit digit: how the time series model was estimated

- `0`: number of clear observations `>= 12`
- `1`: `6 <= number of clear observations < 12`
- `2`: number of clear observations `< 6`
- `3`: permanent snow pixel

### Tens digit: how the time series model was used

- `0`: synthetic data generated within the time range of the time series model
- `1`: generated by projecting the next time series model backward in time
- `2`: generated by projecting the previous time series model forward in time

Example from the paper: `QA = 10` means backward projection with a model estimated from `>=12` clear observations.

## Implementation Checks

Use these checks during code review:

1. Are the simple/advanced/full model equations still harmonic-plus-trend, with the expected annual, biannual, and triannual terms?
2. Does model selection still use the exact `12/18/24` observation thresholds?
3. Is estimation still based on `LASSO`, not quietly replaced by OLS or ridge?
4. Does break detection still require both `2 * RMSE` and `6 consecutive observations`?
5. Does change detection exclude blue and thermal bands?
6. Is RMSE seasonally adjusted using nearest day-of-year neighbors when the series is long enough?
7. Are backup paths for `<12` clear observations and perennial snow preserved?
8. Does the QA band still encode both model-estimation quality and temporal projection mode?

## What The Paper Leaves Flexible

- The paper defines the algorithmic rules, but not one exact software architecture.
- It does not mandate one specific optimization strategy, cache layout, or parallelization shape.
- It motivates using all available Landsat data and at least one model per pixel, but implementation details around bookkeeping can vary.

Optimizations are acceptable only if these paper-level constraints remain true.

## Common Mistakes

- Reducing Zhu2015 to a single LASSO fit with no break handling.
- Forgetting the `12/18/24` thresholds and choosing model order with different ad hoc rules.
- Using OLS as the production method even though the paper explicitly adopts LASSO.
- Replacing temporally adjusted RMSE with one global RMSE for the whole series.
- Dropping the `6 consecutive observations` requirement.
- Treating QA as an opaque extra band instead of preserving its documented digits.
- Using backup models for change detection, even though the paper says they are only for synthetic image generation.
