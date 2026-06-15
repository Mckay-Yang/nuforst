# NUFROST Experiments Skill

Use this skill when planning or running NUFROST parameter experiments,
full-scene evaluations, sample-cache evaluations, or paper-result checks.

## Preferred Evaluation Order

Use the global sample cache for parameter decisions whenever possible. A single
scene can be visually important but may be an extreme case.

Recommended sequence:

1. Run a small smoke test on committed `tests/data`.
2. Run a small sample-cache evaluation, for example `--n-eval 10000`.
3. Run a larger sample-cache evaluation, ideally `--n-eval 1000000`.
4. Run selected full-scene tests for visual inspection.
5. Only then update default config.

## Metrics

For reconstruction comparisons, track at least:

- full-band RMSE and MAE,
- per-band RMSE and MAE,
- NDVI, NDWI, NDSI, NDMI, NBR, and optionally EVI RMSE/MAE,
- paired deltas when comparing two NUFROST variants,
- per-region summaries when full-scene outputs exist.

Reflectance-scale metrics are usually reported in Sentinel-2 scaled DN units.
For paper figures, normalized reflectance units may be clearer:

```text
normalized value = scaled DN / 10000
```

## Current NUFROST Defaults To Remember

Current `config/nufrost.json` uses:

- `frequency_selection = all`
- `modes = 64`
- preferred periods: annual, semiannual, quarterly, monthly-like
- `ridge = 2.0`
- `freq_weight = 256.0`
- `multiband_shrinkage = 1.0`
- `huber_iters = 3`
- `joint_outlier = true`
- `outlier_reject_iters = 2`
- `min_obs = 12`

## Interpreting Results

Do not rely only on one RMSE number. Check whether improvements are:

- stable over 1M samples,
- stable across bands,
- stable across indices,
- not caused by one region,
- not trading lower full-band RMSE for worse vegetation/water/snow indices.

For candidate method changes, keep default config unchanged until the sample
cache result is convincing.
