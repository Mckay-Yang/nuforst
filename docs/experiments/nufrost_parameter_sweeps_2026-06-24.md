# NUFROST Parameter Sweeps - 2026-06-24

This note records parameter-only tuning runs. No NUFROST method code was changed
for these sweeps.

Sample cache:

- `data/cache/sentinel-2/100k`
- seed: `20260609`
- `min_joint_valid=12`
- main validation size: `100000`

Tracked artifacts for plotting:

- `docs/experiments/nufrost_parameter_sweeps_2026-06-24_summary.csv`
- `docs/experiments/nufrost_parameter_sweeps_2026-06-24_best.json`

The summary CSV currently contains 361 valid parameter rows in the current
artifact set, including 271 20k-like rows, 46 100k-like rows, and 3 500k-like
rows. It is a normalized long table extracted from
`target/parameter_sweeps_5m/phase*_results.csv`.

## Previous 100k Baseline

The old same-seed baseline was:

- `normalization_mode=reflectance`
- `frequency_selection=all`
- `modes=192`
- `ridge=0.05`
- `freq_weight=72.0`
- `multiband_shrinkage=1.0`
- `huber_delta=0.14`
- `huber_iters=5`
- `outlier_reject_iters=0`
- 100k RMSE: `231.376435`
- 100k MAE: `101.616508`

## Dynamic Mode Count

Adaptive/dynamic mode-count selection was tested earlier in
`.worktrees/dynamic-frequency-selection`. It was not better for accuracy:

- best dynamic 100k RMSE: `231.761702`
- best dynamic 100k MAE: `101.948533`

Conclusion: dynamic mode count is not useful as an accuracy path. It may remain
an ablation or speed option, but the accuracy search should use fixed mode
counts.

## Phase 29

File:

- `target/parameter_sweeps_5m/phase29_dynamic_range_20k_results.csv`

Purpose:

- test fixed mode perturbations around `modes=192`;
- test high-mode `centered_reflectance` for dynamic-range behavior.

Best RMSE:

- `p29_reflectance_m208_r0p05_fw72_s1_h0p14_hi5`
- 20k RMSE: `221.782880`
- 20k MAE: `99.778630`

Best MAE:

- `p29_centered_m192_r0p05_fw64_s1_h0p14_hi5`
- 20k RMSE: `223.232133`
- 20k MAE: `96.696209`

Interpretation:

- fixed `modes=208` improves RMSE versus `modes=192`;
- `centered_reflectance` improves typical error and may look visually smoother,
  but it did not win on RMSE.

## Phase 30

File:

- `target/parameter_sweeps_5m/phase30_m208_focused_20k_results.csv`

Purpose:

- focus on `modes=208` and test `freq_weight`, `ridge`,
  `multiband_shrinkage`, and `huber_delta`.

Top 20k rows:

|name|RMSE|MAE|
|---|---:|---:|
|p30_reflectance_m208_r0p05_fw72_s1_h0p12_hi5|221.643777|99.529218|
|p30_reflectance_m208_r0p05_fw72_s0p75_h0p14_hi5|221.685865|99.657034|
|p30_reflectance_m208_r0p05_fw80_s1_h0p14_hi5|221.697444|99.806333|

Interpretation:

- lower `huber_delta=0.12` and weaker shrinkage can help;
- `freq_weight=80` is promising with `modes=208`.

## Phase 31

File:

- `target/parameter_sweeps_5m/phase31_m208_100k_results.csv`

Purpose:

- promote top phase 30 rows to 100k and rerun the old baseline.

Results:

|name|100k RMSE|100k MAE|
|---|---:|---:|
|p31_reflectance_m208_r0p05_fw80_s1_h0p14_hi5_100k|231.292065|101.442843|
|p31_reflectance_m208_r0p05_fw72_s0p75_h0p14_hi5_100k|231.337424|101.293795|
|p31_reflectance_m192_r0p05_fw72_s1_h0p14_hi5_baseline_100k|231.376435|101.616508|
|p31_reflectance_m208_r0p05_fw72_s1_h0p12_hi5_100k|231.536441|101.190159|

Interpretation:

- `modes=208, freq_weight=80` improved overall RMSE;
- the 100k baseline reproduced exactly, so the comparison is stable.

## Phase 32

File:

- `target/parameter_sweeps_5m/phase32_fw80_index_guard_20k_results.csv`

Purpose:

- test `fw80` interactions and guard against spectral-index RMSE outliers.

Top 20k rows:

|name|RMSE|MAE|index note|
|---|---:|---:|---|
|p32_reflectance_m208_r0p05_fw80_s0p75_h0p12_hi5|221.485768|99.446515|best overall, but NDMI RMSE outlier|
|p32_reflectance_m208_r0p05_fw80_s1_h0p12_hi5|221.562543|99.555333|more stable indices|
|p32_reflectance_m208_r0p05_fw80_s0p75_h0p14_hi5|221.619851|99.697749|NDMI elevated|
|p32_reflectance_m208_r0p045_fw80_s1_h0p14_hi5|221.623228|99.700262|stable indices|

Interpretation:

- `modes=216` and `freq_weight=88` worsened RMSE;
- the fixed mode-count optimum is near `208`.

## Phase 33

File:

- `target/parameter_sweeps_5m/phase33_fw80_index_guard_100k_results.csv`

Purpose:

- validate the strongest phase 32 rows on 100k.

Results:

|name|100k RMSE|100k MAE|index note|
|---|---:|---:|---|
|p33_reflectance_m208_r0p05_fw80_s0p75_h0p12_hi5_100k|231.182816|101.104107|best overall; NDMI/NDSI RMSE tail outliers|
|p33_reflectance_m208_r0p05_fw80_s1_h0p12_hi5_100k|231.333061|101.215752|more stable NDMI, elevated NDSI|
|p33_reflectance_m208_r0p045_fw80_s1_h0p14_hi5_100k|231.327010|101.344639|stable NDMI, elevated NDSI|

Current best overall parameters:

- `normalization_mode=reflectance`
- `frequency_selection=all`
- `modes=208`
- `ridge=0.05`
- `freq_weight=80.0`
- `multiband_shrinkage=0.75`
- `huber_delta=0.12`
- `huber_iters=5`
- `outlier_reject_iters=0`

Current best overall 100k metrics:

- RMSE: `231.182816`
- MAE: `101.104107`

Improvement over previous same-seed baseline:

- RMSE: `231.376435 -> 231.182816`
- MAE: `101.616508 -> 101.104107`

## Current Interpretation

The new best overall configuration improves both RMSE and MAE, but the
improvement is small. The target of RMSE and MAE below `200` is still not
achieved through parameter tuning.

Spectral-index MAE is generally stable, but spectral-index RMSE is sensitive to
rare samples where an index denominator is very small. This is most visible for
NDMI and NDSI in the best-overall phase 33 row. Since the method cannot be
changed under this tuning objective, index RMSE may not be a reliable tuning
target unless the evaluation protocol clips or masks unstable denominators.

## Phase 34

File:

- `target/parameter_sweeps_5m/phase34_best_basin_fine_20k_results.csv`

Purpose:

- fine tune mode count and `freq_weight` around the phase 33 best-overall
  basin.

Results:

|name|20k RMSE|20k MAE|
|---|---:|---:|
|p34_reflectance_m200_r0p05_fw80_s0p75_h0p12_hi5|221.262298|99.451785|
|p34_reflectance_m208_r0p05_fw84_s0p75_h0p12_hi5|221.492116|99.486440|
|p34_reflectance_m208_r0p05_fw76_s0p75_h0p12_hi5|221.501451|99.419284|
|p34_reflectance_m204_r0p05_fw80_s0p75_h0p12_hi5|221.883966|99.583703|
|p34_reflectance_m212_r0p05_fw80_s0p75_h0p12_hi5|221.886006|99.470241|

Interpretation:

- `modes=200` is better than `modes=208` on 20k under the current best
  regularization settings.
- The mode-count response is not monotonic, likely because the all-frequency
  grid changes with mode count.
- A finer local search around `modes=200` is needed before 100k promotion.

## Phase 35

File:

- `target/parameter_sweeps_5m/phase35_m200_fine_20k_results.csv`

Purpose:

- fine tune mode count around `modes=200` and compare nearby `freq_weight`
  values.

Results:

|name|20k RMSE|20k MAE|
|---|---:|---:|
|p34_reflectance_m200_r0p05_fw80_s0p75_h0p12_hi5|221.262298|99.451785|
|p35_reflectance_m200_r0p05_fw76_s0p75_h0p12_hi5|221.270913|99.427734|
|p35_reflectance_m200_r0p05_fw84_s0p75_h0p12_hi5|221.279076|99.486943|
|p35_reflectance_m202_r0p05_fw80_s0p75_h0p12_hi5|221.357252|99.345462|
|p35_reflectance_m198_r0p05_fw80_s0p75_h0p12_hi5|221.402365|99.508897|
|p35_reflectance_m196_r0p05_fw80_s0p75_h0p12_hi5|221.979740|99.633882|

Interpretation:

- `modes=200, freq_weight=80` remains the best 20k RMSE row.
- `freq_weight=76` is almost tied and has slightly lower MAE.
- `modes=202` has the lowest MAE in this local scan but worse RMSE.
- Promote these three candidates to 100k.

## Phase 36

File:

- `target/parameter_sweeps_5m/phase36_m200_100k_results.csv`

Purpose:

- validate the best phase 35 rows on 100k.

Results:

|name|100k RMSE|100k MAE|index note|
|---|---:|---:|---|
|p36_reflectance_m200_r0p05_fw80_s0p75_h0p12_hi5_100k|230.613347|101.094293|best overall RMSE; NDSI RMSE tail outlier|
|p36_reflectance_m200_r0p05_fw76_s0p75_h0p12_hi5_100k|230.654792|101.063674|best MAE; NDSI stable but NDVI worse|
|p36_reflectance_m202_r0p05_fw80_s0p75_h0p12_hi5_100k|230.761328|101.128255|slightly worse overall|

Current best overall parameters:

- `normalization_mode=reflectance`
- `frequency_selection=all`
- `modes=200`
- `ridge=0.05`
- `freq_weight=80.0`
- `multiband_shrinkage=0.75`
- `huber_delta=0.12`
- `huber_iters=5`
- `outlier_reject_iters=0`

Current best 100k metrics:

- RMSE: `230.613347`
- MAE: `101.094293`

Improvement over the previous `modes=192` same-seed baseline:

- RMSE: `231.376435 -> 230.613347`
- MAE: `101.616508 -> 101.094293`

Improvement over the phase 33 `modes=208` best:

- RMSE: `231.182816 -> 230.613347`
- MAE: `101.104107 -> 101.094293`

The main config files `config/nufrost.json` and
`config/nufrost_best_rmse.json` were updated to this `modes=200` parameter set.

## Phase 37

File:

- `target/parameter_sweeps_5m/phase37_m200_regularization_20k_results.csv`

Purpose:

- tune regularization around the phase 36 best `modes=200` configuration.

Results:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p37_reflectance_m200_r0p05_fw80_s0p75_h0p10_hi5|221.129172|99.123431|best RMSE and MAE|
|p37_reflectance_m200_r0p04_fw80_s0p75_h0p12_hi5|221.148982|99.272720|lower ridge improves RMSE|
|p37_reflectance_m200_r0p045_fw80_s0p75_h0p12_hi5|221.151061|99.338748|similar to ridge 0.04|
|p37_reflectance_m200_r0p05_fw80_s0p625_h0p12_hi5|221.253939|99.409564|weaker shrinkage close but not better|
|p37_reflectance_m200_r0p05_fw80_s0p5_h0p12_hi5|221.281628|99.386222|weaker shrinkage close but not better|

Interpretation:

- reducing `huber_delta` from `0.12` to `0.10` gives the strongest 20k gain;
- reducing `ridge` to `0.04-0.045` also helps;
- shrinkage is less important than Huber/ridge in this local basin.

## Phase 38

File:

- `target/parameter_sweeps_5m/phase38_m200_ridge_huber_20k_results.csv`

Purpose:

- test whether the lower ridge and lower Huber threshold improvements from
  phase 37 combine.

Results:

|name|20k RMSE|20k MAE|index note|
|---|---:|---:|---|
|p38_reflectance_m200_r0p04_fw80_s0p75_h0p08_hi5|220.835714|98.518205|best overall, moderate NDMI/NBR elevation|
|p38_reflectance_m200_r0p04_fw80_s0p75_h0p10_hi5|220.942623|98.934125|NDVI RMSE outlier|
|p38_reflectance_m200_r0p045_fw80_s0p75_h0p08_hi5|220.955939|98.598472|stable indices|
|p38_reflectance_m200_r0p045_fw80_s0p75_h0p10_hi5|220.979101|99.004051|stable indices|
|p38_reflectance_m200_r0p05_fw80_s0p75_h0p08_hi5|221.192273|98.728888|stable indices, worse RMSE|

Interpretation:

- `ridge=0.04, huber_delta=0.08` is the best 20k overall row so far;
- `ridge=0.045, huber_delta=0.08` is a safer index-stable alternative;
- promote both plus `ridge=0.05, huber_delta=0.08` to 100k.

## Phase 39

File:

- `target/parameter_sweeps_5m/phase39_ridge_huber_100k_results.csv`

Purpose:

- validate the low-ridge / low-Huber 20k candidates on 100k.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p39_reflectance_m200_r0p04_fw80_s0p75_h0p08_hi5_100k|230.902332|100.241966|best MAE, worse RMSE than phase 36 best|
|p39_reflectance_m200_r0p045_fw80_s0p75_h0p08_hi5_100k|230.932961|100.324712|worse RMSE/MAE than r0.04|
|p39_reflectance_m200_r0p05_fw80_s0p75_h0p08_hi5_100k|231.087986|100.459631|worse than r0.04|

Interpretation:

- `huber_delta=0.08` is a MAE-optimized direction, but it does not improve
  100k RMSE.
- Keep `config/nufrost.json` and `config/nufrost_best_rmse.json` at the phase
  36 RMSE optimum.
- Save the phase 39 MAE optimum as `config/nufrost_best_mae.json`.

## Phase 40

File:

- `target/parameter_sweeps_5m/phase40_convergence_probe_20k_results.csv`

Purpose:

- check whether the current basin still has obvious untested gains from nearby
  mode counts, Huber iteration count, or alternative normalization.

Results:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p40_reflectance_m200_r0p05_fw80_s0p75_h0p12_hi7|221.262298|99.451785|identical to 5 IRLS iterations|
|p40_reflectance_m200_r0p05_fw80_s0p75_h0p12_hi3|221.391680|99.637520|fewer IRLS iterations are worse|
|p40_reflectance_m206_r0p05_fw80_s0p75_h0p12_hi5|221.847625|99.532713|nearby mode count worse|
|p40_robust_m200_r0p05_fw80_s0p75_h0p12_hi5|222.241277|96.437260|low MAE, worse RMSE|
|p40_centered_m200_r0p05_fw80_s0p75_h0p12_hi5|222.241277|96.437260|same as robust in this path|
|p40_reflectance_m194_r0p05_fw80_s0p75_h0p12_hi5|222.405620|99.651522|nearby mode count worse|
|p40_reflectance_m192_r0p05_fw80_s0p75_h0p12_hi5|222.512910|99.826837|nearby mode count worse|

Interpretation:

- `huber_iters=5` is enough; `7` iterations produces the same output and `3`
  iterations is worse.
- nearby mode counts do not beat `modes=200`;
- `robust` and `centered_reflectance` remain MAE-oriented but worse for RMSE.
- no new 20k candidate from this convergence probe justifies 100k promotion.

## Phase 41

File:

- `target/parameter_sweeps_5m/phase41_outlier_reject_20k_results.csv`

Purpose:

- retest explicit outlier pre-rejection in the current `modes=200` basin.

Results:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p41_rmse_m200_r0p05_fw80_s0p75_h0p12_oit1_sig2p5_frac0p15|223.290677|98.211385|MAE lower, RMSE much worse|
|p41_rmse_m200_r0p05_fw80_s0p75_h0p12_oit1_sig3p0_frac0p15|223.290677|98.211385|same as sigma 2.5|
|p41_mae_m200_r0p04_fw80_s0p75_h0p08_oit1_sig3p0_frac0p15|223.405911|97.147581|lowest 20k MAE, RMSE worse|
|p41_mae_m200_r0p04_fw80_s0p75_h0p08_oit1_sig2p5_frac0p15|223.406116|97.147903|same as sigma 3.0|
|p41_rmse_m200_r0p05_fw80_s0p75_h0p12_oit2_sig2p5_frac0p10|224.483788|98.170954|two rejection rounds worse|
|p41_rmse_m200_r0p05_fw80_s0p75_h0p12_oit2_sig3p0_frac0p10|224.483788|98.170954|same as sigma 2.5|

Interpretation:

- explicit outlier pre-rejection is not an RMSE path in this basin;
- it lowers MAE but substantially increases RMSE;
- no phase 41 candidate should replace the main RMSE config or be promoted to
  100k for the current objective.

## Phase 42

File:

- `target/parameter_sweeps_5m/phase42_lambda_high_20k_results.csv`

Purpose:

- test the high-frequency tiered ridge parameter `lambda_high` in the current
  `modes=200` RMSE basin. This term only adds extra high-frequency ridge when
  `lambda_high > ridge`.

Results:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p15_lfp60|221.243935|99.439265|best, tiny improvement|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p10_lfp60|221.253098|99.445517|tiny improvement|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p08_lfp90|221.253744|99.446827|tiny improvement|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p08_lfp60|221.256774|99.448022|tiny improvement|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p06_lfp60|221.260455|99.450530|near baseline|
|p42_m200_r0p05_fw80_s0p75_h0p12_lh0p08_lfp30|221.262297|same as baseline|

Interpretation:

- additional high-frequency tiered ridge gives only a very small 20k gain;
- not enough evidence to promote to 100k by itself.

## Phase 43

File:

- `target/parameter_sweeps_5m/phase43_lambda_step_20k_results.csv`

Purpose:

- test whether the step/fused residual parameters affect the current
  sample-cache evaluation path.

Results:

All tested rows produced the same 20k metrics:

- RMSE: `221.262298`
- MAE: `99.451785`

Rows:

- `lambda_step=0.0, max_outer_iter=1`
- `lambda_step=0.01, max_outer_iter=5`
- `lambda_step=0.1, max_outer_iter=5`
- `lambda_step=0.5, max_outer_iter=5`
- `lambda_step=1e30, max_outer_iter=1`

Interpretation:

- `lambda_step` and `max_outer_iter` do not affect the current sample-cache
  evaluation output in this configuration;
- do not spend more parameter-search time on this axis for the current goal.

## Phase 44

File:

- `target/parameter_sweeps_5m/phase44_frequency_selection_20k_results.csv`

Purpose:

- retest sparse frequency-selection strategies under the current best
  `modes=200` regularization settings.

Results:

|name|20k RMSE|20k MAE|
|---|---:|---:|
|p44_preferred_m200_pref4_r0p05_fw80_s0p75_h0p12|387.383547|178.675065|
|p44_hybrid_m200_pref4_st16_r0p05_fw80_s0p75_h0p12|389.822880|179.535269|
|p44_spectral_m200_np20_st20_r0p05_fw80_s0p75_h0p12|398.861337|184.067696|
|p44_hybrid_m200_pref4_st32_r0p05_fw80_s0p75_h0p12|406.760743|186.278656|
|p44_hybrid_m200_pref8_st40_r0p05_fw80_s0p75_h0p12|411.193700|188.520482|
|p44_spectral_m200_np40_st40_r0p05_fw80_s0p75_h0p12|412.111275|188.769156|

Interpretation:

- sparse frequency selection is far worse than `frequency_selection=all`;
- keep the main config at all-frequency fitting;
- this also supports the earlier dynamic-mode conclusion: reducing the number
  of active frequency components is not an accuracy path for the current
  sample-cache objective.

## Phase 45

File:

- `target/parameter_sweeps_5m/phase45_final_100k_results.csv`

Purpose:

- promote the small phase 42 `lambda_high` signal and nearby `freq_weight=84`
  candidate to 100k.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p45_m200_r0p05_fw84_s0p75_h0p12_lh0p15_lfp60_100k|230.580501|101.124976|new best RMSE|
|p45_m200_r0p05_fw80_s0p75_h0p12_lh0p15_lfp60_100k|230.593819|101.082369|slightly worse RMSE, better MAE|
|p45_m200_r0p05_fw84_s0p75_h0p12_100k|230.598807|101.136241|small gain from fw84 alone|
|p45_m200_r0p05_fw76_s0p75_h0p12_lh0p15_lfp60_100k|230.633788|101.051043|best MAE in this phase, worse RMSE|

Current best overall RMSE parameters:

- `normalization_mode=reflectance`
- `frequency_selection=all`
- `modes=200`
- `ridge=0.05`
- `freq_weight=84.0`
- `multiband_shrinkage=0.75`
- `huber_delta=0.12`
- `huber_iters=5`
- `outlier_reject_iters=0`
- `lambda_high=0.15`
- `low_freq_period_days=60.0`

Current best 100k metrics:

- RMSE: `230.580501`
- MAE: `101.124976`

This improves the previous phase 36 best (`230.613347`, `101.094293`) by
`0.032846` RMSE, but slightly worsens MAE. The main config files
`config/nufrost.json` and `config/nufrost_best_rmse.json` were updated to this
new best-RMSE parameter set.

## Phase 46

File:

- `target/parameter_sweeps_5m/phase46_phenology_trend_20k_results.csv`

Purpose:

- test whether the phenology-frequency ridge exemption and trend term are still
  necessary in the current phase 45 RMSE basin.

Results:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p46_m200_fw84_lh0p15_pref2_trend1|221.995060|98.385676|lower MAE, worse RMSE|
|p46_m200_fw84_lh0p15_pref4_trend0|227.028964|102.569329|removing trend is worse|
|p46_m200_fw84_lh0p15_pref0_trend1|238.661437|106.413281|removing phenology exemption is worse|
|p46_m200_fw84_lh0p15_pref0_trend0|243.493342|109.062670|removing both is worse|
|p46_m200_fw84_lh0p15_pref6_trend1|278.175825|105.724199|too many unpenalized preferred periods is much worse|

Interpretation:

- keep the default four preferred periods and `include_trend=true`;
- too few preferred periods lowers MAE but hurts RMSE;
- too many unpenalized preferred periods badly destabilize RMSE;
- no phase 46 candidate should replace the main RMSE config.

## Phase 47

File:

- `target/parameter_sweeps_5m/phase47_current_best_500k_results.csv`

Purpose:

- validate the current best-RMSE parameter set on 500k samples from the 5M
  sample cache.

Parameters:

- `normalization_mode=reflectance`
- `frequency_selection=all`
- `modes=200`
- `ridge=0.05`
- `freq_weight=84.0`
- `multiband_shrinkage=0.75`
- `huber_delta=0.12`
- `huber_iters=5`
- `outlier_reject_iters=0`
- `lambda_high=0.15`
- `low_freq_period_days=60.0`
- `include_trend=true`
- `preferred_top_k=4`

Result:

- evaluated: `499771`
- skipped: `229`
- elapsed: `2439.2s`
- RMSE: `229.569462`
- MAE: `100.964803`

Interpretation:

- the current best-RMSE parameter set remains stable when moving from 100k to
  500k samples;
- the 500k RMSE is slightly lower than the same candidate's 100k RMSE
  (`230.580501`), while MAE remains around `101`;
- this still does not reach the requested RMSE/MAE target of `200`, so the goal
  remains incomplete under parameter-only tuning.

## Phase 48

File:

- `target/parameter_sweeps_5m/phase48_low_ridge_high_penalty_20k_results.csv`

Purpose:

- test whether the strong 20k basin at `ridge=0.04`, `huber_delta=0.08`,
  `freq_weight=80` can be improved by adding the phase45 high-frequency tier
  penalty;
- probe neighboring mode counts around `modes=200`.

Best rows:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p48_m200_r0p04_fw88_s0p75_h0p08_lh0p15|220.770309|98.500345|best RMSE in phase48|
|p48_m200_r0p04_fw80_s0p75_h0p08_lh0p3|220.774358|98.477331|best MAE among top RMSE rows|
|p48_m200_r0p04_fw84_s0p75_h0p08_lh0p15|220.774441|98.495069|middle freq-weight check|

Interpretation:

- low ridge and low Huber threshold remain a strong 20k MAE/RMSE basin;
- `freq_weight=88` gives the best phase48 20k RMSE, but the gain over the
  previous 20k best (`220.835714`) is only `0.065405`;
- neighboring mode counts `196`, `204`, and `208` are worse than `200`, so
  `modes=200` remains the local optimum.

## Phase 49

File:

- `target/parameter_sweeps_5m/phase49_low_ridge_high_penalty_100k_results.csv`

Purpose:

- promote the best phase48 20k candidates to 100k validation.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p49_m200_r0p04_fw88_s0p75_h0p08_lh0p15_100k|230.749841|100.232375|best RMSE in phase49|
|p49_m200_r0p04_fw80_s0p75_h0p08_lh0p3_100k|230.837121|100.202781|best MAE in phase49|
|p49_m200_r0p04_fw84_s0p75_h0p08_lh0p15_100k|230.797627|100.223303|middle freq-weight check|

Interpretation:

- the phase48 20k improvement does not transfer to 100k RMSE;
- phase49 improves MAE versus the current best-RMSE config, but worsens RMSE;
- at this point the main RMSE config remained
  `p45_m200_r0p05_fw84_s0p75_h0p12_lh0p15_lfp60_100k` with RMSE `230.580501`;
- this supports the current tradeoff: `ridge=0.04, huber_delta=0.08` is an MAE
  route, while `ridge=0.05, huber_delta=0.12` is still better for RMSE.

## Phase 50

File:

- `target/parameter_sweeps_5m/phase50_current_basin_fw_lh_20k_results.csv`

Purpose:

- check whether the current best-RMSE basin (`ridge=0.05`, `huber_delta=0.12`)
  can be improved by pushing `freq_weight`, `lambda_high`, or
  `low_freq_period_days`.

Best rows:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p50_m200_r0p05_fw84_s0p75_h0p12_lh0p2|221.253532|99.469268|best phase50 RMSE|
|p50_m200_r0p05_fw84_s0p75_h0p12_lh0p15|221.262013|99.475145|current 100k basin reference|
|p50_m200_r0p05_fw88_s0p75_h0p12_lh0p2|221.294518|99.515423|higher `freq_weight` worsens|
|p50_m200_r0p05_fw88_s0p75_h0p12_lh0p15_lfp75|221.297182|99.519436|changing low-frequency threshold does not help|

Interpretation:

- `lambda_high=0.20` gives a tiny 20k improvement over `0.15`, but the absolute
  gain is only `0.008482` RMSE;
- increasing `freq_weight` above `84` worsens the current RMSE basin;
- no phase50 row is strong enough to promote to 100k validation.

## Phase 51

File:

- `target/parameter_sweeps_5m/phase51_norm_basin_20k_results.csv`

Purpose:

- test whether the underexplored `centered_reflectance` and `robust`
  normalization basins can improve RMSE when combined with the high-frequency
  tier penalty;
- check whether their known lower MAE can be retained at higher mode counts.

Best rows:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p51_centered_m208_r0p05_fw64_s0p75_h0p12_lh0p15|220.744179|96.106830|best phase51 RMSE and new 20k best|
|p51_centered_m200_r0p05_fw64_s0p75_h0p12_lh0p15|221.032842|96.159359|centered m200 check|
|p51_robust_m200_r0p05_fw64_s1_h0p14_lh0p15|221.550106|96.339409|best robust row in phase51|

Interpretation:

- `centered_reflectance` has the best 20k MAE basin and briefly gives the best
  20k RMSE seen so far;
- the low MAE is real, but the 20k RMSE gain is still small enough that 100k
  validation is required;
- `robust` with high-frequency tiering improves over some old robust settings
  but does not beat the centered row.

## Phase 52

File:

- `target/parameter_sweeps_5m/phase52_norm_basin_100k_results.csv`

Purpose:

- promote the best phase51 centered/robust rows to 100k validation.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p52_centered_m200_r0p05_fw64_s0p75_h0p12_lh0p15_100k|232.795602|97.979899|new best 100k MAE|
|p52_centered_m208_r0p05_fw64_s0p75_h0p12_lh0p15_100k|232.955571|97.995404|20k best does not transfer to RMSE|
|p52_robust_m200_r0p05_fw64_s1_h0p14_lh0p15_100k|233.318264|98.157121|robust remains worse on RMSE|

Interpretation:

- `centered_reflectance` is now the best 100k MAE route, but it worsens RMSE by
  more than `2.2` relative to the main RMSE config;
- centered/robust normalization does not help the primary RMSE objective;
- this confirms two distinct parameter regimes: reflectance scaling for RMSE,
  centered/robust scaling for MAE and smoother visual behavior.

## Phase 53

File:

- `target/parameter_sweeps_5m/phase53_shrinkage_neighborhood_20k_results.csv`

Purpose:

- test `multiband_shrinkage` around both the current RMSE basin
  (`ridge=0.05`, `freq_weight=84`, `huber_delta=0.12`) and the lower-MAE basin
  (`ridge=0.04`, `freq_weight=88`, `huber_delta=0.08`).

Best rows:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p53_maebasin_s0p625|220.739646|98.447234|new 20k RMSE best|
|p53_maebasin_s0p5|220.748035|98.413360|slightly lower MAE, slightly worse RMSE|
|p53_rmsebasin_s0p625|221.257690|99.436426|best current-basin shrinkage row|
|p53_rmsebasin_s0p75|221.262013|99.475145|old main shrinkage reference|

Interpretation:

- `multiband_shrinkage=0.625` is better than `0.75` in both tested basins;
- strong shrinkage (`>=1.0`) worsens both RMSE and MAE;
- the 20k gain is small, but the direction is consistent enough to promote
  `s=0.625` to 100k validation.

## Phase 54

File:

- `target/parameter_sweeps_5m/phase54_shrinkage_100k_results.csv`

Purpose:

- validate the best phase53 shrinkage candidates on 100k samples.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p54_rmsebasin_m200_r0p05_fw84_s0p625_h0p12_lh0p15_100k|230.535757|101.087882|new best 100k RMSE|
|p54_maebasin_m200_r0p04_fw88_s0p625_h0p08_lh0p15_100k|230.681096|100.184219|better MAE, worse RMSE|

Interpretation:

- `multiband_shrinkage=0.625` gives a small but real 100k RMSE improvement over
  the previous main config (`230.580501`);
- the main config files `config/nufrost.json` and `config/nufrost_best_rmse.json`
  were updated to `multiband_shrinkage=0.625`;
- the result still does not approach the requested RMSE/MAE target of `200`, but
  it is the current best same-seed 100k RMSE.

## Phase 55

File:

- `target/parameter_sweeps_5m/phase55_current_best_500k_results.csv`

Purpose:

- validate the new phase54 best-RMSE configuration on 500k samples.

Result:

- evaluated: `499771`
- skipped: `229`
- elapsed: `2418.7s`
- RMSE: `229.524675`
- MAE: `100.930893`

Interpretation:

- the `multiband_shrinkage=0.625` update transfers from 100k to 500k;
- the 500k RMSE improves from the old phase47 value `229.569462` to
  `229.524675`;
- this is the current best large-sample RMSE, but it still does not approach the
  requested RMSE/MAE target of `200`.

## Phase 56

Files:

- `target/parameter_sweeps_5m/phase56_tiny_neighborhood_20k_results.csv`
- `target/parameter_sweeps_5m/phase56b_lowridge_retry_20k_results.csv`

Purpose:

- run a tight neighborhood search around the current RMSE basin and the low
  ridge/low Huber basin;
- retry the low-ridge block after a transient sample-cache read failure.

Best rows:

|name|20k RMSE|20k MAE|note|
|---|---:|---:|---|
|p56b_lowridge_fw86_s0p55_lh0p15|220.736069|98.417799|new 20k RMSE best|
|p56b_lowridge_fw86_s0p625_lh0p15|220.737563|98.442062|near tie|
|p56_rmse_fw80_s0p625_lh0p2|221.226615|99.390794|best current-basin 20k row|
|p56_rmse_fw84_s0p625_lh0p2|221.249258|99.430534|current 100k basin with higher `lambda_high`|

Interpretation:

- low ridge can keep improving 20k RMSE by tiny amounts, but prior 100k
  validations show this basin does not beat the main RMSE config;
- `lambda_high=0.20` gives a tiny but consistent improvement within the current
  RMSE basin, so the best current-basin variants were promoted to 100k.

## Phase 57

File:

- `target/parameter_sweeps_5m/phase57_tiny_neighborhood_100k_results.csv`

Purpose:

- validate the best phase56 low-ridge row and the current-basin
  `lambda_high=0.20` candidates.

Results:

|name|100k RMSE|100k MAE|note|
|---|---:|---:|---|
|p57_rmse_m200_r0p05_fw84_s0p625_h0p12_lh0p2_100k|230.526678|101.082281|new best 100k RMSE|
|p57_rmse_m200_r0p05_fw80_s0p625_h0p12_lh0p2_100k|230.535890|101.037152|near tie with slightly lower MAE|
|p57_lowridge_m200_r0p04_fw86_s0p55_h0p08_lh0p15_100k|230.670306|100.154625|low-ridge still worse on RMSE|

Interpretation:

- `lambda_high=0.20` improves the 100k RMSE by `0.009079` over phase54;
- the main config files were updated from `lambda_high=0.15` to `0.20`;
- this is another small same-basin gain, not a path toward RMSE below `200`.

## Phase 58

File:

- `target/parameter_sweeps_5m/phase58_current_best_500k_results.csv`

Purpose:

- validate the phase57 `lambda_high=0.20` update on 500k samples.

Result:

- evaluated: `499771`
- skipped: `229`
- elapsed: `2560.2s`
- RMSE: `229.516106`
- MAE: `100.925330`

Interpretation:

- the phase57 `lambda_high=0.20` update transfers from 100k to 500k;
- 500k RMSE improves from phase55 `229.524675` to `229.516106`;
- the improvement is real but extremely small, confirming that the current
  basin is close to saturated under parameter-only tuning.

## Spectral Index Status

Same-seed 100k comparison methods:

|method|overall RMSE|overall MAE|
|---|---:|---:|
|NUFROST phase57 best RMSE|230.526678|101.082281|
|Zhu2015|544.749065|264.654682|
|HANTS|716.024259|282.380616|

The phase57 best-RMSE NUFROST index RMSE values are:

|index|NUFROST RMSE|Zhu2015 RMSE|HANTS RMSE|
|---|---:|---:|---:|
|NDVI|0.232376|0.130848|0.362073|
|NDWI|0.075858|0.107953|3.251308|
|NDMI|0.226729|0.152882|0.876934|
|NDSI|0.144153|0.173469|1.019298|
|NBR|0.114675|0.163179|0.236983|
|EVI|2.563005|2.448048|2.445812|

The same current best-RMSE NUFROST config on the phase58 500k validation gives:

|index|500k NUFROST RMSE|500k NUFROST MAE|
|---|---:|---:|
|NDVI|0.252324|0.033676|
|NDWI|0.375413|0.030186|
|NDMI|0.317259|0.037836|
|NDSI|0.261626|0.039151|
|NBR|10.477903|0.055286|
|EVI|5.336630|0.044637|

Interpretation:

- overall band RMSE/MAE is much better than both comparison methods;
- NUFROST is better on NDWI, NDSI, and NBR relative to Zhu2015, and better on
  most indices relative to HANTS in the 100k check;
- the 500k index RMSE values are less favorable for NDWI, NDSI, NBR, and EVI,
  so the 100k index improvements should not be treated as stable enough for the
  main claim;
- phase58 improves overall RMSE but makes NBR/EVI RMSE worse, confirming that
  overall band RMSE and ratio-index RMSE are not aligned objectives here;
- NDVI, NDMI, and EVI RMSE are not better than Zhu2015 for the current
  best-RMSE config;
- the requested "one quarter of comparison-method index RMSE" is not achieved;
- EVI is a shared hard case across all methods and is dominated by denominator
  instability in the current index definition.

Current conclusion: without changing the evaluation definition or the method,
parameter tuning alone is unlikely to make all spectral-index RMSE values reach
one quarter of the comparison methods while preserving the best overall RMSE.
