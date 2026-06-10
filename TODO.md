# Project TODO

## Reconstruction Diagnostics

1. Compute signed residual images for every fitted reconstruction:
   `prediction - ground_truth`.
2. Design a cache format for randomly sampled time series:
   sample 1,000,000 sequences across scenes, then support repeated synthetic
   pixel removals per sequence for NUFROST, HANTS, and Zhu2015 gap experiments.
3. Verify whether the current NUFROST sinc-based frequency matching/fitting path
   is enabled in the default workflow and actually affects selected frequencies.
4. Optimize `build-sample-cache` writing before rebuilding the 1,000,000-sequence
   sample cache:
   reuse per-sample buffers, write sample and mask blocks in larger chunks,
   flush or checkpoint index progress, and write into a temporary output
   directory before atomically replacing the final cache to avoid half-built
   `samples.f32.bin` / `index.u64.bin` outputs.
