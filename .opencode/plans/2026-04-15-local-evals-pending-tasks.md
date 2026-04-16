# Local Evals Pending Tasks

> Updated 2026-04-16

---

### 1. ~~验证 summarize_gap_candidates hot-fix~~ ✅ RESOLVED
- Replaced by `scan_pixel_stats_from_source` + `scan_gap_candidates_from_source` batch approach
- No longer uses per-pixel `read_pixel_series`

### 2. ~~排查 random-point pool 只得 1 sample 的异常~~ ✅ RESOLVED
- Root cause: `int(weights.sum())` = 1 on normalized weights
- Fixed: changed to `min(num_points, len(valid_pixels))`

### 3. ~~summarize_gap_candidates() 性能优化~~ ✅ RESOLVED
- Replaced by batch window scan; 8 min vs 1+ hour

### 4. GAP_MAX_NATIVE_GAP_DAYS 调整 ⬜ TODO
- Changed from 20 → 60 days in notebook
- 65/67 chunks were filtered to 0 with 20-day limit
- Need to re-run experiments after parameter change

### 5. 增强断点续跑 ⬜ TODO
- Random-point pool and gap-pixel pool caching not yet implemented
- Medium priority — experiments can run without it

### 6. 阶段性日志 ✅ PARTIAL
- Pixel stats scan has progress logging
- Gap candidate and random-point sampling still lack progress output