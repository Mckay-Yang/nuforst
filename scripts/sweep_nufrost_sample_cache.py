#!/usr/bin/env python3
"""Run NUFROST parameter sweeps on a sample cache and collect metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path


def fmt_value(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def candidate_name(params: dict) -> str:
    parts = [
        params["normalization_mode"],
        params["frequency_selection"],
        f"m{params['modes']}",
        f"r{fmt_value(params['ridge'])}",
        f"fw{fmt_value(params['freq_weight'])}",
        f"s{fmt_value(params['multiband_shrinkage'])}",
        f"h{fmt_value(params['huber_delta'])}",
        f"oit{params['outlier_reject_iters']}",
    ]
    if "lambda_high" in params:
        parts.append(f"lh{fmt_value(params['lambda_high'])}")
    if "low_freq_period_days" in params:
        parts.append(f"lfp{fmt_value(params['low_freq_period_days'])}")
    if "include_trend" in params:
        parts.append(f"trend{int(bool(params['include_trend']))}")
    if "huber_iters" in params:
        parts.append(f"hi{params['huber_iters']}")
    return "_".join(parts)


def build_candidates(args: argparse.Namespace) -> list[dict]:
    if args.candidate_file is not None:
        payload = read_json(args.candidate_file)
        if not isinstance(payload, list):
            raise ValueError("--candidate-file must contain a JSON array")
        candidates = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("each candidate must be a JSON object")
            params = dict(item)
            params.setdefault("normalization_mode", "reflectance")
            params.setdefault("frequency_selection", "all")
            params.setdefault("outlier_reject_iters", 2)
            params.setdefault("outlier_reject_sigma", 2.5)
            params.setdefault("name", candidate_name(params))
            candidates.append(params)
        return candidates

    keys = [
        "normalization_mode",
        "frequency_selection",
        "modes",
        "ridge",
        "freq_weight",
        "multiband_shrinkage",
        "huber_delta",
        "outlier_reject_iters",
        "outlier_reject_sigma",
    ]
    values = [
        args.normalization_mode,
        args.frequency_selection,
        args.modes,
        args.ridge,
        args.freq_weight,
        args.multiband_shrinkage,
        args.huber_delta,
        args.outlier_reject_iters,
        args.outlier_reject_sigma,
    ]
    optional_grid = [
        ("lambda_high", args.lambda_high),
        ("low_freq_period_days", args.low_freq_period_days),
        ("include_trend", args.include_trend),
        ("huber_iters", args.huber_iters_grid),
    ]
    for key, value in optional_grid:
        if value is not None:
            keys.append(key)
            values.append(value)
    candidates = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["name"] = candidate_name(params)
        candidates.append(params)
    return candidates


def run_candidate(
    *,
    cli: Path,
    base_config: dict,
    params: dict,
    cache_dir: Path,
    out_dir: Path,
    n_eval: int,
    seed: int,
    min_joint_valid: int,
    threads: int | None,
    dyld_library_path: str | None,
) -> dict:
    config = dict(base_config)
    config.update({k: v for k, v in params.items() if k != "name"})

    config_path = out_dir / "configs" / f"{params['name']}_{n_eval}.json"
    json_path = out_dir / "json" / f"{params['name']}_{n_eval}.json"
    log_path = out_dir / "logs" / f"{params['name']}_{n_eval}.log"
    write_json(config_path, config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cli),
        "eval-sample-cache",
        "--method",
        "nufrost",
        "--cache-dir",
        str(cache_dir),
        "--n-eval",
        str(n_eval),
        "--seed",
        str(seed),
        "--min-joint-valid",
        str(min_joint_valid),
        "--config",
        str(config_path),
        "--output-json",
        str(json_path),
    ]
    if threads is not None:
        cmd.extend(["--threads", str(threads)])

    started = time.monotonic()
    env = os.environ.copy()
    if dyld_library_path:
        env["DYLD_LIBRARY_PATH"] = dyld_library_path
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        return {
            **params,
            "status": f"failed:{completed.returncode}",
            "elapsed_seconds": elapsed,
            "rmse": "",
            "mae": "",
        }

    summary = read_json(json_path)
    overall = summary.get("overall", {})
    indices = summary.get("indices", {})
    row = {
        **params,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "evaluated": summary.get("evaluated", ""),
        "skipped": summary.get("skipped", ""),
        "rmse": overall.get("rmse", ""),
        "mae": overall.get("mae", ""),
    }
    for index_name in ["NDVI", "NDWI", "NDMI", "NDSI", "NBR", "EVI"]:
        metrics = indices.get(index_name, {})
        row[f"{index_name.lower()}_rmse"] = metrics.get("rmse", "")
        row[f"{index_name.lower()}_mae"] = metrics.get("mae", "")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=Path("target/release/nufrost-cli"))
    parser.add_argument("--base-config", type=Path, default=Path("config/nufrost.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/samples/sentinel-2_v1_5m"))
    parser.add_argument("--out-dir", type=Path, default=Path("target/parameter_sweeps_5m"))
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--min-joint-valid", type=int, default=12)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dyld-library-path")
    parser.add_argument("--normalization-mode", nargs="+", default=["reflectance"])
    parser.add_argument("--frequency-selection", nargs="+", default=["all"])
    parser.add_argument("--modes", nargs="+", type=int)
    parser.add_argument("--ridge", nargs="+", type=float)
    parser.add_argument("--freq-weight", nargs="+", type=float)
    parser.add_argument("--multiband-shrinkage", nargs="+", type=float)
    parser.add_argument("--huber-delta", nargs="+", type=float, default=[0.18])
    parser.add_argument("--outlier-reject-iters", nargs="+", type=int, default=[2])
    parser.add_argument("--outlier-reject-sigma", nargs="+", type=float, default=[2.5])
    parser.add_argument("--lambda-high", nargs="+", type=float)
    parser.add_argument("--low-freq-period-days", nargs="+", type=float)
    parser.add_argument("--include-trend", nargs="+", type=lambda v: v.lower() in {"1", "true", "yes"})
    parser.add_argument("--huber-iters-grid", nargs="+", type=int)
    args = parser.parse_args()

    base_config = read_json(args.base_config)
    if args.candidate_file is None:
        required_grid_args = {
            "--modes": args.modes,
            "--ridge": args.ridge,
            "--freq-weight": args.freq_weight,
            "--multiband-shrinkage": args.multiband_shrinkage,
        }
        missing = [name for name, value in required_grid_args.items() if not value]
        if missing:
            parser.error("missing grid arguments: " + ", ".join(missing))
    candidates = build_candidates(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.result_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, params in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] {params['name']}", flush=True)
        row = run_candidate(
            cli=args.cli,
            base_config=base_config,
            params=params,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            n_eval=args.n_eval,
            seed=args.seed,
            min_joint_valid=args.min_joint_valid,
            threads=args.threads,
            dyld_library_path=args.dyld_library_path,
        )
        print(
            f"  status={row['status']} rmse={row.get('rmse')} mae={row.get('mae')} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
        rows.append(row)
        rows.sort(key=lambda item: float(item["rmse"]) if item.get("rmse") != "" else float("inf"))
        with args.result_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {args.result_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
