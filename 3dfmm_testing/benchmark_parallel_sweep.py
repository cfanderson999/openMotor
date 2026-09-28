#!/usr/bin/env python3
"""A/B benchmark for experimental parallel Fmm3D regression-area sweep.

This script benchmarks `generateRegressionMap` in serial vs parallel sweep mode
and checks that the generated face-area curve is numerically equivalent.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import motorlib.grain as grain_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parallel 3DFMM regression sweep")
    parser.add_argument(
        "--motor",
        type=Path,
        default=Path("3dfmm_testing/3dfmm_finocyl.ric"),
        help="Path to .ric motor file",
    )
    parser.add_argument("--reps", type=int, default=3, help="Timing reps per mode")
    parser.add_argument(
        "--mapdims",
        type=str,
        default="128",
        help="Comma-separated 3D map dimensions to benchmark (e.g. 64,128,192)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers for parallel mode (0 uses CPU count)",
    )
    return parser.parse_args()


def load_motor(path: Path) -> Motor:
    return Motor(loadFile(str(path), fileTypes.MOTOR))


def find_3dfmm_grain(motor: Motor):
    for g in motor.grains:
        if isinstance(g, grain_mod.Fmm3DGrain):
            return g
    raise RuntimeError("No Fmm3DGrain found in motor")


def timed_regression(path: Path, map_dim: int, parallel: bool, workers: int):
    os.environ["OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP"] = "1" if parallel else "0"
    if workers > 0:
        os.environ["OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP_WORKERS"] = str(workers)
    else:
        os.environ.pop("OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP_WORKERS", None)

    motor = load_motor(path)
    motor.config.props["3DmapDim"].setValue(int(map_dim))
    grain = find_3dfmm_grain(motor)
    grain.mapDim = motor.config.getProperty("3DmapDim")

    grain.generateCoreMap()

    t0 = time.perf_counter()
    grain.generateRegressionMap()
    dt = time.perf_counter() - t0

    return {
        "time": dt,
        "wall_web": float(grain.wallWeb),
        "face_area": np.asarray(grain.faceArea, dtype=float).copy(),
    }


def summarize_times(times):
    arr = np.asarray(times, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))


def main() -> int:
    args = parse_args()

    map_dims = [int(tok.strip()) for tok in args.mapdims.split(",") if tok.strip()]
    if not map_dims:
        print("No valid map dimensions provided")
        return 1

    print("=== Parallel Sweep A/B Benchmark ===")
    print(f"motor     : {args.motor}")
    print(f"reps      : {args.reps}")
    print(f"workers   : {args.workers if args.workers > 0 else 'auto'}")
    print(f"mapdims   : {map_dims}")

    for map_dim in map_dims:
        serial_times = []
        parallel_times = []
        serial_ref = None
        parallel_ref = None

        for _ in range(args.reps):
            out = timed_regression(args.motor, map_dim, parallel=False, workers=args.workers)
            serial_times.append(out["time"])
            serial_ref = out

            out = timed_regression(args.motor, map_dim, parallel=True, workers=args.workers)
            parallel_times.append(out["time"])
            parallel_ref = out

        s_mean, s_std, s_min, s_max = summarize_times(serial_times)
        p_mean, p_std, p_min, p_max = summarize_times(parallel_times)

        if serial_ref is None or parallel_ref is None:
            print(f"\n--- mapDim={map_dim} ---")
            print("No benchmark data collected")
            continue

        n = min(len(serial_ref["face_area"]), len(parallel_ref["face_area"]))
        if n == 0:
            max_abs = float("nan")
            max_rel = float("nan")
        else:
            s = serial_ref["face_area"][:n]
            p = parallel_ref["face_area"][:n]
            max_abs = float(np.max(np.abs(p - s)))
            denom = np.maximum(np.abs(s), 1e-12)
            max_rel = float(np.max(np.abs(p - s) / denom))

        speedup = s_mean / p_mean if p_mean > 0 else float("inf")

        print(f"\n--- mapDim={map_dim} ---")
        print("Serial:")
        print(f"  mean={s_mean:.6f}s std={s_std:.6f}s min={s_min:.6f}s max={s_max:.6f}s")
        print("Parallel:")
        print(f"  mean={p_mean:.6f}s std={p_std:.6f}s min={p_min:.6f}s max={p_max:.6f}s")
        print(f"Speedup (serial/parallel): {speedup:.3f}x")
        print(f"wallWeb delta: {parallel_ref['wall_web'] - serial_ref['wall_web']:.12e}")
        print(f"faceArea max_abs_diff: {max_abs:.12e}")
        print(f"faceArea max_rel_diff: {max_rel:.12e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
