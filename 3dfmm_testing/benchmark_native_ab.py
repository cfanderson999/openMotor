#!/usr/bin/env python3
"""Quick parity + speed A/B checks for experimental native 3DFMM kernels.

Compares:
1) Core-map voxelization path (Cython voxelizer)
2) Regression face-area sweep path (skimage fallback vs Cython marching area)

This is intentionally a rough/fast harness, not a full certification suite.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import motorlib.grain as grain_mod
from motorlib.grains.custom3d import custom3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick parity + performance A/B for native 3DFMM kernels"
    )
    parser.add_argument(
        "--motor",
        type=Path,
        default=Path("3dfmm_testing/3dfmm_finocyl.ric"),
        help="Path to .ric motor containing a custom3d grain",
    )
    parser.add_argument(
        "--mapdim",
        type=int,
        default=128,
        help="3D map dimension override",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Timing repetitions per mode",
    )
    return parser.parse_args()


def load_motor(path: Path) -> Motor:
    data = loadFile(str(path), fileTypes.MOTOR)
    return Motor(data)


def find_3dfmm_grain(motor: Motor):
    for g in motor.grains:
        if isinstance(g, grain_mod.Fmm3DGrain):
            return g
    raise RuntimeError("No Fmm3DGrain found in motor")


def run_coremap(path: Path, mapdim: int, use_cy: bool):
    os.environ["OPENMOTOR_EXPERIMENTAL_CY_VOXEL"] = "1" if use_cy else "0"

    # Cold-run behavior for fair timing.
    custom3d._VOXEL_CACHE.clear()

    motor = load_motor(path)
    motor.config.props["3DmapDim"].setValue(int(mapdim))
    grain = find_3dfmm_grain(motor)
    grain.mapDim = motor.config.getProperty("3DmapDim")

    t0 = time.perf_counter()
    grain.generateCoreMap()
    dt = time.perf_counter() - t0

    return {
        "time": dt,
        "core_map": np.ascontiguousarray(grain.coreMap, dtype=np.uint8),
        "shape": tuple(grain.coreMap.shape),
        "map_length": int(grain.mapLength),
        "total_length_m": float(grain.totalLength.getValue()) if grain.totalLength else 0.0,
    }


def run_regression(path: Path, mapdim: int, use_cy: bool):
    os.environ["OPENMOTOR_EXPERIMENTAL_CY_MARCH"] = "1" if use_cy else "0"
    os.environ["OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP"] = "0"

    motor = load_motor(path)
    motor.config.props["3DmapDim"].setValue(int(mapdim))
    grain = find_3dfmm_grain(motor)
    grain.mapDim = motor.config.getProperty("3DmapDim")

    # Keep voxelizer path stable for this check.
    os.environ["OPENMOTOR_EXPERIMENTAL_CY_VOXEL"] = "0"
    custom3d._VOXEL_CACHE.clear()
    grain.generateCoreMap()

    t0 = time.perf_counter()
    grain.generateRegressionMap()
    dt = time.perf_counter() - t0

    face = np.asarray(grain.faceArea, dtype=float).copy()

    return {
        "time": dt,
        "wall_web": float(grain.wallWeb),
        "face_area": face,
        "face_len": int(len(face)),
    }


def summarize(ts):
    arr = np.asarray(ts, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def main() -> int:
    args = parse_args()

    print("=== Native Kernel Quick A/B ===")
    print(f"motor  : {args.motor}")
    print(f"mapDim : {args.mapdim}")
    print(f"reps   : {args.reps}")

    # ------------------------------------------------------------------
    # 1) Voxelization parity + timing
    # ------------------------------------------------------------------
    py_core = run_coremap(args.motor, args.mapdim, use_cy=False)
    cy_core = run_coremap(args.motor, args.mapdim, use_cy=True)

    same_shape = py_core["shape"] == cy_core["shape"]
    if same_shape:
        xor = np.bitwise_xor(py_core["core_map"], cy_core["core_map"]) > 0
        mismatch = int(np.count_nonzero(xor))
        total = int(py_core["core_map"].size)
        mismatch_pct = (100.0 * mismatch / total) if total else 0.0
    else:
        mismatch = -1
        mismatch_pct = float("nan")

    py_core_times = [run_coremap(args.motor, args.mapdim, use_cy=False)["time"] for _ in range(args.reps)]
    cy_core_times = [run_coremap(args.motor, args.mapdim, use_cy=True)["time"] for _ in range(args.reps)]
    py_core_mean, py_core_std = summarize(py_core_times)
    cy_core_mean, cy_core_std = summarize(cy_core_times)
    core_speedup = py_core_mean / cy_core_mean if cy_core_mean > 0 else float("inf")

    print("\n--- CoreMap / Voxelization ---")
    print(f"shape fallback   : {py_core['shape']}")
    print(f"shape cython     : {cy_core['shape']}")
    print(f"mapLength delta  : {cy_core['map_length'] - py_core['map_length']}")
    print(f"totalLength delta: {cy_core['total_length_m'] - py_core['total_length_m']:.12e} m")
    print(f"voxel mismatch   : {mismatch} ({mismatch_pct:.6f}%)")
    voxel_pass = same_shape and mismatch_pct < 1.0
    print(f"parity verdict   : {'PASS' if voxel_pass else 'CHECK'}")
    print(
        f"timing fallback  : mean={py_core_mean:.6f}s std={py_core_std:.6f}s"
    )
    print(
        f"timing cython    : mean={cy_core_mean:.6f}s std={cy_core_std:.6f}s"
    )
    print(f"speedup          : {core_speedup:.3f}x")

    # ------------------------------------------------------------------
    # 2) Regression sweep parity + timing
    # ------------------------------------------------------------------
    py_reg = run_regression(args.motor, args.mapdim, use_cy=False)
    cy_reg = run_regression(args.motor, args.mapdim, use_cy=True)

    n = min(py_reg["face_len"], cy_reg["face_len"])
    if n > 0:
        a = py_reg["face_area"][:n]
        b = cy_reg["face_area"][:n]
        max_abs = float(np.max(np.abs(a - b)))
        denom = np.maximum(np.abs(a), 1e-12)
        max_rel = float(np.max(np.abs(a - b) / denom))
    else:
        max_abs = float("nan")
        max_rel = float("nan")

    py_reg_times = [run_regression(args.motor, args.mapdim, use_cy=False)["time"] for _ in range(args.reps)]
    cy_reg_times = [run_regression(args.motor, args.mapdim, use_cy=True)["time"] for _ in range(args.reps)]
    py_reg_mean, py_reg_std = summarize(py_reg_times)
    cy_reg_mean, cy_reg_std = summarize(cy_reg_times)
    reg_speedup = py_reg_mean / cy_reg_mean if cy_reg_mean > 0 else float("inf")

    print("\n--- Regression Sweep / Marching Area ---")
    print(f"faceArea len fallback: {py_reg['face_len']}")
    print(f"faceArea len cython  : {cy_reg['face_len']}")
    print(f"wallWeb delta        : {cy_reg['wall_web'] - py_reg['wall_web']:.12e}")
    print(f"faceArea max_abs_diff: {max_abs:.12e}")
    print(f"faceArea max_rel_diff: {max_rel:.12e}")
    march_pass = (n > 0) and (max_rel < 5e-3)
    print(f"parity verdict       : {'PASS' if march_pass else 'CHECK'}")
    print(
        f"timing fallback      : mean={py_reg_mean:.6f}s std={py_reg_std:.6f}s"
    )
    print(
        f"timing cython        : mean={cy_reg_mean:.6f}s std={cy_reg_std:.6f}s"
    )
    print(f"speedup              : {reg_speedup:.3f}x")

    # ------------------------------------------------------------------
    # 3) One-level method diagnosis: native vs Lewiner vs Lorensen
    # ------------------------------------------------------------------
    os.environ["OPENMOTOR_EXPERIMENTAL_CY_MARCH"] = "1"
    os.environ["OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP"] = "0"
    os.environ["OPENMOTOR_EXPERIMENTAL_CY_VOXEL"] = "0"

    motor = load_motor(args.motor)
    motor.config.props["3DmapDim"].setValue(int(args.mapdim))
    grain = find_3dfmm_grain(motor)
    grain.mapDim = motor.config.getProperty("3DmapDim")
    custom3d._VOXEL_CACHE.clear()
    grain.generateCoreMap()
    grain.generateRegressionMap()

    from mathlib._march_cy import mc_surface_area

    level = min(max(0.5 * float(np.max(grain.regressionMap)), 0.0), float(np.max(grain.regressionMap)) * 0.95)
    grid = np.ascontiguousarray(grain.regressionMap, dtype=np.float64)
    vmask = np.ascontiguousarray(np.logical_not(grain.mask), dtype=np.uint8)
    native_area = float(mc_surface_area(grid, vmask, level))

    mask_bool = np.logical_not(grain.mask)
    try:
        v_l, f_l, _, _ = measure.marching_cubes(grid, level=level, mask=mask_bool, method="lewiner")
        area_l = float(measure.mesh_surface_area(v_l, f_l))
    except Exception:
        area_l = float("nan")

    try:
        v_c, f_c, _, _ = measure.marching_cubes(grid, level=level, mask=mask_bool, method="lorensen")
        area_c = float(measure.mesh_surface_area(v_c, f_c))
    except Exception:
        area_c = float("nan")

    print("\n--- Marching Method Diagnosis (single iso-level) ---")
    print(f"level                : {level:.12e}")
    print(f"native area          : {native_area:.12e}")
    print(f"lewiner area         : {area_l:.12e}")
    print(f"lorensen area        : {area_c:.12e}")
    print(f"native-lewiner abs   : {abs(native_area - area_l):.12e}")
    print(f"native-lorensen abs  : {abs(native_area - area_c):.12e}")

    # Also compare with no mask to detect whether mismatch is from cube-mask handling.
    one_mask = np.ones_like(vmask, dtype=np.uint8)
    native_nomask = float(mc_surface_area(grid, one_mask, level))
    try:
        v_nm, f_nm, _, _ = measure.marching_cubes(grid, level=level, method="lorensen")
        area_nomask = float(measure.mesh_surface_area(v_nm, f_nm))
    except Exception:
        area_nomask = float("nan")
    print(f"native(no mask) area : {native_nomask:.12e}")
    print(f"sk(no mask) area     : {area_nomask:.12e}")
    print(f"no-mask abs diff     : {abs(native_nomask - area_nomask):.12e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
