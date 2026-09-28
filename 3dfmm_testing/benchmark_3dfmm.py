#!/usr/bin/env python3
"""Targeted validation and performance checks for 3dfmm Cython kernels."""

from __future__ import annotations

import argparse
import math
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
    parser = argparse.ArgumentParser(description="Validate/benchmark 3dfmm Cython hot paths")
    parser.add_argument(
        "--motor",
        type=Path,
        default=Path("3dfmm_testing/3dfmm_bates.ric"),
        help="Path to .ric file containing a 3dfmm grain",
    )
    parser.add_argument("--samples", type=int, default=24, help="Regression sample points")
    parser.add_argument("--reps", type=int, default=4, help="Timing repetitions")
    return parser.parse_args()


def load_motor(path: Path) -> Motor:
    data = loadFile(str(path), fileTypes.MOTOR)
    return Motor(data)


def find_3dfmm_grain(motor: Motor):
    for g in motor.grains:
        if isinstance(g, grain_mod.Fmm3DGrain):
            return g
    raise RuntimeError("No Fmm3DGrain found in motor")


def evaluate_methods(grain, regs, dreg, density):
    results = {
        "endpos": [],
        "port_area": [],
        "volume": [],
        "mass_flux": [],
    }

    for reg in regs:
        aft, fore = grain.getEndPositionsInMapDim(float(reg))
        results["endpos"].append((int(aft), int(fore)))
        results["port_area"].append(float(grain.getPortArea(float(reg))))
        results["volume"].append(float(grain.getVolumeAtRegression(float(reg))))

        pos = int(aft)
        flux = grain.getMassFlux(
            massIn=0.0,
            dTime=0.01,
            regDist=float(reg),
            dRegDist=float(dreg),
            position=pos,
            density=float(density),
        )
        results["mass_flux"].append(float(flux))

    return results


def filter_valid_regs(grain, regs):
    valid = []
    for reg in regs:
        try:
            aft, fore = grain.getEndPositionsInMapDim(float(reg))
            if fore >= aft:
                valid.append(float(reg))
        except Exception:
            continue
    return np.array(valid, dtype=float)


def time_method(label, fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t0) / reps
    return label, dt


def max_abs_diff(a, b):
    if len(a) != len(b):
        return math.inf
    out = 0.0
    for av, bv in zip(a, b):
        if isinstance(av, tuple):
            out = max(out, abs(av[0] - bv[0]), abs(av[1] - bv[1]))
        else:
            out = max(out, abs(av - bv))
    return out


def main() -> int:
    args = parse_args()

    motor = load_motor(args.motor)
    grain = find_3dfmm_grain(motor)
    grain.props["meshedMassFlux"].setValue(False)
    grain.props["massFlux3D"].setValue(False)

    grain.simulationSetup(motor.config)

    density = motor.propellant.getProperty("density")
    max_reg = 0.8 * grain.wallWeb
    regs = np.linspace(0.0, max_reg, args.samples)
    dreg = max(1e-6, 0.003 * grain.wallWeb)

    grain_mod._HAS_FMM3D_CY = False
    regs = filter_valid_regs(grain, regs)
    if len(regs) == 0:
        raise RuntimeError("No valid regression samples available for benchmark")

    py_results = evaluate_methods(grain, regs, dreg, density)

    grain_mod._HAS_FMM3D_CY = True
    cy_results = evaluate_methods(grain, regs, dreg, density)

    checks = [
        ("endpos", py_results["endpos"], cy_results["endpos"]),
        ("port_area", py_results["port_area"], cy_results["port_area"]),
        ("volume", py_results["volume"], cy_results["volume"]),
        ("mass_flux", py_results["mass_flux"], cy_results["mass_flux"]),
    ]

    print("=== 3DFMM Validation (Cython vs fallback) ===")
    for name, a, b in checks:
        mad = max_abs_diff(a, b)
        print(f"{name:10s} max_abs_diff={mad:.12g}")

    def bench_variant(use_cy):
        grain_mod._HAS_FMM3D_CY = use_cy

        def fn_endpos():
            for reg in regs:
                grain.getEndPositionsInMapDim(float(reg))

        def fn_port():
            for reg in regs:
                grain.getPortArea(float(reg))

        def fn_vol():
            for reg in regs:
                grain.getVolumeAtRegression(float(reg))

        def fn_flux():
            for reg in regs:
                aft, _ = grain.getEndPositionsInMapDim(float(reg))
                grain.getMassFlux(0.0, 0.01, float(reg), float(dreg), int(aft), float(density))

        timings = {}
        for label, func in (
            ("endpos", fn_endpos),
            ("port_area", fn_port),
            ("volume", fn_vol),
            ("mass_flux", fn_flux),
        ):
            _, dt = time_method(label, func, args.reps)
            timings[label] = dt
        return timings

    py_t = bench_variant(False)
    cy_t = bench_variant(True)

    print("\n=== 3DFMM Timing (seconds per sweep) ===")
    for key in ("endpos", "port_area", "volume", "mass_flux"):
        a = py_t[key]
        b = cy_t[key]
        speedup = (a / b) if b > 0 else float("inf")
        print(f"{key:10s} fallback={a:.6f} cython={b:.6f} speedup={speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
