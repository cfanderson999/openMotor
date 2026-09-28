#!/usr/bin/env python3
"""Benchmark meshed mass-flux quality modes for 3DFMM grains."""

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


_DEFAULT_MOTORS = [
    Path("3dfmm_testing/3dfmm_bates.ric"),
    Path("3dfmm_testing/3dfmm_conical.ric"),
    Path("3dfmm_testing/3dfmm_finocyl.ric"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark meshed mass-flux quality settings")
    parser.add_argument(
        "--motors",
        nargs="+",
        type=Path,
        default=_DEFAULT_MOTORS,
        metavar="FILE",
        help="One or more .ric files containing a 3dfmm grain (default: all three test motors)",
    )
    parser.add_argument("--samples", type=int, default=36, help="Regression sample points")
    parser.add_argument("--reps", type=int, default=4, help="Timing repetitions")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Max acceptable mean_rel_diff for non-Exact recommendation (default: 0.05)",
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


def evaluate_flux(grain, regs, dreg, density):
    fluxes = []
    for reg in regs:
        aft, _ = grain.getEndPositionsInMapDim(float(reg))
        flux = grain.getMassFlux(
            massIn=0.0,
            dTime=0.01,
            regDist=float(reg),
            dRegDist=float(dreg),
            position=int(aft),
            density=float(density),
        )
        fluxes.append(float(flux))
    return np.array(fluxes, dtype=float)


def time_flux(grain, regs, dreg, density, reps):
    evaluate_flux(grain, regs, dreg, density)

    t0 = time.perf_counter()
    for _ in range(reps):
        evaluate_flux(grain, regs, dreg, density)
    return (time.perf_counter() - t0) / reps


def summarize_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    if len(reference) != len(candidate):
        return math.inf, math.inf

    abs_diff = np.abs(candidate - reference)
    max_abs = float(np.max(abs_diff)) if len(abs_diff) else 0.0

    denom = np.where(np.abs(reference) > 1e-12, np.abs(reference), np.nan)
    rel = abs_diff / denom
    mean_rel = float(np.nanmean(rel)) if np.any(np.isfinite(rel)) else 0.0
    return max_abs, mean_rel


def benchmark_motor(path: Path, samples: int, reps: int, threshold: float) -> dict:
    """Run all quality modes against one motor; return a result dict."""
    motor = load_motor(path)
    grain = find_3dfmm_grain(motor)
    density = motor.propellant.getProperty("density")

    grain_mod._HAS_FMM3D_CY = True
    grain.props["massFlux3D"].setValue(False)

    grain.simulationSetup(motor.config)

    regs = np.linspace(0.0, 0.7 * grain.wallWeb, samples)
    regs = filter_valid_regs(grain, regs)
    if len(regs) == 0:
        raise RuntimeError(f"No valid regression samples for {path.name}")

    dreg = max(1e-6, 0.003 * grain.wallWeb)

    quality_order = ("Exact", "Fast", "Faster")
    raw = {}
    for quality in quality_order:
        grain.props["meshedMassFluxQuality"].setValue(quality)
        flux = evaluate_flux(grain, regs, dreg, density)
        dt = time_flux(grain, regs, dreg, density, reps)
        raw[quality] = {"flux": flux, "time": dt}

    ref = raw["Exact"]["flux"]
    exact_time = raw["Exact"]["time"]

    rows = []
    for quality in quality_order:
        dt = raw[quality]["time"]
        speedup = exact_time / dt if dt > 0 else float("inf")
        max_abs, mean_rel = summarize_error(ref, raw[quality]["flux"])
        rows.append({"quality": quality, "time": dt, "speedup": speedup,
                     "max_abs": max_abs, "mean_rel": mean_rel})

    acceptable = [r for r in rows if r["quality"] != "Exact" and r["mean_rel"] <= threshold and r["speedup"] > 1.0]
    if acceptable:
        best = max(acceptable, key=lambda r: r["speedup"])
        recommended = best["quality"]
    else:
        recommended = "Exact"

    return {"path": path, "n_samples": len(regs), "rows": rows, "recommended": recommended,
            "threshold": threshold}


def print_motor_result(result: dict) -> None:
    path = result["path"]
    print(f"\n--- {path.name} (samples={result['n_samples']}) ---")
    for r in result["rows"]:
        print(
            f"  {r['quality']:7s} time={r['time']:.6f}s speedup={r['speedup']:.2f}x "
            f"max_abs_diff={r['max_abs']:.8f} mean_rel_diff={r['mean_rel']:.6f}"
        )
    rec = result["recommended"]
    if rec == "Exact":
        print(f"  recommended=Exact (no mode met mean_rel_diff <= {result['threshold']})")
    else:
        best = next(r for r in result["rows"] if r["quality"] == rec)
        print(
            f"  recommended={rec} "
            f"(speedup={best['speedup']:.2f}x, mean_rel_diff={best['mean_rel']:.6f})"
        )


def main() -> int:
    args = parse_args()

    print("=== Meshed Mass Flux Quality Benchmark ===")
    print(f"reps={args.reps}  threshold={args.threshold}")

    all_results = []
    for motor_path in args.motors:
        result = benchmark_motor(motor_path, args.samples, args.reps, args.threshold)
        print_motor_result(result)
        all_results.append(result)

    if len(all_results) > 1:
        print("\n=== Summary ===")
        for result in all_results:
            rec = result["recommended"]
            if rec != "Exact":
                best = next(r for r in result["rows"] if r["quality"] == rec)
                print(
                    f"  {result['path'].name}: recommended={rec} "
                    f"(speedup={best['speedup']:.2f}x, mean_rel_diff={best['mean_rel']:.6f})"
                )
            else:
                print(f"  {result['path'].name}: recommended=Exact")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
