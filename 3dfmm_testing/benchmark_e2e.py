#!/usr/bin/env python3
"""End-to-end performance and accuracy benchmark for 3dfmm grains.

Tests all combinations of massFlux3D (on/off) and meshedMassFluxQuality
(Exact / Fast / Faster) across all three test motors, reporting wall-clock
time and accuracy vs the analytical reference for each configuration.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import motorlib.grain as grain_mod


# ---------------------------------------------------------------------------
# Motor / shape pairs
# ---------------------------------------------------------------------------

SHAPES = [
    ("bates",   "analytical_bates.ric",   "3dfmm_bates.ric"),
    ("conical", "analytical_conical.ric", "3dfmm_conical.ric"),
    ("finocyl", "analytical_finocyl.ric", "3dfmm_finocyl.ric"),
]

TESTING_DIR = Path(__file__).resolve().parent

# Ordered list of (label, massFlux3D, quality)
CONFIGS: list[tuple[str, bool, str | None]] = [
    ("3D=off       ", False, None),
    ("3D=on/Exact  ", True,  "Exact"),
    ("3D=on/Fast   ", True,  "Fast"),
    ("3D=on/Faster ", True,  "Faster"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_motor(path: Path) -> Motor:
    data = loadFile(str(path), fileTypes.MOTOR)
    return Motor(data)


def find_3dfmm_grain(motor: Motor):
    for g in motor.grains:
        if isinstance(g, grain_mod.Fmm3DGrain):
            return g
    return None


def has_errors(sim_res) -> bool:
    from motorlib.simResult import SimAlertLevel
    return len(sim_res.getAlertsByLevel(SimAlertLevel.ERROR)) > 0


def extract_metrics(sim_res) -> dict:
    return {
        "burn_time":     sim_res.getBurnTime(),
        "peak_force":    sim_res.channels["force"].getMax(),
        "peak_pressure": sim_res.getMaxPressure(),
        "total_impulse": sim_res.getImpulse(),
    }


def pct_delta(ref: float, val: float) -> float:
    if ref == 0:
        return math.nan
    return (val - ref) / ref * 100.0


def run_once(ric_path: Path, mass_flux_3d: bool, quality: str | None, timestep_override: float | None = None, map_dim_3d: int | None = None) -> tuple[dict, float]:
    """Load motor, configure grain, run simulation. Returns (metrics, elapsed_s)."""
    motor = load_motor(ric_path)
    if timestep_override is not None:
        motor.config.props["timestep"].setValue(timestep_override)
    if map_dim_3d is not None:
        motor.config.props["3DmapDim"].setValue(map_dim_3d)
    grain = find_3dfmm_grain(motor)
    if grain is not None:
        grain.props["massFlux3D"].setValue(mass_flux_3d)
        if quality is not None and mass_flux_3d:
            grain.props["meshedMassFluxQuality"].setValue(quality)

    t0 = time.perf_counter()
    result = motor.runSimulation()
    elapsed = time.perf_counter() - t0

    if has_errors(result):
        raise RuntimeError(f"Simulation errors in {ric_path.name}")

    return extract_metrics(result), elapsed


# ---------------------------------------------------------------------------
# Benchmark one shape
# ---------------------------------------------------------------------------

def benchmark_shape(shape: str, analytical_ric: str, fmm_ric: str, reps: int, timestep_override: float | None = None, map_dim_3d: int | None = None) -> dict:
    analytical_path = TESTING_DIR / analytical_ric
    fmm_path        = TESTING_DIR / fmm_ric

    # Analytical baseline (not timed for performance — just for accuracy reference)
    analytical_motor = load_motor(analytical_path)
    if timestep_override is not None:
        analytical_motor.config.props["timestep"].setValue(timestep_override)
    analytical_result = analytical_motor.runSimulation()
    if has_errors(analytical_result):
        raise RuntimeError(f"Analytical run failed for {shape}")
    ref = extract_metrics(analytical_result)

    configs_out = []
    for label, use_3d, quality in CONFIGS:
        times = []
        last_metrics = None
        for _ in range(reps):
            metrics, elapsed = run_once(fmm_path, use_3d, quality, timestep_override, map_dim_3d)
            times.append(elapsed)
            last_metrics = metrics

        avg_time = sum(times) / len(times)
        min_time = min(times)

        deltas = {k: pct_delta(ref[k], last_metrics[k]) for k in ref}
        configs_out.append({
            "label":    label,
            "use_3d":   use_3d,
            "quality":  quality,
            "avg_s":    avg_time,
            "min_s":    min_time,
            "metrics":  last_metrics,
            "deltas":   deltas,
        })

    return {"shape": shape, "ref": ref, "configs": configs_out}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_shape_result(result: dict, reps: int) -> None:
    shape = result["shape"]
    configs = result["configs"]

    # Use 3D=off as the speed reference for speedup column
    base_time = next(c["avg_s"] for c in configs if not c["use_3d"])

    print(f"\n{'='*80}")
    print(f"  {shape.upper()}  (reps={reps})")
    print(f"{'='*80}")

    # Header
    print(
        f"  {'Config':<20} {'avg(s)':>8} {'min(s)':>8} {'speedup':>8}"
        f"  {'burn_t%':>8} {'pk_F%':>8} {'pk_P%':>8} {'impulse%':>8}"
    )
    print("  " + "-" * 79)

    for c in configs:
        speedup = base_time / c["avg_s"] if c["avg_s"] > 0 else float("inf")
        d = c["deltas"]
        print(
            f"  {c['label']:<20} {c['avg_s']:>8.3f} {c['min_s']:>8.3f} {speedup:>8.2f}x"
            f"  {d['burn_time']:>+8.2f}% {d['peak_force']:>+8.2f}%"
            f" {d['peak_pressure']:>+8.2f}% {d['total_impulse']:>+8.2f}%"
        )

    # Raw metrics table
    print()
    print(f"  {'Config':<20} {'burn_time(s)':>14} {'peak_F(N)':>12} {'peak_P(Pa)':>12} {'impulse(Ns)':>13}")
    print("  " + "-" * 74)
    ref = result["ref"]
    print(
        f"  {'(analytical)':<20} {ref['burn_time']:>14.4f} {ref['peak_force']:>12.2f}"
        f" {ref['peak_pressure']:>12.2f} {ref['total_impulse']:>13.2f}"
    )
    for c in configs:
        m = c["metrics"]
        print(
            f"  {c['label']:<20} {m['burn_time']:>14.4f} {m['peak_force']:>12.2f}"
            f" {m['peak_pressure']:>12.2f} {m['total_impulse']:>13.2f}"
        )


def print_summary(all_results: list[dict]) -> None:
    """Print per-config averages across all shapes."""
    print(f"\n{'='*80}")
    print("  SUMMARY — averaged across all shapes")
    print(f"{'='*80}")

    # Collect per-config stats
    config_labels = [c["label"] for c in all_results[0]["configs"]]
    base_times = {r["shape"]: next(c["avg_s"] for c in r["configs"] if not c["use_3d"])
                  for r in all_results}

    print(f"  {'Config':<20} {'avg_time(s)':>12} {'avg_speedup':>12} {'avg|impulse%|':>14} {'avg|pk_P%|':>12}")
    print("  " + "-" * 74)

    for label in config_labels:
        times, speedups, impulse_errs, pressure_errs = [], [], [], []
        for r in all_results:
            c = next(x for x in r["configs"] if x["label"] == label)
            bt = base_times[r["shape"]]
            times.append(c["avg_s"])
            speedups.append(bt / c["avg_s"] if c["avg_s"] > 0 else float("inf"))
            impulse_errs.append(abs(c["deltas"]["total_impulse"]))
            pressure_errs.append(abs(c["deltas"]["peak_pressure"]))

        print(
            f"  {label:<20} {sum(times)/len(times):>12.3f} {sum(speedups)/len(speedups):>12.2f}x"
            f" {sum(impulse_errs)/len(impulse_errs):>14.2f}% {sum(pressure_errs)/len(pressure_errs):>12.2f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end 3dfmm performance+accuracy benchmark")
    parser.add_argument("--reps", type=int, default=2, help="Timing repetitions per config (default: 2)")
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=["bates", "conical", "finocyl"],
        default=["bates", "conical", "finocyl"],
        help="Which shapes to benchmark",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Override simulation timestep (s). Use e.g. 0.05 for a fast ~5x speedup.",
    )
    parser.add_argument(
        "--3d-map-dim",
        dest="map_dim_3d",
        type=int,
        default=None,
        help="Override 3D grain map dimension (e.g. 32 for fast timing runs vs default 128).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {s[0]: s for s in SHAPES if s[0] in args.shapes}

    all_results = []
    if args.timestep is not None:
        print(f"(timestep override: {args.timestep}s)")
    if args.map_dim_3d is not None:
        print(f"(3D map dim override: {args.map_dim_3d})")
    for shape, analytical_ric, fmm_ric in SHAPES:
        if shape not in selected:
            continue
        print(f"\nBenchmarking {shape}...", flush=True)
        result = benchmark_shape(shape, analytical_ric, fmm_ric, args.reps, args.timestep, args.map_dim_3d)
        print_shape_result(result, args.reps)
        all_results.append(result)

    if len(all_results) > 1:
        print_summary(all_results)

    print()


if __name__ == "__main__":
    main()
