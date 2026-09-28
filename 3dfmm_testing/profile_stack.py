#!/usr/bin/env python3
"""
Full 3DFMM stack profiler.

Usage:
    python 3dfmm_testing/profile_stack.py [motor.ric] [--mapDim N]

If no .ric file is given it uses the bundled finocyl reference motor.
The script prints a wall-clock breakdown of every major phase and then
a cProfile top-20 sorted by cumulative time so you can see exactly
what changed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import motorlib.grain as grain_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, label: str):
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        return False


def _find_3dfmm_grain(motor: Motor) -> grain_mod.Fmm3DGrain:
    for g in motor.grains:
        if isinstance(g, grain_mod.Fmm3DGrain):
            return g
    raise RuntimeError("No Fmm3DGrain found — is this a 3D-FMM motor?")


# ---------------------------------------------------------------------------
# Simulation runner with per-step instrumentation
# ---------------------------------------------------------------------------

def run_with_instrumentation(motor: Motor):
    """Run the simulation and collect per-call timing on getMassFlux/getPeakMassFlux."""
    # Grab the real method references so we can wrap them
    grain = _find_3dfmm_grain(motor)
    orig_getmf  = grain.getMassFlux
    orig_getpmf = grain.getPeakMassFlux

    mf_times:  list[float] = []
    pmf_times: list[float] = []

    def timed_getMassFlux(*a, **kw):
        t0 = time.perf_counter()
        result = orig_getmf(*a, **kw)
        mf_times.append(time.perf_counter() - t0)
        return result

    def timed_getPeakMassFlux(*a, **kw):
        t0 = time.perf_counter()
        result = orig_getpmf(*a, **kw)
        pmf_times.append(time.perf_counter() - t0)
        return result

    grain.getMassFlux     = timed_getMassFlux
    grain.getPeakMassFlux = timed_getPeakMassFlux

    results = motor.runSimulation()

    # Restore
    grain.getMassFlux     = orig_getmf
    grain.getPeakMassFlux = orig_getpmf

    return results, mf_times, pmf_times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full 3DFMM stack profiler")
    parser.add_argument(
        "motor_file",
        nargs="?",
        default=str(REPO_ROOT / "3dfmm_testing" / "3dfmm_finocyl.ric"),
        help="Path to a .ric motor file (default: 3dfmm_finocyl.ric)",
    )
    parser.add_argument(
        "--mapDim",
        type=int,
        default=None,
        help="Override the motor's 3DmapDim config value",
    )
    parser.add_argument(
        "--no-cprofile",
        action="store_true",
        help="Skip the cProfile pass (faster, just prints wall-clock breakdown)",
    )
    args = parser.parse_args()

    motor_path = Path(args.motor_file)
    print(f"\n{'='*70}")
    print(f"  Profile target : {motor_path.name}")
    print(f"{'='*70}\n")

    # ---- Load ---------------------------------------------------------------
    with _Timer("load") as t_load:
        data = loadFile(str(motor_path), fileTypes.MOTOR)
        motor = Motor(data)
    print(f"[load]           {t_load.elapsed:6.3f} s  — file parse + Motor.__init__")

    # ---- Inspect settings ---------------------------------------------------
    map_dim_cfg = motor.config.getProperty("3DmapDim")
    if args.mapDim:
        # patch the config
        motor.config.props["3DmapDim"].setValue(args.mapDim)
        map_dim_cfg = args.mapDim
        print(f"[config]  3DmapDim overridden → {map_dim_cfg}")

    grain = _find_3dfmm_grain(motor)
    print()
    print("  Motor settings")
    print(f"    3DmapDim             : {map_dim_cfg}")
    print(f"    meshedMassFlux       : {grain.props['meshedMassFlux'].getValue()}")
    print(f"    meshedMassFluxQuality: {grain.props['meshedMassFluxQuality'].getValue()}")
    print(f"    massFlux3D           : {grain.props['massFlux3D'].getValue()}")
    print(f"    inhibitedEnds        : {grain.props['inhibitedEnds'].getValue()}")
    print()

    # ---- simulationSetup (coreMap + regressionMap) --------------------------
    with _Timer("generateCoreMap") as t_core:
        grain.mapDim = motor.config.getProperty("3DmapDim")
        grain.generateCoreMap()

    reg_map_shape = None

    with _Timer("generateRegressionMap") as t_reg:
        grain.generateRegressionMap()
        reg_map_shape = grain.regressionMap.shape

    print(f"[generateCoreMap]    {t_core.elapsed:6.3f} s")
    print(f"[generateRegressionMap] {t_reg.elapsed:6.3f} s  — skfmm + marching-cubes sweep")
    print(f"  regressionMap shape: {reg_map_shape}")
    print(f"  wallWeb: {grain.wallWeb*1000:.2f} mm")
    print()

    # Re-load a fresh motor so simulationSetup can assign everything cleanly
    motor2 = Motor(loadFile(str(motor_path), fileTypes.MOTOR))
    if args.mapDim:
        motor2.config.props["3DmapDim"].setValue(args.mapDim)

    with _Timer("simulationSetup") as t_setup:
        for g in motor2.grains:
            g.simulationSetup(motor2.config)
    print(f"[simulationSetup]    {t_setup.elapsed:6.3f} s  (total for all grains)")
    print()

    # ---- Simulation loop ----------------------------------------------------
    motor3 = Motor(loadFile(str(motor_path), fileTypes.MOTOR))
    if args.mapDim:
        motor3.config.props["3DmapDim"].setValue(args.mapDim)
    for g in motor3.grains:
        g.simulationSetup(motor3.config)

    with _Timer("simulation_loop") as t_sim:
        results, mf_times, pmf_times = run_with_instrumentation(motor3)

    steps = len(mf_times) or len(pmf_times)
    total_mf  = sum(mf_times)
    total_pmf = sum(pmf_times)

    def _pct(t): return 100.0 * t / t_sim.elapsed if t_sim.elapsed > 0 else 0.0

    print(f"[simulation_loop]    {t_sim.elapsed:6.3f} s  ({steps} steps)")
    print(f"  getMassFlux     total: {total_mf:6.3f} s  ({_pct(total_mf):.1f}%),  "
          f"avg/step {1000*total_mf/max(steps,1):.2f} ms")
    print(f"  getPeakMassFlux total: {total_pmf:6.3f} s  ({_pct(total_pmf):.1f}%),  "
          f"avg/step {1000*total_pmf/max(steps,1):.2f} ms")
    overhead = t_sim.elapsed - total_mf - total_pmf
    print(f"  other loop overhead : {overhead:6.3f} s  ({_pct(overhead):.1f}%)")
    print()

    if results.success:
        burnTime = results.getBurnTime()
        totalImpulse = results.getImpulse()
        print(f"  Burn time    : {burnTime:.3f} s")
        print(f"  Total impulse: {totalImpulse:.1f} Ns")
    else:
        print("  Simulation did not succeed!")
    print()

    # ---- cProfile pass ------------------------------------------------------
    if not args.no_cprofile:
        print(f"{'='*70}")
        print("  cProfile — full simulation pass (top 25 by cumulative time)")
        print(f"{'='*70}\n")

        motor4 = Motor(loadFile(str(motor_path), fileTypes.MOTOR))
        if args.mapDim:
            motor4.config.props["3DmapDim"].setValue(args.mapDim)
        for g in motor4.grains:
            g.simulationSetup(motor4.config)

        pr = cProfile.Profile()
        pr.enable()
        motor4.runSimulation()
        pr.disable()

        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(25)
        print(stream.getvalue())

        # Also print a tottime-sorted view for hot leaves
        stream2 = io.StringIO()
        ps2 = pstats.Stats(pr, stream=stream2)
        ps2.sort_stats("tottime")
        ps2.print_stats(15)
        print("  --- top 15 by tottime (hot leaves) ---")
        print(stream2.getvalue())


if __name__ == "__main__":
    main()
