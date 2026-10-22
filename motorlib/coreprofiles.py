"""Quantities along the core of a simulated motor, for shading the bore in the 3D views.

A profile holds one value per slice of a grain's regression map, counted from the aft end, for the instant a simulation
result was recorded at. Slices are the unit because the views draw the grain slice by slice. The values come from the
simulation, so nothing here belongs to a particular burn model: the steady model has a flow that depends only on the
burning surface upstream of a slice, and other models can supply their own profiles on the result.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .grain import Fmm3DGrain


@dataclass(frozen=True)
class CoreMetric:
    """A quantity that can shade the core. `unit` is the internal unit, as for a log channel."""
    key: str
    name: str
    unit: str


PRESSURE = CoreMetric("pressure", "Pressure", "Pa")
MASS_FLUX = CoreMetric("massFlux", "Mass Flux", "kg/(m^2*s)")
MACH = CoreMetric("machNumber", "Core Mach Number", "")
BURN_RATE = CoreMetric("burnRate", "Burn Rate", "m/s")
KN = CoreMetric("kn", "Kn Upstream", "")
PORT_RATIO = CoreMetric("portRatio", "Burning Area / Port Area", "")
EROSIVE_RATIO = CoreMetric("erosiveRatio", "Erosive Burn Rate Ratio", "")

STEADY_METRICS = [PRESSURE, MASS_FLUX, MACH, BURN_RATE, KN, PORT_RATIO]
ERODING_METRICS = [PRESSURE, MASS_FLUX, MACH, BURN_RATE, EROSIVE_RATIO, KN, PORT_RATIO]
METRICS: Dict[str, CoreMetric] = {metric.key: metric for metric in ERODING_METRICS}


@dataclass
class CoreProfile:
    """The core of one grain at one instant. `regression` is how deep the core is drawn, one value per slice (a model
    where the grain burns evenly gives the same value on every slice). `metrics` maps a metric key to its value on every
    slice, NaN where a slice has no port or the value is not defined."""
    regression: np.ndarray
    metrics: Dict[str, np.ndarray]


def availableMetrics(simResult) -> List[CoreMetric]:
    """The metrics that can be shaded for this result: those a model reported profiles of, or else the steady ones."""
    reported = {key for profiles in simResult.coreProfiles if profiles for p in profiles.values() for key in p.metrics}
    return [metric for metric in ERODING_METRICS if metric.key in reported] if reported else list(STEADY_METRICS)


def _pointOrNone(simResult, channel, step):
    data = simResult.channels[channel].getData()
    return data[step] if 0 <= step < len(data) else None


def getCoreProfiles(simResult, step) -> Dict[int, CoreProfile]:
    """Core profiles of every 3D FMM grain still burning at recorded step `step`, keyed by grain index. A burn model that
    followed the flow along the bore reports them itself. Otherwise they are worked out for the steady model, in which
    the gas leaves the propellant at the burn rate of the chamber pressure and flows toward the nozzle,
    so the mass flow past a slice is the flow arriving from the grains ahead of it plus the gas made by the part of the
    burning surface at or ahead of that slice. From it:

    - mass flux: that mass flow over the slice's port area;
    - core Mach number: from the mass flux and pressure, as the simulation reports it;
    - Kn upstream: the burning area at or ahead of the slice over the throat area, which reaches the motor's Kn at the nozzle;
    - burning area over port area: the same area over the slice's port, the measure of how hard the port is worked;
    - pressure and burn rate: the chamber pressure and the burn rate it gives, which do not vary along the core.
    """
    motor = simResult.motor
    reported = simResult.coreProfiles[step] if 0 <= step < len(simResult.coreProfiles) else None
    if reported is not None:
        return reported
    propellant = motor.propellant
    density = propellant.getProperty("density")
    regressions = _pointOrNone(simResult, "regression", step)
    if regressions is None or propellant is None:
        return {}
    dThroat = _pointOrNone(simResult, "dThroat", step) or 0.0
    pressure = _pointOrNone(simResult, "pressure", step)
    if pressure is None:
        pressure = motor.calcIdealPressure(regressions, dThroat)
    burnRate = propellant.getBurnRate(pressure)
    throatArea = motor.nozzle.getThroatArea(dThroat)
    burnoutWebThres = motor.config.getProperty("burnoutWebThres")

    profiles: Dict[int, CoreProfile] = {}
    upstreamArea = 0.0
    for gid, grain in enumerate(motor.grains):
        regDist = regressions[gid]
        burning = grain.isWebLeft(regDist, burnoutWebThres)
        if isinstance(grain, Fmm3DGrain) and grain.regressionMap is not None and burning:
            core = grain.getCoreProfile(regDist)
            if core is not None:
                portArea, areaAhead = core
                areaPast = upstreamArea + areaAhead
                massFlow = density * burnRate * areaPast
                with np.errstate(divide="ignore", invalid="ignore"):
                    massFlux = np.where(portArea > 0, massFlow / portArea, np.nan)
                    portRatio = np.where(portArea > 0, areaPast / portArea, np.nan)
                mach = np.array([motor.calcMachNumber(pressure, flux) if np.isfinite(flux) else np.nan for flux in massFlux])
                zdim = len(portArea)
                profiles[gid] = CoreProfile(
                    regression=np.full(zdim, regDist),
                    metrics={
                        PRESSURE.key: np.full(zdim, pressure),
                        MASS_FLUX.key: massFlux,
                        MACH.key: mach,
                        BURN_RATE.key: np.full(zdim, burnRate),
                        KN.key: areaPast / throatArea,
                        PORT_RATIO.key: portRatio,
                    },
                )
        if burning:
            upstreamArea += grain.getSurfaceAreaAtRegression(regDist)
    return profiles
