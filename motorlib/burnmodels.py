"""The burn physics of one simulation timestep.

`Motor.runSimulation` steps a burn model through time and records what it reports. A model says how the grains regress and
where the chamber pressure ends up after `dTime` seconds:

- `SteadyBurn` regresses every grain at the burn rate of the last pressure and puts the chamber at the steady pressure for
  its Kn. This is the normal simulation.
- `ErosiveBurn` lets the burn rate follow the gas flow along the bore (Ma et al., 2020) and follows the chamber pressure
  through its filling and emptying, on the 3D FMM version of the grains.

Everything else, the channels, the units, the stopping rule, the progress callback and the checks on the result, is the
simulation loop's and is the same for both.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .simResult import SimAlert, SimAlertLevel, SimAlertType

INITIAL_TEMPERATURE = 293.0


@dataclass
class BurnStep:
    """What a model reports after a step: one entry per grain unless noted. `machNumber` is None when the model has none
    of its own and the loop derives it from the mass flux."""
    regression: List[float]
    web: List[float]
    volumeLoading: float
    mass: List[float]
    massFlow: List[float]
    massFlux: List[float]
    kn: float
    pressure: float
    machNumber: Optional[List[float]] = None


class SteadyBurn:
    """Every grain regresses at the burn rate of the last pressure; the chamber sits at the steady pressure for its Kn."""

    def __init__(self, motor):
        self.motor = motor
        self.density = motor.propellant.getProperty("density")
        self.burnoutWebThres = motor.config.getProperty("burnoutWebThres")
        self.motorVolume = motor.calcTotalVolume()
        self.regression = [0 for _ in motor.grains]

    def initial(self):
        motor = self.motor
        return BurnStep(
            regression=[0 for _ in motor.grains],
            web=[grain.getWebLeft(0) for grain in motor.grains],
            volumeLoading=100 * (1 - (motor.calcFreeVolume(self.regression) / self.motorVolume)),
            mass=[grain.getVolumeAtRegression(0) * self.density for grain in motor.grains],
            massFlow=[0 for _ in motor.grains],
            massFlux=[0 for _ in motor.grains],
            kn=motor.calcKN(self.regression, 0),
            pressure=motor.calcIdealPressure(self.regression, 0, None),
        )

    def step(self, dTime, dThroat, previous):
        motor, density = self.motor, self.density
        count = len(motor.grains)
        massFlow = 0
        perGrainMass = [0] * count
        perGrainMassFlow = [0] * count
        perGrainMassFlux = [0] * count
        perGrainWeb = [0] * count
        for gid, grain in enumerate(motor.grains):
            if grain.getWebLeft(self.regression[gid]) > self.burnoutWebThres:
                # Calculate regression at the current pressure
                reg = dTime * motor.propellant.getBurnRate(previous.pressure)
                # Find the mass flux through the grain based on the mass flow fed into from grains above it
                perGrainMassFlux[gid] = grain.getPeakMassFlux(massFlow, dTime, self.regression[gid], reg, density)
                # Find the mass of the grain after regression
                perGrainMass[gid] = grain.getVolumeAtRegression(self.regression[gid]) * density
                # Add the change in grain mass to the mass flow
                massFlow += (previous.mass[gid] - perGrainMass[gid]) / dTime
                # Apply the regression
                self.regression[gid] += reg
                perGrainWeb[gid] = grain.getWebLeft(self.regression[gid])
            perGrainMassFlow[gid] = massFlow
        kn = motor.calcKN(self.regression, dThroat)
        return BurnStep(
            regression=self.regression[:],
            web=perGrainWeb,
            volumeLoading=100 * (1 - (motor.calcFreeVolume(self.regression) / self.motorVolume)),
            mass=perGrainMass,
            massFlow=perGrainMassFlow,
            massFlux=perGrainMassFlux,
            kn=kn,
            pressure=motor.calcIdealPressure(self.regression, dThroat, kn),
        )

    def progress(self, step):
        """The share of the web that has burned, in the grain with the most left."""
        return 1 - max(g.getWebLeft(r) / g.getWebLeft(0) for g, r in zip(self.motor.grains, self.regression))

    def getAlerts(self):
        return []


def flowProperties(motor, initialPressure):
    """The solver's gas and propellant inputs. The combustion state (specific heat ratio, flame temperature, molar mass) is
    that of the propellant tab holding `initialPressure`, because the solver follows one state through the burn."""
    from .srm1d import FlowProperties
    from .srm1d.surface import DEFAULT_AP_VOLUME_FRACTION

    propellant = motor.propellant
    tabs = propellant.getProperty("tabs")
    _, _, gamma, temperature, molarMass = propellant.getCombustionProperties(initialPressure)
    apFraction = propellant.getProperty("apVolumeFraction")
    return FlowProperties(
        rhoPropellant=propellant.getProperty("density"),
        cpPropellant=propellant.getProperty("solidSpecificHeat"),
        tFlame=temperature,
        tInitial=INITIAL_TEMPERATURE,
        gamma=gamma,
        molarMass=molarMass / 1000.0,
        cpGas=propellant.getProperty("gasSpecificHeat"),
        muGas=propellant.getProperty("gasViscosity"),
        kGas=propellant.getProperty("gasConductivity"),
        tabMinP=np.array([tab["minPressure"] for tab in tabs], dtype=float),
        tabMaxP=np.array([tab["maxPressure"] for tab in tabs], dtype=float),
        tabA=np.array([tab["a"] for tab in tabs], dtype=float),
        tabN=np.array([tab["n"] for tab in tabs], dtype=float),
        roughness=motor.config.getProperty("erosiveRoughness"),
        apVolumeFraction=apFraction if apFraction > 0 else DEFAULT_AP_VOLUME_FRACTION,
    )


class ErosiveBurn:
    """The grains burn on one shared level set, each axial position at the rate the gas flow gives it, and the gas flow
    along the bore is solved by the quasi-steady solver of `srm1d`. A step of `dTime` seconds advances both by exactly that
    long, in as many solver spans as the surface and the pressure call for, so the step may be as long as the motor's
    timestep; a short one only costs more geometry updates. All grains must be 3D FMM grains of one casting tube."""

    def __init__(self, motor):
        from .srm1d import BurnSimulation
        from .srm1d.adapter import burnbackFromFmm3dGrains

        self.motor = motor
        propellant, nozzle, config = motor.propellant, motor.nozzle, motor.config
        self.density = propellant.getProperty("density")
        self.motorVolume = motor.calcTotalVolume()
        self.throat0 = nozzle.props["throat"].getValue()
        self.grid = burnbackFromFmm3dGrains(motor.grains)
        voxel = self.grid.voxel
        starts = [a for a, _ in self.grid.grainSlices] + [self.grid.phi.shape[0] - 1]
        self.sliceRanges = self.grid.grainSlices
        self.slabRanges = [(starts[i], starts[i + 1]) for i in range(len(self.sliceRanges))]
        self.zStarts = np.array([a * voxel for a, _ in self.slabRanges])
        self.zEnds = np.array([b * voxel for _, b in self.slabRanges])
        self._cached = (None, None)

        self.initialKn = float(self.grid.slabBurnArea().sum()) / nozzle.getThroatArea(0)
        self.initialPressure = propellant.getPressureFromKn(self.initialKn)
        self.props = flowProperties(motor, self.initialPressure)
        # The loop tracks the throat, so the solver is not given the erosion and slag coefficients.
        self.sim = BurnSimulation(self.grid, self.props, self.throat0, pAmbient=config.getProperty("ambPressure"), pCutoff=0.0,
                                  cellLength=config.getProperty("erosiveCellLength"))
        self.initialWeb = None

    def _geometry(self):
        """Per grain propellant volume, deepest remaining web and mean regression, kept while the surface is not moving."""
        grid = self.grid
        if self._cached[0] == grid.elapsed:
            return self._cached[1]
        inside = (grid.phi > 0) & grid.valid
        counts = np.count_nonzero(inside, axis=(1, 2))
        deepest = grid.phi.max(axis=(1, 2), initial=0.0, where=inside)
        volume = np.array([counts[a:b].sum() for a, b in self.sliceRanges]) * grid.voxel ** 3
        web = np.array([deepest[a:b].max() for a, b in self.sliceRanges])
        regression = np.array([grid.slabRegression[a:b].mean() for a, b in self.slabRanges])
        self._cached = (grid.elapsed, (volume, web, regression))
        return self._cached[1]

    def _largestPerGrain(self, profile):
        """Largest value of a per-cell profile inside each grain; a grain shorter than a cell takes its nearest cell."""
        centers = self.sim.flowGrid.centers
        out = []
        for z0, z1 in zip(self.zStarts, self.zEnds):
            inGrain = (centers >= z0) & (centers <= z1)
            out.append(profile[inGrain].max() if inGrain.any() else profile[int(np.argmin(np.abs(centers - 0.5 * (z0 + z1))))])
        return [float(v) for v in out]

    def initial(self):
        volume, web, regression = self._geometry()
        self.initialWeb = np.maximum(web, 1e-12)
        count = len(volume)
        return BurnStep(
            regression=[float(v) for v in regression], web=[float(v) for v in web],
            volumeLoading=float(100 * volume.sum() / self.motorVolume), mass=[float(v) for v in volume * self.density],
            massFlow=[0.0] * count, massFlux=[0.0] * count, kn=self.initialKn, pressure=self.initialPressure,
        )

    def step(self, dTime, dThroat, previous):
        sim, nozzle = self.sim, self.motor.nozzle
        sim.setThroat(self.throat0 + dThroat)
        sim.advance(dTime)
        volume, web, regression = self._geometry()
        faces = np.arange(sim.flowGrid.cells + 1) * sim.flowGrid.length
        return BurnStep(
            regression=[float(v) for v in regression], web=[float(v) for v in web],
            volumeLoading=float(100 * volume.sum() / self.motorVolume), mass=[float(v) for v in volume * self.density],
            massFlow=[float(v) for v in np.interp(self.zEnds, faces, sim.massFlowFaces())],
            massFlux=self._largestPerGrain(sim.massFluxCells()),
            kn=sim.burnArea() / nozzle.getThroatArea(dThroat), pressure=float(sim.pressure[0]),
            machNumber=self._largestPerGrain(sim.machCells()),
        )

    def progress(self, step):
        return 1 - float(np.max(np.array(step.web) / self.initialWeb))

    def getAlerts(self):
        """Notes on what the model assumed, and where it left the data its correlation was validated on."""
        alerts = []
        propellant = self.motor.propellant
        tabs = propellant.getProperty("tabs")
        if len({(tab["k"], tab["t"], tab["m"]) for tab in tabs}) > 1:
            alerts.append(SimAlert(
                SimAlertLevel.WARNING, SimAlertType.VALUE,
                "The propellant has several tabs with different combustion states; erosive burning uses the tab that "
                "holds the initial pressure for the whole burn.", "Propellant"))
        if propellant.getProperty("apVolumeFraction") <= 0:
            alerts.append(SimAlert(
                SimAlertLevel.MESSAGE, SimAlertType.VALUE,
                "The propellant has no ammonium perchlorate volume fraction; erosive burning assumes {:.2f}, the value of "
                "the propellant in Ma et al. (2020).".format(self.props.apVolumeFraction), "Propellant"))
        for text in self.sim.result.outsideValidatedRange():
            alerts.append(SimAlert(
                SimAlertLevel.WARNING, SimAlertType.VALUE,
                "Beyond the data the erosive burning correlation was validated on: " + text, "Erosive burning"))
        return alerts
