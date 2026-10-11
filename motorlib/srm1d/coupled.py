"""Time-marching driver: 1D transient bore flow coupled to a burnback geometry provider.

A geometry provider exposes `nSlabs`, `slabCenters`, `voxel` (cell length), `flowGeometry()` -> (aPort, dHyd,
burnLength) and `advance(slabSpeed, dt)`.
"""

from dataclasses import dataclass, field
import numpy as np

from .core import FlowProperties, advanceFlow
from .quasisteady import advanceQuasiSteady


# Largest values in the strand-burner data of Ma et al. (2020), test/data/ma2020_experimental.json. Mass flux is a rough
# figure that assumes a 25 g/mol combustion gas; the others come straight from the data points.
MA2020_VALIDATED = {'erosiveRatio': 4.1, 'pressure': 17.2e6, 'velocity': 914.0, 'massFlux': 7.8e3}


@dataclass
class BurnResult:
    time: list = field(default_factory=list)
    headPressure: list = field(default_factory=list)
    aftPressure: list = field(default_factory=list)
    massFlow: list = field(default_factory=list)
    burnArea: list = field(default_factory=list)
    pressureProfile: list = field(default_factory=list)
    erosiveRatioProfile: list = field(default_factory=list)
    machProfile: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    crossSectionValid: np.ndarray = None
    peakVelocity: list = field(default_factory=list)
    peakMassFlux: list = field(default_factory=list)
    throatDiameter: list = field(default_factory=list)
    massFluxProfile: list = field(default_factory=list)
    massFlowProfile: list = field(default_factory=list)
    cellCenters: np.ndarray = None
    cellLength: float = None

    def outsideValidatedRange(self):
        """Describes each quantity that exceeded the range covered by the Ma et al. (2020) experimental data."""
        observed = {
            'erosiveRatio': max(float(r.max()) for r in self.erosiveRatioProfile),
            'pressure': max(max(self.headPressure), max(self.aftPressure)),
            'velocity': max(self.peakVelocity),
            'massFlux': max(self.peakMassFlux),
        }
        return [f'{name} reached {observed[name]:.4g}, validated up to {limit:.4g}'
                for name, limit in MA2020_VALIDATED.items() if observed[name] > limit]


def chooseStride(nSlabs: int, targetCellLength: float, voxel: float) -> int:
    """Largest divisor of nSlabs whose cell length does not exceed targetCellLength."""
    limit = max(1, int(targetCellLength / voxel))
    return max(d for d in range(1, min(limit, nSlabs) + 1) if nSlabs % d == 0)


class _FlowGrid:
    """Uniform flow cells laid over the slabs. A cell's value is the exact average of the slabs it covers, so the cell
    length does not have to be a whole number of slabs."""

    def __init__(self, nSlabs, voxel, cells):
        self.cells = int(cells)
        self.voxel = voxel
        self.length = nSlabs * voxel / self.cells
        self.edges = np.arange(self.cells + 1) * (self.length / voxel)  # in slabs
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:]) * voxel

    def average(self, perSlab):
        cumulative = np.concatenate([[0.0], np.cumsum(perSlab)])
        return np.diff(np.interp(self.edges, np.arange(len(perSlab) + 1), cumulative)) / (self.edges[1] - self.edges[0])

    def group(self, aPort, dHyd, burnLength):
        aCell = self.average(aPort)
        wetted = self.average(4.0 * aPort / np.maximum(dHyd, 1e-12))
        return aCell, 4.0 * aCell / np.maximum(wetted, 1e-12), self.average(burnLength)


def _groupGeometry(aPort, dHyd, burnLength, stride):
    return _FlowGrid(len(aPort), 1.0, len(aPort) // stride).group(aPort, dHyd, burnLength)


def _seedSteadyState(aPort, dHyd, burnLength, dx, aThroat, props, pAmbient, pressure, temperature, density, velocity):
    """Start from the normal-burn steady chamber state so the instant-ignition fill transient is skipped."""
    gamma = props.gamma
    gammaFn = np.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    cStar = np.sqrt(props.rSpecific * props.tFlame) / gammaFn
    kn = burnLength.sum() * dx / aThroat
    a, n = props.tabA[0], props.tabN[0]
    p0 = max((props.rhoPropellant * a * cStar * kn) ** (1.0 / (1.0 - n)), pAmbient)
    pressure[:] = p0
    temperature[:] = props.tFlame
    density[:] = p0 / (props.rSpecific * props.tFlame)
    mdot = props.rhoPropellant * a * p0 ** n * np.cumsum(burnLength) * dx
    velocity[1:] = mdot / (density[-1] * np.concatenate([0.5 * (aPort[:-1] + aPort[1:]), aPort[-1:]]))


class BurnSimulation:
    """The coupled burn, held as a state that can be advanced by any duration.

    The gas flows in cells laid over the slabs, and the burn geometry moves at the rate each flow cell gives its slabs.
    `advance()` takes one adaptive span; `advance(duration)` takes as many spans as it needs to cover exactly `duration`
    seconds. Spans are short at start-up and when the pressure is changing, and never move the surface more than
    `maxStepFraction` of a voxel.

    `flow` is 'quasi-steady' for the steady bore flow with chamber storage, which takes substeps of at most
    `quasiSteadyStep` seconds and `quasiSteadyLagFraction` of the shortest burn rate lag, or 'transient' for the acoustic
    PISO solver, which follows the pressure waves and takes microsecond steps.

    The flow cells are `cellLength` metres long when given (rounded to a whole number of cells that tile the grain), or
    `flowStride` slabs. The throat diameter grows by 2 * throatErosionCoeff * P and shrinks by 2 * throatSlagCoeff / P as
    in the openMotor simulation loop; `setThroat` replaces it when a caller tracks the throat itself."""

    def __init__(self, geometry, props: FlowProperties, throatDiameter, pAmbient=101325.0, pCutoff=0.5e6,
                 erosive=True, maxStepFraction=0.75, startupSpan=5e-4, cfl=0.5, dtMax=1e-4,
                 burnUpdateInterval=10, flowStride=1, steadyStart=True, maxSpan=0.05,
                 minSpan=2.5e-4, pressureStep=0.03, throatErosionCoeff=0.0, snapshotInterval=0.0, flow='quasi-steady',
                 quasiSteadyStep=1e-3, quasiSteadyLagFraction=0.25, cellLength=None, throatSlagCoeff=0.0):
        if flow not in ('transient', 'quasi-steady'):
            raise ValueError(f"flow must be 'transient' or 'quasi-steady', got {flow!r}")
        if cellLength is not None:
            cells = int(np.clip(round(geometry.nSlabs * geometry.voxel / cellLength), 1, geometry.nSlabs))
        else:
            if geometry.nSlabs % flowStride:
                raise ValueError(f"flowStride {flowStride} must divide the {geometry.nSlabs} slabs")
            cells = geometry.nSlabs // flowStride
        self.geometry, self.props, self.pAmbient, self.pCutoff, self.erosive = geometry, props, pAmbient, pCutoff, erosive
        self.flow, self.cfl, self.dtMax, self.burnUpdateInterval = flow, cfl, dtMax, burnUpdateInterval
        self.quasiSteadyStep, self.quasiSteadyLagFraction = quasiSteadyStep, quasiSteadyLagFraction
        self.maxStepFraction, self.minSpan, self.maxSpan, self.pressureStep = maxStepFraction, minSpan, maxSpan, pressureStep
        self.throatErosionCoeff, self.throatSlagCoeff = throatErosionCoeff, throatSlagCoeff
        self.flowGrid = _FlowGrid(geometry.nSlabs, geometry.voxel, cells)
        n, dx = self.flowGrid.cells, self.flowGrid.length
        self.throat = throatDiameter
        self.surfaceTables = props.surfaceTables()

        self.pressure = np.full(n, pAmbient)
        self.temperature = np.full(n, props.tInitial)
        self.density = self.pressure / (props.rSpecific * self.temperature)
        self.velocity = np.zeros(n + 1)
        if steadyStart:
            _seedSteadyState(*self.flowGrid.group(*geometry.flowGeometry()), dx, np.pi / 4 * throatDiameter ** 2, props,
                             pAmbient, self.pressure, self.temperature, self.density, self.velocity)
        self.flux = np.zeros(n + 1)
        self.work = {k: np.zeros(n) for k in ('rTotal', 'rErosive', 'rIntegral', 'fDarcy', 'Re', 'Mach', 'uCell', 'rDyn', 'rQs',
                                              'rQsErosive')}
        self.result = BurnResult()
        self.result.cellCenters, self.result.cellLength = self.flowGrid.centers, dx
        self.snapshotInterval, self.nextSnapshot = snapshotInterval, 0.0
        if snapshotInterval > 0.0:
            self.midRow = geometry.phi.shape[1] // 2
            self.result.crossSectionValid = geometry.valid[0, self.midRow].copy()
        self.t = 0.0
        self.span = startupSpan
        self.burning = True
        self.finished = False
        self.previousHead = self.pressure[0]
        self._flowArrays = None

    def setThroat(self, diameter):
        self.throat = diameter

    def flowArrays(self):
        """Port area, hydraulic diameter and burning area per unit length of every flow cell, for the surface as it is now."""
        if self._flowArrays is None:
            self._flowArrays = self.flowGrid.group(*self.geometry.flowGeometry())
        return self._flowArrays

    def burnArea(self):
        return float(self.flowArrays()[2].sum() * self.flowGrid.length)

    def massFlowFaces(self):
        """Mass flow through each cell face, positive toward the nozzle."""
        aPort = self.flowArrays()[0]
        aFace = np.concatenate([aPort[:1], 0.5 * (aPort[:-1] + aPort[1:]), aPort[-1:]])
        rhoFace = np.concatenate([self.density[:1], 0.5 * (self.density[:-1] + self.density[1:]), self.density[-1:]])
        return rhoFace * self.velocity * aFace

    def massFluxCells(self):
        return self.density * np.abs(self.work['uCell'])

    def machCells(self):
        return np.abs(self.work['Mach'])

    def _runSpan(self, span):
        props, work, geometry, flowGrid, result = self.props, self.work, self.geometry, self.flowGrid, self.result
        n, dx, x = flowGrid.cells, flowGrid.length, flowGrid.centers
        aPort, dHyd, burnLength = self.flowArrays()
        burning = bool(burnLength.sum() > 0.0)
        work['rIntegral'][:] = 0.0
        common = (self.density, self.velocity, self.pressure, self.temperature, aPort, dHyd, burnLength, x,
                  work['rTotal'], work['rErosive'], work['rIntegral'], work['fDarcy'], work['Re'], work['Mach'], work['uCell'],
                  dx, n, np.pi / 4 * self.throat ** 2, self.pAmbient, props.tInitial,
                  props.rhoPropellant, props.tInitial, props.tFlame,
                  props.cpGas, props.muGas, props.kGas, props.prandtl, props.gamma, props.rSpecific,
                  props.roughness, props.kappa, props.tabMinP, props.tabMaxP, props.tabA, props.tabN, len(props.tabA))
        burnState = (work['rDyn'], work['rQs'], work['rQsErosive'], props.solidDiffusivity, props.kappaZn, *self.surfaceTables)
        if self.flow == 'quasi-steady':
            elapsed, massNozzle, _ = advanceQuasiSteady(*common, self.erosive, self.quasiSteadyStep,
                                                        self.quasiSteadyLagFraction, span, *burnState, self.flux)
        else:
            elapsed, massNozzle, _ = advanceFlow(*common, self.erosive, self.cfl, self.dtMax, self.burnUpdateInterval, 0.0,
                                                 span, 100_000_000, *burnState)
        pressure = self.pressure
        self.t += elapsed
        self.throat += (2.0 * self.throatErosionCoeff * pressure[0] - 2.0 * self.throatSlagCoeff / max(pressure[0], 1.0)) * elapsed
        rAverage = work['rIntegral'] / elapsed
        if burning:
            geometry.advance(np.interp(geometry.slabCenters, x, rAverage), elapsed)
            self._flowArrays = None
        result.time.append(self.t)
        result.headPressure.append(pressure[0])
        result.aftPressure.append(pressure[-1])
        result.massFlow.append(massNozzle / elapsed)
        result.burnArea.append(float(burnLength.sum() * dx))
        result.pressureProfile.append(pressure.copy())
        result.erosiveRatioProfile.append(work['rErosive'] / np.maximum(work['rTotal'] - work['rErosive'], 1e-12))
        result.machProfile.append(self.machCells().copy())
        if self.snapshotInterval > 0.0 and self.t >= self.nextSnapshot:
            result.snapshots.append({'time': self.t, 'section': geometry.phi[:, self.midRow].astype(np.float32),
                                     'mach': self.machCells().copy(), 'pressure': pressure.copy()})
            self.nextSnapshot += self.snapshotInterval
        result.peakVelocity.append(float(np.abs(work['uCell']).max()))
        result.peakMassFlux.append(float(self.massFluxCells().max()))
        result.throatDiameter.append(self.throat)
        result.massFlowProfile.append(self.massFlowFaces())
        result.massFluxProfile.append(self.massFluxCells().copy())
        self.burning = burning
        self.finished = not burning and pressure[0] < self.pCutoff
        if not self.finished:
            rMax = float(rAverage.max())
            regressionSpan = self.maxStepFraction * geometry.voxel / rMax if burning and rMax > 1e-9 else 5e-3
            change = abs(pressure[0] - self.previousHead) / max(self.previousHead, 1.0)
            self.span = self.span * 0.5 if change > self.pressureStep else self.span * 1.5
            self.span = min(max(self.span, self.minSpan), regressionSpan, self.maxSpan)
            self.previousHead = pressure[0]
        return elapsed

    def advance(self, duration=None):
        """One adaptive span when `duration` is None, otherwise as many spans as cover exactly `duration` seconds."""
        if duration is None:
            self._runSpan(self.span)
            return
        end = self.t + duration
        while end - self.t > 1e-12 and not self.finished:
            span = min(self.span, end - self.t)
            if 0.0 < end - self.t - span < self.minSpan:  # never leave a sliver of a span at the end of the duration
                span = end - self.t
            self._runSpan(span)


def runBurn(geometry, props: FlowProperties, throatDiameter, pAmbient=101325.0, tMax=60.0, pCutoff=0.5e6,
            erosive=True, maxStepFraction=0.75, startupSpan=5e-4, cfl=0.5, dtMax=1e-4,
            burnUpdateInterval=10, flowStride=1, steadyStart=True, maxSpan=0.05,
            minSpan=2.5e-4, pressureStep=0.03, throatErosionCoeff=0.0, snapshotInterval=0.0, flow='quasi-steady',
            quasiSteadyStep=1e-3, quasiSteadyLagFraction=0.25, cellLength=None, throatSlagCoeff=0.0, observer=None) -> BurnResult:
    """Runs a `BurnSimulation` span by span until `tMax`, until the surface is gone and the pressure has fallen to
    `pCutoff`, or until `observer(result, geometry)` returns true. See `BurnSimulation` for the other arguments."""
    sim = BurnSimulation(geometry, props, throatDiameter, pAmbient, pCutoff, erosive, maxStepFraction, startupSpan, cfl,
                         dtMax, burnUpdateInterval, flowStride, steadyStart, maxSpan, minSpan, pressureStep,
                         throatErosionCoeff, snapshotInterval, flow, quasiSteadyStep, quasiSteadyLagFraction, cellLength,
                         throatSlagCoeff)
    while sim.t < tMax:
        sim.advance()
        if observer is not None and observer(sim.result, geometry):
            break
        if sim.finished:
            break
    return sim.result
