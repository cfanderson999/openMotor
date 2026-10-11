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


def _groupGeometry(aPort, dHyd, burnLength, stride):
    n = len(aPort) // stride
    aCell = aPort.reshape(n, stride).mean(axis=1)
    wetted = (4.0 * aPort / np.maximum(dHyd, 1e-12)).reshape(n, stride).mean(axis=1)
    return aCell, 4.0 * aCell / np.maximum(wetted, 1e-12), burnLength.reshape(n, stride).mean(axis=1)


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


def runBurn(geometry, props: FlowProperties, throatDiameter, pAmbient=101325.0, tMax=60.0, pCutoff=0.5e6,
            erosive=True, maxStepFraction=0.75, startupSpan=5e-4, cfl=0.5, dtMax=1e-4,
            burnUpdateInterval=10, flowStride=1, steadyStart=True, maxSpan=0.05,
            minSpan=2.5e-4, pressureStep=0.03, throatErosionCoeff=0.0, snapshotInterval=0.0, flow='quasi-steady',
            quasiSteadyStep=1e-3, quasiSteadyLagFraction=0.25) -> BurnResult:
    """`flow` is 'quasi-steady' for the steady bore flow with chamber storage, which takes substeps of at most
    `quasiSteadyStep` seconds and `quasiSteadyLagFraction` of the shortest burn rate lag, or 'transient' for the acoustic
    PISO solver, which follows the pressure waves and takes microsecond steps."""
    if flow not in ('transient', 'quasi-steady'):
        raise ValueError(f"flow must be 'transient' or 'quasi-steady', got {flow!r}")
    if geometry.nSlabs % flowStride:
        raise ValueError(f"flowStride {flowStride} must divide the {geometry.nSlabs} slabs")
    n = geometry.nSlabs // flowStride
    dx = geometry.voxel * flowStride
    x = geometry.slabCenters.reshape(n, flowStride).mean(axis=1)
    aThroat = np.pi / 4 * throatDiameter ** 2
    throat = throatDiameter
    tAmbient = props.tInitial
    surfaceTables = props.surfaceTables()

    pressure = np.full(n, pAmbient)
    temperature = np.full(n, tAmbient)
    density = pressure / (props.rSpecific * temperature)
    velocity = np.zeros(n + 1)
    if steadyStart:
        _seedSteadyState(*_groupGeometry(*geometry.flowGeometry(), flowStride), dx, aThroat, props, pAmbient,
                         pressure, temperature, density, velocity)
    flux = np.zeros(n + 1)
    work = {k: np.zeros(n) for k in ('rTotal', 'rErosive', 'rIntegral', 'fDarcy', 'Re', 'Mach', 'uCell', 'rDyn', 'rQs', 'rQsErosive')}

    result = BurnResult()
    nextSnapshot = 0.0
    if snapshotInterval > 0.0:
        midRow = geometry.phi.shape[1] // 2
        result.crossSectionValid = geometry.valid[0, midRow].copy()
    t = 0.0
    span = startupSpan
    burning = True
    peak = 0.0
    previousHead = pressure[0]
    while t < tMax:
        aThroat = np.pi / 4 * throat ** 2
        aPort, dHyd, burnLength = _groupGeometry(*geometry.flowGeometry(), flowStride)
        burning = bool(burnLength.sum() > 0.0)
        work['rIntegral'][:] = 0.0
        common = (density, velocity, pressure, temperature, aPort, dHyd, burnLength, x,
                  work['rTotal'], work['rErosive'], work['rIntegral'], work['fDarcy'], work['Re'], work['Mach'], work['uCell'],
                  dx, n, aThroat, pAmbient, tAmbient,
                  props.rhoPropellant, props.tInitial, props.tFlame,
                  props.cpGas, props.muGas, props.kGas, props.prandtl, props.gamma, props.rSpecific,
                  props.roughness, props.kappa, props.tabMinP, props.tabMaxP, props.tabA, props.tabN, len(props.tabA))
        burnState = (work['rDyn'], work['rQs'], work['rQsErosive'], props.solidDiffusivity, props.kappaZn, *surfaceTables)
        if flow == 'quasi-steady':
            elapsed, massNozzle, _ = advanceQuasiSteady(*common, erosive, quasiSteadyStep, quasiSteadyLagFraction, span, *burnState, flux)
        else:
            elapsed, massNozzle, _ = advanceFlow(*common, erosive, cfl, dtMax, burnUpdateInterval, 0.0, span, 100_000_000, *burnState)
        t += elapsed
        throat += 2.0 * throatErosionCoeff * pressure[0] * elapsed
        rAverage = work['rIntegral'] / elapsed
        if burning:
            geometry.advance(np.interp(geometry.slabCenters, x, rAverage), elapsed)
        peak = max(peak, pressure[0])
        result.time.append(t)
        result.headPressure.append(pressure[0])
        result.aftPressure.append(pressure[-1])
        result.massFlow.append(massNozzle / elapsed)
        result.burnArea.append(float(burnLength.sum() * dx))
        result.pressureProfile.append(pressure.copy())
        result.erosiveRatioProfile.append(work['rErosive'] / np.maximum(work['rTotal'] - work['rErosive'], 1e-12))
        result.machProfile.append(np.abs(work['Mach']).copy())
        if snapshotInterval > 0.0 and t >= nextSnapshot:
            result.snapshots.append({'time': t, 'section': geometry.phi[:, midRow].astype(np.float32), 'mach': np.abs(work['Mach']).copy(),
                                     'pressure': pressure.copy()})
            nextSnapshot += snapshotInterval
        result.peakVelocity.append(float(np.abs(work['uCell']).max()))
        result.peakMassFlux.append(float((density * np.abs(work['uCell'])).max()))
        if not burning and pressure[0] < pCutoff:
            break
        rMax = float(rAverage.max())
        regressionSpan = maxStepFraction * geometry.voxel / rMax if burning and rMax > 1e-9 else 5e-3
        change = abs(pressure[0] - previousHead) / max(previousHead, 1.0)
        span = span * 0.5 if change > pressureStep else span * 1.5
        span = min(max(span, minSpan), regressionSpan, maxSpan)
        previousHead = pressure[0]
    return result
