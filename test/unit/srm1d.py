import json
import os
import unittest
import numpy as np

from motorlib.grains import Finocyl

from motorlib.srm1d import FlowProperties, advanceFlow, BurnbackGrid, runBurn, chooseStride
from motorlib.srm1d.coupled import MA2020_VALIDATED

_N_EXP = 0.35
_A = 0.005 / 6e6 ** _N_EXP
_BORE, _LENGTH, _THROAT, _CELLS = 0.05, 0.5, 0.02, 40


def _flowProperties():
    return FlowProperties(rhoPropellant=1700., cpPropellant=1500., tFlame=2800., tSurface=800., tInitial=300.,
                          gamma=1.2, molarMass=0.025, cpGas=1995., muGas=8e-5, kGas=0.4,
                          tabMinP=np.array([0.]), tabMaxP=np.array([1e9]), tabA=np.array([_A]), tabN=np.array([_N_EXP]))


def _runStraightBore(erosive):
    fp = _flowProperties()
    n, dx = _CELLS, _LENGTH / _CELLS
    x = (np.arange(n) + 0.5) * dx
    aPort = np.full(n, np.pi / 4 * _BORE ** 2)
    P = np.full(n, 101325.)
    T = np.full(n, 300.)
    rho = P / (fp.rSpecific * T)
    u = np.zeros(n + 1)
    work = [np.zeros(n) for _ in range(10)]
    advanceFlow(rho, u, P, T, aPort, np.full(n, _BORE), np.full(n, np.pi * _BORE), x, *work[:7],
                dx, n, np.pi / 4 * _THROAT ** 2, 101325., 300.,
                fp.rhoPropellant, fp.cpPropellant, fp.tSurface, fp.tInitial, fp.tFlame,
                fp.cpGas, fp.muGas, fp.kGas, fp.prandtl, fp.gamma, fp.rSpecific, fp.roughness, fp.kappa,
                fp.tabMinP, fp.tabMaxP, fp.tabA, fp.tabN, 1, erosive, 0.5, 1e-4, 1, 0.0, 0.3, 10_000_000,
                *work[7:], fp.solidDiffusivity, fp.kappaZn)
    return fp, P, work[0], work[1]


class TestStraightBoreSteadyState(unittest.TestCase):
    def testNormalBurnMatchesAnalyticChamberPressure(self):
        fp, P, _, _ = _runStraightBore(erosive=False)
        g = fp.gamma
        gammaFn = np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        cStar = np.sqrt(fp.rSpecific * fp.tFlame) / gammaFn
        kn = np.pi * _BORE * _LENGTH / (np.pi / 4 * _THROAT ** 2)
        expected = (fp.rhoPropellant * _A * cStar * kn) ** (1 / (1 - _N_EXP))
        self.assertAlmostEqual(P[-1] / expected, 1.0, delta=0.01)

    def testErosiveBurningRaisesPressureAndPeaksAft(self):
        _, pNormal, _, _ = _runStraightBore(erosive=False)
        _, pErosive, rTotal, rErosive = _runStraightBore(erosive=True)
        self.assertGreater(pErosive[-1], 1.03 * pNormal[-1])
        ratio = rErosive / rTotal
        self.assertLess(ratio[0], 0.01)
        self.assertTrue(np.all(np.diff(ratio) >= -1e-6))


_CASTING, _VOXEL_COUNT, _SLICES, _GRID_BORE = 0.08, 32, 21, 0.015


def _cylinderField():
    voxel = _CASTING / _VOXEL_COUNT
    n = _VOXEL_COUNT + 2
    centre = (n - 1) / 2
    y, x = np.meshgrid(np.arange(n) - centre, np.arange(n) - centre, indexing='ij')
    radius = np.hypot(x, y) * voxel
    return np.broadcast_to(radius - _GRID_BORE, (_SLICES, n, n)), radius <= _CASTING / 2, voxel


class _AnalyticBore:
    def __init__(self, voxel):
        self.voxel = voxel
        self.nSlabs = _SLICES - 1
        self.slabCenters = (np.arange(self.nSlabs) + 0.5) * voxel
        self.radius = np.full(self.nSlabs, _GRID_BORE)

    def flowGeometry(self):
        wall = _CASTING / 2
        open_ = np.pi * np.minimum(self.radius, wall) ** 2
        burning = self.radius < wall
        return open_, np.where(burning, 2 * self.radius, _CASTING), np.where(burning, 2 * np.pi * self.radius, 0.0)

    def advance(self, slabSpeed, dt):
        self.radius = self.radius + slabSpeed * dt


class TestBurnbackGrid(unittest.TestCase):
    def setUp(self):
        phi, valid, self.voxel = _cylinderField()
        self.grid = BurnbackGrid(phi, valid, self.voxel)

    def testInitialGeometryMatchesCylinder(self):
        aPort, dHyd, burnLength = self.grid.flowGeometry()
        self.assertAlmostEqual(burnLength.mean() / (2 * np.pi * _GRID_BORE), 1.0, delta=0.03)
        self.assertAlmostEqual(aPort.mean() / (np.pi * _GRID_BORE ** 2), 1.0, delta=0.03)
        self.assertAlmostEqual(dHyd.mean() / (2 * _GRID_BORE), 1.0, delta=0.03)

    def testUniformSpeedGrowsBoreByNormalDistance(self):
        for _ in range(6):
            self.grid.advance(np.full(self.grid.nSlabs, 0.01), 0.1)
        aPort = self.grid.flowGeometry()[0]
        self.assertAlmostEqual(np.sqrt(aPort.mean() / np.pi), _GRID_BORE + 0.006, delta=0.5 * self.voxel)

    def testNoBurningSurfaceRemainsAfterBurnout(self):
        for _ in range(60):
            self.grid.advance(np.full(self.grid.nSlabs, 0.01), 0.1)
        aPort, _, burnLength = self.grid.flowGeometry()
        self.assertEqual(burnLength.sum(), 0.0)
        self.assertAlmostEqual(aPort.mean() / (np.pi * (_CASTING / 2) ** 2), 1.0, delta=0.03)


class TestCoupledBurn(unittest.TestCase):
    def testVoxelGeometryMatchesAnalyticPressureTrace(self):
        phi, valid, voxel = _cylinderField()
        kwargs = dict(props=_flowProperties(), throatDiameter=0.0064, tMax=0.3, pCutoff=0.0)
        voxelResult = runBurn(BurnbackGrid(phi, valid, voxel), **kwargs)
        analyticResult = runBurn(_AnalyticBore(voxel), **kwargs)
        for t in (0.15, 0.3):
            voxelP = np.interp(t, voxelResult.time, voxelResult.headPressure)
            analyticP = np.interp(t, analyticResult.time, analyticResult.headPressure)
            self.assertAlmostEqual(voxelP / analyticP, 1.0, delta=0.03)

    def testGroupingSlabsIntoFlowCellsPreservesPressureTrace(self):
        phi, valid, voxel = _cylinderField()
        kwargs = dict(props=_flowProperties(), throatDiameter=0.0064, tMax=0.3, pCutoff=0.0)
        fine = runBurn(BurnbackGrid(phi, valid, voxel), flowStride=1, **kwargs)
        coarse = runBurn(BurnbackGrid(phi, valid, voxel), flowStride=chooseStride(_SLICES - 1, 5 * voxel, voxel), **kwargs)
        self.assertAlmostEqual(coarse.headPressure[-1] / fine.headPressure[-1], 1.0, delta=0.01)


class TestFinocylBurnback(unittest.TestCase):
    MAP_DIM, DIAMETER = 96, 0.08

    @classmethod
    def setUpClass(cls):
        cls.reference = Finocyl()
        cls.reference.setProperties({'diameter': cls.DIAMETER, 'length': 0.3, 'coreDiameter': 0.02, 'numFins': 4,
                                     'finWidth': 0.004, 'finLength': 0.015, 'invertedFins': False,
                                     'inhibitedEnds': 'Both'})
        cls.reference.initGeometry(cls.MAP_DIM)
        cls.reference.generateCoreMap()
        cls.reference.generateRegressionMap()

    def _extrudedGrid(self):
        propellant = np.broadcast_to(self.reference.coreMap.astype(bool), (13,) + self.reference.coreMap.shape)
        return BurnbackGrid.fromMask(propellant.copy(), ~self.reference.mask, self.DIAMETER / self.MAP_DIM)

    def testPerimeterAndFaceAreaTrackTheTwoDimensionalFmmReference(self):
        grid = self._extrudedGrid()
        casting = np.count_nonzero(grid.valid[0]) * grid.voxel ** 2
        regressed = 0.0
        for target in (0.0, 0.006, 0.012):
            if target > regressed:
                grid.advance(np.full(grid.nSlabs, 0.01), (target - regressed) / 0.01)
                regressed = target
            aPort, _, burnLength = grid.flowGeometry()
            mid = grid.nSlabs // 2
            self.assertAlmostEqual(burnLength[mid] / self.reference.getCorePerimeter(target), 1.0, delta=0.02)
            self.assertAlmostEqual((casting - aPort[mid]) / self.reference.getFaceArea(target), 1.0, delta=0.03)

    def testNoPropellantOrBurningSurfaceRemainsAfterBurnout(self):
        grid = self._extrudedGrid()
        for _ in range(60):
            grid.advance(np.full(grid.nSlabs, 0.01), 0.2)
        self.assertEqual(np.count_nonzero((grid.phi > 0) & grid.valid), 0)
        self.assertEqual(grid.flowGeometry()[2].sum(), 0.0)


_MA2020 = os.path.join(os.path.dirname(__file__), '..', 'data', 'ma2020_experimental.json')


class TestMa2020ExperimentalData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(_MA2020) as f:
            cls.data = json.load(f)
        cls.figures = cls.data['figures']

    def testAllElevenFiguresPresentWithAssumedUncertainty(self):
        self.assertEqual(sorted(self.figures), sorted(['Fig%d' % n for n in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)]))
        self.assertEqual(set(self.data['uncertainty']['assumedRelative']), {'note', 'burnRate', 'pressure', 'timeSeconds'})

    def testNormalBurnRateIncreasesWithPressureAsAPowerLaw(self):
        for name in ('Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig8', 'Fig9'):
            pressure, rate = np.array(self.figures[name]['gasVelocityMetersPerSecond']['0']).T
            exponent = np.polyfit(np.log(pressure), np.log(rate), 1)[0]
            self.assertGreater(exponent, 0.1, name)
            self.assertLess(exponent, 0.9, name)

    def testErosiveBurnRateExceedsNormalBurnRateAtTheSamePressure(self):
        for name in ('Fig1', 'Fig2', 'Fig3', 'Fig8', 'Fig9'):
            series = self.figures[name]['gasVelocityMetersPerSecond']
            baseP, baseR = np.array(series['0']).T
            order = np.argsort(baseP)
            fastest = max(series, key=float)
            pressure, rate = np.array(series[fastest]).T
            normal = np.exp(np.interp(np.log(pressure), np.log(baseP[order]), np.log(baseR[order])))
            self.assertGreater(np.median(rate / normal), 1.1, name)

    def testHasegawaPressureHistoriesMatchPublishedPeaks(self):
        peaks = {'Fig10': 6.44, 'Fig11': 6.16, 'Fig12': 4.52}
        for name, peak in peaks.items():
            pressure = np.array(self.figures[name]['points'])[:, 1]
            self.assertAlmostEqual(pressure.max(), peak, delta=0.02, msg=name)
            self.assertLess(pressure[-1], 0.5, msg=name)

    def testHasegawaMotorAMatchesAnIndependentHandDigitization(self):
        points = np.array(self.figures['Fig10']['points'])
        handDigitized = {2.348: 5.990, 3.036: 3.550, 4.113: 1.219}
        for t, pressure in handDigitized.items():
            self.assertAlmostEqual(np.interp(t - 0.052, points[:, 0], points[:, 1]), pressure, delta=0.05)

    def testValidatedRangeConstantsMatchTheDataPoints(self):
        ratios, pressures, velocities = [], [], []
        for name in ('Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig8', 'Fig9'):
            series = self.figures[name]['gasVelocityMetersPerSecond']
            baseP, baseR = np.array(series['0']).T
            order = np.argsort(baseP)
            for velocity, points in series.items():
                if velocity == '0':
                    continue
                for pressure, rate in points:
                    normal = np.exp(np.interp(np.log(pressure), np.log(baseP[order]), np.log(baseR[order])))
                    ratios.append(rate / normal)
                    pressures.append(pressure)
                    velocities.append(float(velocity))
        self.assertAlmostEqual(max(ratios), MA2020_VALIDATED['erosiveRatio'], delta=0.05)
        self.assertAlmostEqual(max(pressures) * 1e6, MA2020_VALIDATED['pressure'], delta=0.05e6)
        self.assertEqual(max(velocities), MA2020_VALIDATED['velocity'])
