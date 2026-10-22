import json
import os
import unittest
import numpy as np

from motorlib.grains import Finocyl

from motorlib.srm1d import FlowProperties, advanceFlow, BurnbackGrid, BurnSimulation, runBurn, chooseStride
from motorlib.srm1d.burnback import MIN_HYDRAULIC_DIAMETER, MIN_PORT_AREA
from motorlib.srm1d.quasisteady import advanceQuasiSteady
from motorlib.srm1d.coupled import _seedSteadyState, _FlowGrid
from motorlib.srm1d.coupled import MA2020_VALIDATED
from motorlib.srm1d.burn_rate import surface_temperature, erosive_energy_factor, compute_burn_rates
from motorlib.srm1d.surface import (apSurfaceTemperature, binderSurfaceTemperature, surfaceTables, apSurfaceEnthalpy,
                                    binderSurfaceEnthalpy, metalEnthalpy, DEFAULT_AP_VOLUME_FRACTION)

_N_EXP = 0.35
_A = 0.005 / 6e6 ** _N_EXP
_BORE, _LENGTH, _THROAT, _CELLS = 0.05, 0.5, 0.02, 40


def _flowProperties():
    return FlowProperties(rhoPropellant=1700., cpPropellant=1500., tFlame=2800., tInitial=300.,
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
                fp.rhoPropellant, fp.tInitial, fp.tFlame,
                fp.cpGas, fp.muGas, fp.kGas, fp.prandtl, fp.gamma, fp.rSpecific, fp.roughness, fp.kappa,
                fp.tabMinP, fp.tabMaxP, fp.tabA, fp.tabN, 1, erosive, 0.5, 1e-4, 1, 0.0, 0.3, 10_000_000,
                *work[7:], fp.solidDiffusivity, fp.kappaZn, *fp.surfaceTables())
    return fp, P, work[0], work[1]


class TestSurfaceTemperature(unittest.TestCase):
    def testApLawMatchesLengelleAtTenMillimetresPerSecond(self):
        self.assertAlmostEqual(apSurfaceTemperature(0.010), 925., delta=10.)

    def testBinderLawMatchesQuotedPyrometryPoint(self):
        self.assertAlmostEqual(binderSurfaceTemperature(0.4e-3, 293.), 1060., delta=1.)

    def testSurfaceRunsHotterAtHigherBurnRate(self):
        rates = [0.002, 0.005, 0.01, 0.02]
        for law in (apSurfaceTemperature, lambda r: binderSurfaceTemperature(r, 293.)):
            self.assertTrue(np.all(np.diff([law(r) for r in rates]) > 0))

    def testTableBlendsIngredientsByVolumeFraction(self):
        lnRate, temp, _ = surfaceTables(0.8, 293., cps=1500.)
        rate = 0.01
        expected = 0.8 * apSurfaceTemperature(rate) + 0.2 * binderSurfaceTemperature(rate, 293.)
        self.assertAlmostEqual(surface_temperature(rate, lnRate, temp), expected, delta=1.)

    def testPureApTableEqualsApLaw(self):
        lnRate, temp, _ = surfaceTables(1.0, 293., cps=1500.)
        self.assertAlmostEqual(surface_temperature(0.007, lnRate, temp), apSurfaceTemperature(0.007), delta=0.5)

    def testDefaultUsesDocumentedApVolumeFraction(self):
        fp = _flowProperties()
        self.assertEqual(fp.apVolumeFraction, DEFAULT_AP_VOLUME_FRACTION)
        lnRate, temp, _ = fp.surfaceTables()
        expected = surfaceTables(DEFAULT_AP_VOLUME_FRACTION, fp.tInitial, cps=fp.cpPropellant)[1]
        self.assertTrue(np.array_equal(temp, expected))

    def testErosiveGainFallsAsBurnRateRises(self):
        lnRate, temp, enthalpy = surfaceTables(0.8, 293., cps=1500.)
        gains = [erosive_energy_factor(r, 2800., 1700., lnRate, temp, enthalpy) for r in (0.005, 0.01, 0.02)]
        self.assertTrue(np.all(np.diff(gains) < 0))


class TestSurfaceEnergy(unittest.TestCase):
    def testSensibleTableIsCpTimesTemperatureRise(self):
        _, temp, enthalpy = surfaceTables(0.6, 293., cps=1600.)
        self.assertTrue(np.allclose(enthalpy, 1600. * (temp - 293.)))

    def testApEnergyMatchesLengelleAtTenMillimetresPerSecond(self):
        self.assertAlmostEqual(apSurfaceEnthalpy(925.), 175. * 4184., delta=1.)

    def testBinderEnergyMatchesAblationHeatAtQuotedPoint(self):
        self.assertAlmostEqual(binderSurfaceEnthalpy(1060., 293.) / 4184., 749., delta=2.)

    def testMetalEnergyIncludesFusion(self):
        jump = metalEnthalpy(1000., 293.) - metalEnthalpy(900., 293.)
        self.assertGreater(jump, 5.0e5)

    def testPureApCondensedTableEqualsApEnergy(self):
        lnRate, temp, enthalpy = surfaceTables(1.0, 293., None, 1.0, 0.0)
        i = np.abs(np.exp(lnRate) - 0.01).argmin()
        self.assertAlmostEqual(enthalpy[i], apSurfaceEnthalpy(apSurfaceTemperature(np.exp(lnRate[i]))), delta=1.)

    def testCondensedEnergyIsMassWeightedSum(self):
        lnRate, _, enthalpy = surfaceTables(0.6, 293., None, 0.7, 0.16)
        r = np.exp(lnRate[64]); tA = apSurfaceTemperature(r); tB = binderSurfaceTemperature(r, 293.)
        expected = 0.7 * apSurfaceEnthalpy(tA) + 0.16 * binderSurfaceEnthalpy(tB, 293.) + 0.14 * metalEnthalpy(tB, 293.)
        self.assertAlmostEqual(enthalpy[64], expected, delta=1.)

    def testMassFractionsAboveOneAreRejected(self):
        with self.assertRaises(ValueError):
            surfaceTables(0.6, 293., None, 0.8, 0.3)

    def testMetalSpeciesEnergyDiffersFromAl(self):
        # Zn's enthalpy table is its own (lower, from CEA Zn(cr)/Zn(L)), not Al's.
        self.assertLess(metalEnthalpy(1500., 293., 'Zn'), 0.5 * metalEnthalpy(1500., 293., 'Al'))

    def testCondensedEnergyUsesPerMetalSpecies(self):
        # Fractions sum to exactly one so no leftover falls back to Al.
        lnRate, _, enthalpy = surfaceTables(0.6, 293., None, 0.5, 0.15, {'Zn': 0.20, 'Mg': 0.15})
        r = np.exp(lnRate[64]); tA = apSurfaceTemperature(r); tB = binderSurfaceTemperature(r, 293.)
        expected = (0.5 * apSurfaceEnthalpy(tA) + 0.15 * binderSurfaceEnthalpy(tB, 293.)
                    + 0.20 * metalEnthalpy(tB, 293., 'Zn') + 0.15 * metalEnthalpy(tB, 293., 'Mg'))
        self.assertAlmostEqual(enthalpy[64], expected, delta=1.)

    def testUnassignedMetalRemainderFallsBackToAl(self):
        lnRate, _, enthalpy = surfaceTables(0.6, 293., None, 0.7, 0.16, {'Zn': 0.05})
        r = np.exp(lnRate[64]); tA = apSurfaceTemperature(r); tB = binderSurfaceTemperature(r, 293.)
        expected = (0.7 * apSurfaceEnthalpy(tA) + 0.16 * binderSurfaceEnthalpy(tB, 293.)
                    + 0.05 * metalEnthalpy(tB, 293., 'Zn') + 0.09 * metalEnthalpy(tB, 293., 'Al'))
        self.assertAlmostEqual(enthalpy[64], expected, delta=1.)

    def testMetalSpeciesFractionsAboveRemainderAreRejected(self):
        with self.assertRaises(ValueError):
            surfaceTables(0.6, 293., None, 0.7, 0.16, {'Zn': 0.50})


class TestBurnRateRootFind(unittest.TestCase):
    """The erosive burn rate is the root of a monotone residual, found from any starting guess."""

    @classmethod
    def setUpClass(cls):
        cls.fp = _flowProperties()
        cls.tables = cls.fp.surfaceTables()
        rng = np.random.default_rng(5)
        cls.n = 60
        cls.pressure = rng.uniform(2e6, 1.4e7, cls.n)
        cls.reynolds = 10 ** rng.uniform(4.5, 6.5, cls.n)
        cls.diameter = rng.uniform(0.02, 0.04, cls.n)
        cls.x = np.linspace(0.01, 1.0, cls.n)

    def _rates(self, guess):
        fp = self.fp
        return compute_burn_rates(
            self.pressure, self.reynolds, self.diameter, self.x, np.ones(self.n, dtype=bool), fp.roughness, fp.prandtl,
            fp.kGas, fp.cpGas, fp.tFlame, fp.rhoPropellant, fp.tInitial, fp.tabMinP, fp.tabMaxP, fp.tabA, fp.tabN, 1,
            fp.kappa, self.n, *self.tables, guess,
        )

    def testRootIsIndependentOfTheStartingGuess(self):
        cold, _ = self._rates(np.zeros(self.n))
        for scale in (1.001, 0.9, 3.0):
            warm, _ = self._rates(cold * scale)
            np.testing.assert_allclose(warm, cold, rtol=0, atol=2e-8)

    def testErosiveRateIsTheTotalMinusTheNormalRate(self):
        total, erosive = self._rates(np.zeros(self.n))
        normal = self.fp.tabA[0] * self.pressure ** self.fp.tabN[0]
        np.testing.assert_allclose(total - erosive, normal, rtol=1e-9)
        self.assertTrue(np.all(erosive >= 0.0))
        self.assertGreater(float(erosive.max()), 0.0)


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
        self.assertGreater(pErosive[-1], 1.015 * pNormal[-1])
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


_CAP_SLICES = 6


def _blindBoreGrid():
    """The cylinder bore with a solid cap of `_CAP_SLICES` slices at the head end, which burns where the bore opens onto it."""
    phi, valid, voxel = _cylinderField()
    propellant = phi > 0
    propellant[:_CAP_SLICES] = True
    return BurnbackGrid.fromMask(propellant, valid, voxel), voxel


class TestBlindBore(unittest.TestCase):
    def test_slabs_without_a_bore_get_a_negligible_one_so_the_flow_stays_finite(self):
        grid, _ = _blindBoreGrid()
        aPort, dHyd, burnLength = grid.flowGeometry()
        for values in (aPort, dHyd, burnLength):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(aPort >= MIN_PORT_AREA))
        self.assertTrue(np.all(dHyd >= MIN_HYDRAULIC_DIAMETER))
        solid = slice(0, _CAP_SLICES - 2)
        np.testing.assert_allclose(aPort[solid], MIN_PORT_AREA)
        np.testing.assert_allclose(dHyd[solid], MIN_HYDRAULIC_DIAMETER)
        self.assertEqual(float(burnLength[solid].sum()), 0.0)
        self.assertGreater(float(burnLength[_CAP_SLICES - 2:_CAP_SLICES + 1].sum()), 0.0)

    def test_the_bore_opening_onto_the_cap_burns_the_cap_in_place_of_the_wall_it_covers(self):
        grid, voxel = _blindBoreGrid()
        openBore = _cylinderField()[0] > 0
        plain = BurnbackGrid.fromMask(openBore, grid.valid[0], voxel)
        change = float(grid.slabBurnArea().sum() - plain.slabBurnArea().sum())
        # The cap trades the bore wall it covers for the disc of the bore's cross section; the wall is the larger of the two
        expected = np.pi * _GRID_BORE ** 2 - 2 * np.pi * _GRID_BORE * _CAP_SLICES * voxel
        self.assertAlmostEqual((change - expected) / (np.pi * _GRID_BORE ** 2), 0.0, delta=0.35)

    def test_the_steady_pressure_follows_the_burning_area_with_a_cap_in_the_flow(self):
        grid, _ = _blindBoreGrid()
        props = _flowProperties()
        result = runBurn(grid, props, throatDiameter=0.0064, tMax=0.01, pCutoff=0.0)
        self.assertTrue(np.all(np.isfinite(result.headPressure)))
        self.assertTrue(np.all(np.isfinite(result.aftPressure)))
        g = props.gamma
        cStar = np.sqrt(props.rSpecific * props.tFlame) / (np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))
        kn = grid.slabBurnArea().sum() / (np.pi / 4 * 0.0064 ** 2)
        expected = (props.rhoPropellant * _A * cStar * kn) ** (1 / (1 - _N_EXP))
        self.assertAlmostEqual(result.aftPressure[0] / expected, 1.0, delta=0.02)

    def test_the_whole_burn_stays_finite_as_the_cap_burns_away(self):
        grid, _ = _blindBoreGrid()
        result = runBurn(grid, _flowProperties(), throatDiameter=0.0064, tMax=0.5, pCutoff=0.0)
        self.assertGreater(result.time[-1], 0.2)
        for series in (result.headPressure, result.aftPressure, result.massFlow, result.peakVelocity):
            self.assertTrue(np.all(np.isfinite(series)))


class TestFlowGrid(unittest.TestCase):
    def test_cells_that_are_whole_slab_groups_average_like_a_reshape(self):
        grid = _FlowGrid(12, 0.5, 4)
        perSlab = np.arange(12.0)
        np.testing.assert_allclose(grid.average(perSlab), perSlab.reshape(4, 3).mean(axis=1))
        np.testing.assert_allclose(grid.centers, 0.5 * (np.arange(4) * 3 + 1.5))

    def test_cells_that_split_slabs_keep_the_total(self):
        grid = _FlowGrid(10, 1.0, 4)
        perSlab = np.random.default_rng(2).random(10)
        self.assertAlmostEqual(float(grid.average(perSlab).sum() * grid.length), float(perSlab.sum()), places=12)
        self.assertAlmostEqual(grid.length, 2.5)


class TestBurnSimulation(unittest.TestCase):
    def _sim(self, **kwargs):
        phi, valid, voxel = _cylinderField()
        return BurnSimulation(_AnalyticBore(voxel), _flowProperties(), 0.0064, **kwargs), voxel

    def test_advancing_by_a_duration_lands_on_it(self):
        sim, _ = self._sim()
        for duration in (0.03, 0.0123, 0.05):
            before = sim.t
            sim.advance(duration)
            self.assertAlmostEqual(sim.t - before, duration, places=9)

    def test_stepping_in_pieces_matches_one_run(self):
        pieces, _ = self._sim()
        for _ in range(10):
            pieces.advance(0.03)
        whole = runBurn(_AnalyticBore(_cylinderField()[2]), _flowProperties(), 0.0064, tMax=0.3, pCutoff=0.0)
        self.assertAlmostEqual(pieces.pressure[0] / np.interp(pieces.t, whole.time, whole.headPressure), 1.0, delta=0.01)

    def test_odd_durations_leave_no_sliver_that_collapses_the_pressure(self):
        sim, _ = self._sim()
        sim.advance(0.01)
        start = sim.pressure[0]
        for _ in range(40):
            sim.advance(0.00731)
            self.assertGreater(sim.pressure[0], 0.5 * start)
        self.assertGreater(min(np.diff(sim.result.time)), 1e-5)

    def test_a_cell_length_that_does_not_divide_the_slabs_is_accepted(self):
        sim, voxel = self._sim(cellLength=7.3 * _CASTING / _VOXEL_COUNT)
        self.assertAlmostEqual(sim.flowGrid.length * sim.flowGrid.cells, (_SLICES - 1) * voxel, places=12)
        sim.advance(0.02)
        self.assertGreater(sim.pressure[0], 101325.0)

    def test_the_throat_follows_erosion_and_slag(self):
        sim, _ = self._sim(throatErosionCoeff=1e-9, throatSlagCoeff=1e-3)
        before = sim.throat
        sim.advance(0.05)
        expected = sum((2e-9 * p - 2e-3 / p) * dt for p, dt in zip(sim.result.headPressure, np.diff([0.0] + sim.result.time)))
        self.assertAlmostEqual(sim.throat - before, expected, delta=0.05 * abs(expected))


class TestQuasiSteadyFlow(unittest.TestCase):
    """The quasi-steady bore flow is the steady limit of the transient solver, with the chamber filling at its own pace."""

    @staticmethod
    def _straightBore(solver, span, erosive=True):
        fp = _flowProperties()
        n, dx = _CELLS, _LENGTH / _CELLS
        x = (np.arange(n) + 0.5) * dx
        aPort = np.full(n, np.pi / 4 * _BORE ** 2)
        dHyd, burn = np.full(n, _BORE), np.full(n, np.pi * _BORE)
        aThroat = np.pi / 4 * _THROAT ** 2
        P = np.full(n, 101325.)
        T = np.full(n, fp.tInitial)
        rho = P / (fp.rSpecific * T)
        u = np.zeros(n + 1)
        _seedSteadyState(aPort, dHyd, burn, dx, aThroat, fp, 101325., P, T, rho, u)
        work = [np.zeros(n) for _ in range(10)]
        common = (rho, u, P, T, aPort, dHyd, burn, x, *work[:7], dx, n, aThroat, 101325., fp.tInitial, fp.rhoPropellant,
                  fp.tInitial, fp.tFlame, fp.cpGas, fp.muGas, fp.kGas, fp.prandtl, fp.gamma, fp.rSpecific, fp.roughness,
                  fp.kappa, fp.tabMinP, fp.tabMaxP, fp.tabA, fp.tabN, 1)
        tail = (*work[7:], fp.solidDiffusivity, fp.kappaZn, *fp.surfaceTables())
        if solver == 'transient':
            advanceFlow(*common, erosive, 0.5, 1e-4, 10, 0.0, span, 100_000_000, *tail)
        else:
            advanceQuasiSteady(*common, erosive, 1e-3, 0.25, span, *tail, np.zeros(n + 1))
        return P, u, work[0]

    def testSettledStateMatchesTheTransientSolver(self):
        pT, uT, rT = self._straightBore('transient', 0.3)
        pQ, uQ, rQ = self._straightBore('quasi-steady', 0.3)
        np.testing.assert_allclose(pQ, pT, rtol=2e-4)
        np.testing.assert_allclose(uQ[1:], uT[1:], rtol=2e-4)
        np.testing.assert_allclose(rQ, rT, rtol=2e-4)

    def testPressureFallsAlongTheBoreAndErosionRaisesIt(self):
        pErosive, _, _ = self._straightBore('quasi-steady', 0.3, erosive=True)
        pNormal, _, _ = self._straightBore('quasi-steady', 0.3, erosive=False)
        self.assertTrue(np.all(np.diff(pErosive) < 0.0))
        self.assertGreater(pErosive[-1], 1.015 * pNormal[-1])

    def testChamberFillsTowardTheSteadyPressureFromBelow(self):
        early, _, _ = self._straightBore('quasi-steady', 0.002)
        late, _, _ = self._straightBore('quasi-steady', 0.3)
        self.assertLess(early[-1], late[-1])

    def testCoupledBurnTracksTheTransientSolver(self):
        phi, valid, voxel = _cylinderField()
        kwargs = dict(props=_flowProperties(), throatDiameter=0.0064, tMax=0.3, pCutoff=0.0)
        transient = runBurn(_AnalyticBore(voxel), flow='transient', **kwargs)
        quasi = runBurn(_AnalyticBore(voxel), flow='quasi-steady', **kwargs)
        for t in (0.05, 0.15, 0.3):
            self.assertAlmostEqual(np.interp(t, quasi.time, quasi.headPressure) / np.interp(t, transient.time, transient.headPressure),
                                   1.0, delta=0.02)

    def testQuasiSteadyIsTheDefaultFlowSolver(self):
        phi, valid, voxel = _cylinderField()
        kwargs = dict(props=_flowProperties(), throatDiameter=0.0064, tMax=0.1, pCutoff=0.0)
        default = runBurn(_AnalyticBore(voxel), **kwargs)
        quasi = runBurn(_AnalyticBore(voxel), flow='quasi-steady', **kwargs)
        np.testing.assert_array_equal(default.headPressure, quasi.headPressure)

    def testUnknownFlowSolverIsRejected(self):
        phi, valid, voxel = _cylinderField()
        with self.assertRaises(ValueError):
            runBurn(_AnalyticBore(voxel), _flowProperties(), 0.0064, flow='nope')


class TestLevelSetAdvance(unittest.TestCase):
    """The narrow band advance against the scikit-fmm one it can replace."""

    @classmethod
    def setUpClass(cls):
        slices = 150
        cls.voxel = _CASTING / _VOXEL_COUNT
        n = _VOXEL_COUNT + 2
        centre = (n - 1) / 2
        y, x = np.meshgrid(np.arange(n) - centre, np.arange(n) - centre, indexing='ij')
        radial = np.hypot(x, y) * cls.voxel
        taper = _GRID_BORE * (1.0 + 0.4 * np.linspace(0.0, 1.0, slices))
        cls.phi = radial[None] - taper[:, None, None]
        cls.valid2d = radial <= _CASTING / 2
        cls.slabSpeed = np.linspace(0.004, 0.009, slices - 1)
        cls.dt = 0.6 * cls.voxel / 0.009

    def _grid(self, method):
        return BurnbackGrid(self.phi, self.valid2d, self.voxel, advanceMethod=method)

    def testNarrowBandIsTheDefault(self):
        self.assertEqual(BurnbackGrid(self.phi, self.valid2d, self.voxel).advanceMethod, 'narrow-band')

    def testUnknownMethodIsRejected(self):
        with self.assertRaises(ValueError):
            BurnbackGrid(self.phi, self.valid2d, self.voxel, advanceMethod='nope')

    def testNarrowBandIsAnExactShiftWhenTheStepIsTiny(self):
        grid = self._grid('narrow-band')
        grid.advance(self.slabSpeed, 1e-9)
        sliceSpeed = np.interp(np.arange(self.phi.shape[0]), np.arange(self.phi.shape[0] - 1) + 0.5, self.slabSpeed)
        shifted = np.where(grid.valid, self.phi - sliceSpeed[:, None, None] * 1e-9, self.phi)
        np.testing.assert_allclose(grid.phi, shifted, rtol=0, atol=1e-12)

    def testNarrowBandStaysCloseToScikitFmmOverARealStep(self):
        banded, marched = self._grid('narrow-band'), self._grid('scikit-fmm')
        banded.advance(self.slabSpeed, self.dt)
        marched.advance(self.slabSpeed, self.dt)
        nearSurface = np.abs(marched.phi) < self.voxel
        self.assertLess(float(np.abs(banded.phi - marched.phi)[nearSurface].mean()) / self.voxel, 0.05)
        newLength, oldLength = banded.wettedPerimeter(), marched.wettedPerimeter()
        self.assertLess(float(np.abs(newLength - oldLength).max() / oldLength.max()), 0.03)

    def testNarrowBandResultDoesNotDependOnThreadCount(self):
        from mathlib._levelset import advance_level_set
        speed = np.linspace(0.004, 0.009, self.phi.shape[0])
        one = advance_level_set(self.phi, self.valid2d, speed, self.voxel, self.dt, threads=1)
        many = advance_level_set(self.phi, self.valid2d, speed, self.voxel, self.dt, threads=4)
        np.testing.assert_array_equal(one, many)

    def testMaskedVoxelsAreUntouched(self):
        grid = self._grid('narrow-band')
        grid.advance(self.slabSpeed, self.dt)
        outside = np.broadcast_to(~self.valid2d, self.phi.shape)
        np.testing.assert_array_equal(grid.phi[outside], self.phi[outside])


class TestBurnbackWallRing(unittest.TestCase):
    """With the wall ring the burning surface of a tapered bore runs all the way to the casting wall."""

    @classmethod
    def setUpClass(cls):
        slices, n = 140, 36
        cls.voxel = 0.001
        centre = (n - 1) / 2
        y, x = np.meshgrid(np.arange(n) - centre, np.arange(n) - centre, indexing='ij')
        cls.radial = np.hypot(x, y)
        cls.wall = 17.0
        cls.valid = cls.radial <= cls.wall
        cls.bore = 4.0 + 5.0 * np.linspace(0.0, 1.0, slices)
        cls.propellant = cls.radial[None] > cls.bore[:, None, None]
        cls.slope = 5.0 / (slices - 1)

    def _burnAreaError(self, wallRing, distance):
        grid = BurnbackGrid.fromMask(self.propellant, self.valid, self.voxel, advanceMethod='narrow-band', wallRing=wallRing)
        steps = int(round(distance / (0.5 * self.voxel)))
        for _ in range(steps):
            grid.advance(np.full(grid.nSlabs, 1.0), 0.5 * self.voxel)
        z = np.arange(grid.nSlabs) + 0.5
        bore = np.interp(z, np.arange(len(self.bore)), self.bore) * self.voxel
        cosine = 1.0 / np.sqrt(1.0 + self.slope ** 2)
        surface = bore + steps * 0.5 * self.voxel / cosine
        burning = surface < self.wall * self.voxel
        analytic = float(np.sum(2.0 * np.pi * surface / cosine * self.voxel * burning))
        return grid.slabBurnArea().sum() / analytic - 1.0

    def testRingHasNoEffectEarlyInTheBurn(self):
        self.assertAlmostEqual(self._burnAreaError(True, 2e-3), self._burnAreaError(False, 2e-3), delta=0.005)

    def testRingBringsTheLateBurnAreaCloserToTheClosedForm(self):
        flat, ring = self._burnAreaError(False, 9e-3), self._burnAreaError(True, 9e-3)
        self.assertLess(abs(ring), abs(flat))

    def testNoBurningSurfaceRemainsOnceTheTubeIsBurnedThrough(self):
        grid = BurnbackGrid.fromMask(self.propellant, self.valid, self.voxel, advanceMethod='narrow-band', wallRing=True)
        for _ in range(60):
            grid.advance(np.full(grid.nSlabs, 1.0), 0.5 * self.voxel)
        self.assertEqual(float(grid.slabBurnArea().sum()), 0.0)

    def testRingIsTheDefaultWhenBuildingFromAMask(self):
        self.assertIsNotNone(BurnbackGrid.fromMask(self.propellant, self.valid, self.voxel).wallRing)

    def testRingCanBeLeftOut(self):
        self.assertIsNone(BurnbackGrid.fromMask(self.propellant, self.valid, self.voxel, wallRing=False).wallRing)


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
