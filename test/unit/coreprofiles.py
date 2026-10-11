import unittest

import numpy as np

import motorlib.grains
import motorlib.motor
import motorlib.propellant
from motorlib.coreprofiles import METRICS, availableMetrics, getCoreProfiles
from motorlib.simResult import SimulationResult
from uilib import coreShading

_PROPELLANT = {
    'name': 'Test', 'density': 1700,
    'tabs': [{'minPressure': 0, 'maxPressure': 2e7, 'a': 1.569e-5, 'n': 0.4, 'k': 1.19, 't': 3105, 'm': 24.98}],
}


def _motor():
    motor = motorlib.motor.Motor()
    motor.config.setProperties({
        'maxPressure': 3e7, 'maxMassFlux': 1e4, 'maxMachNumber': 1, 'minPortThroat': 1, 'flowSeparationWarnPercent': 0.05,
        'burnoutWebThres': 2.5e-5, 'burnoutThrustThres': 1.0, 'timestep': 0.05, 'ambPressure': 101325, 'mapDim': 750,
        '3DmapDim': 32, 'sepPressureRatio': 0.4, 'convertTo3D': True,
    })
    motor.nozzle.setProperties({'throat': 0.012, 'exit': 0.03, 'efficiency': 0.95, 'divAngle': 15, 'convAngle': 45,
                                'throatLength': 0.005, 'slagCoeff': 0, 'erosionCoeff': 0})
    motor.propellant = motorlib.propellant.Propellant(_PROPELLANT)
    for _ in range(2):
        grain = motorlib.grains.grainTypes['BATES']()
        grain.setProperties({'diameter': 0.05, 'length': 0.12, 'coreDiameter': 0.02, 'inhibitedEnds': 'Both'})
        motor.grains.append(grain)
    return motor


class TestCoreProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _motor().runSimulation()
        cls.step = len(cls.result.channels['time'].getData()) // 3

    def test_every_3d_grain_gets_a_profile_with_a_value_per_slice(self):
        profiles = getCoreProfiles(self.result, self.step)
        self.assertEqual(sorted(profiles), [0, 1])
        for gid, profile in profiles.items():
            zdim = self.result.motor.grains[gid].regressionMap.shape[0]
            self.assertEqual(len(profile.regression), zdim)
            self.assertEqual(set(profile.metrics), set(METRICS))
            for values in profile.metrics.values():
                self.assertEqual(len(values), zdim)

    def test_kn_reaches_the_motors_kn_at_the_nozzle(self):
        last = getCoreProfiles(self.result, self.step)[1]
        nozzleEnd = last.metrics['kn'][np.isfinite(last.metrics['kn'])][0]
        self.assertAlmostEqual(nozzleEnd / self.result.channels['kn'].getPoint(self.step), 1.0, delta=0.03)

    def test_the_peak_mass_flux_matches_what_the_simulation_reports(self):
        profiles = getCoreProfiles(self.result, self.step)
        reported = self.result.channels['massFlux'].getPoint(self.step)
        for gid, profile in profiles.items():
            peak = np.nanmax(profile.metrics['massFlux'])
            self.assertAlmostEqual(peak / reported[gid], 1.0, delta=0.1, msg='grain {}'.format(gid))

    def test_flux_builds_up_toward_the_nozzle_and_carries_over_between_grains(self):
        profiles = getCoreProfiles(self.result, self.step)
        for profile in profiles.values():
            flux = profile.metrics['massFlux']
            valid = flux[np.isfinite(flux)]
            self.assertGreater(valid[0], valid[len(valid) // 2])
            self.assertGreater(valid[len(valid) // 2], valid[-1])
        forward = profiles[0].metrics['massFlux']
        aft = profiles[1].metrics['massFlux']
        leaving, entering = forward[np.isfinite(forward)][0], aft[np.isfinite(aft)][-1]
        self.assertAlmostEqual(entering / leaving, 1.0, delta=0.1)

    def test_pressure_and_burn_rate_do_not_vary_along_the_core_of_the_steady_model(self):
        profile = getCoreProfiles(self.result, self.step)[0]
        self.assertEqual(len(set(profile.metrics['pressure'])), 1)
        self.assertAlmostEqual(profile.metrics['pressure'][0], self.result.channels['pressure'].getPoint(self.step))
        self.assertEqual(len(set(profile.metrics['burnRate'])), 1)

    def test_the_core_is_drawn_at_the_grains_regression(self):
        profile = getCoreProfiles(self.result, self.step)[0]
        self.assertTrue(np.all(profile.regression == self.result.channels['regression'].getPoint(self.step)[0]))

    def test_mach_number_follows_the_mass_flux(self):
        metrics = getCoreProfiles(self.result, self.step)[1].metrics
        ok = np.isfinite(metrics['massFlux'])
        order = np.argsort(metrics['massFlux'][ok])
        self.assertTrue(np.all(np.diff(metrics['machNumber'][ok][order]) >= -1e-9))

    @staticmethod
    def _preview(regressions):
        """A result holding only a time and a regression, which is all the viewer's preview has."""
        motor = _motor().getWithGrainsConvertedTo3D()
        for grain in motor.grains:
            grain.simulationSetup(motor.config)
        preview = SimulationResult(motor)
        preview.channels['time'].addData(0.0)
        preview.channels['regression'].addData(regressions(motor))
        return preview

    def test_a_preview_result_with_only_zero_regression_still_has_profiles(self):
        profile = getCoreProfiles(self._preview(lambda motor: [0.0, 0.0]), 0)[0]
        self.assertGreater(np.nanmax(profile.metrics['massFlux']), 0)
        self.assertGreater(profile.metrics['pressure'][0], 1e5)

    def test_a_burned_out_grain_has_no_profile_and_the_grain_behind_it_gets_no_flow_from_it(self):
        preview = self._preview(lambda motor: [motor.grains[0].wallWeb * 1.01, 0.0])
        profiles = getCoreProfiles(preview, 0)
        self.assertEqual(sorted(profiles), [1])
        self.assertGreater(np.nanmax(profiles[1].metrics['massFlux']), 0)

    def test_a_step_that_was_not_recorded_gives_nothing(self):
        self.assertEqual(getCoreProfiles(self.result, len(self.result.channels['time'].getData()) + 5), {})

    def test_the_steady_model_offers_every_metric(self):
        self.assertEqual([m.key for m in availableMetrics(self.result)], list(METRICS))


class TestCoreShading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _motor().runSimulation()
        cls.shading = coreShading.CoreShading(cls.result)
        cls.steps = cls.shading.steps

    def test_the_range_covers_every_recorded_step(self):
        low, high = self.shading.range('massFlux')
        for step in range(0, self.steps, 4):
            for profile in self.shading.profiles(step).values():
                values = profile.metrics['massFlux']
                values = values[np.isfinite(values)]
                self.assertGreaterEqual(values.min(), low * (1 - 1e-9))
                self.assertLessEqual(values.max(), high * (1 + 1e-6))

    def test_the_ends_of_the_range_are_the_ends_of_the_colour_table(self):
        low, high = self.shading.range('massFlux')
        np.testing.assert_array_equal(coreShading.sliceColors(np.array([low]), low, high)[0], coreShading.CORE_LUT[0])
        np.testing.assert_array_equal(coreShading.sliceColors(np.array([high]), low, high)[0], coreShading.CORE_LUT[-1])

    def test_a_grain_has_a_colour_per_slice_while_it_burns(self):
        colors = self.shading.colors('massFlux', 1, 0)
        self.assertEqual(colors.shape, (len(self.shading.profile(1, 0).regression), 4))

    def test_a_step_that_was_not_recorded_has_no_colours(self):
        self.assertIsNone(self.shading.colors('massFlux', self.steps + 3, 0))

    def test_slices_without_a_value_are_grey(self):
        colors = coreShading.sliceColors(np.array([np.nan, 1.0]), 0.0, 2.0)
        self.assertEqual(tuple(colors[0]), coreShading.NO_VALUE)
        self.assertNotEqual(tuple(colors[1]), coreShading.NO_VALUE)

    def test_a_constant_metric_is_one_colour(self):
        colors = coreShading.sliceColors(np.full(5, 3.0), 3.0, 3.0)
        self.assertEqual(len({tuple(c) for c in colors}), 1)

    def test_images_put_the_head_end_in_the_top_row_and_the_axial_plane_in_one_colour(self):
        colors = coreShading.sliceColors(np.arange(4.0), 0.0, 3.0)
        side = coreShading.sideImage(colors, 5)
        self.assertEqual(side.shape, (4, 5, 4))
        np.testing.assert_array_equal(side[0, 0], colors[3])
        np.testing.assert_array_equal(side[3, 4], colors[0])
        axial = coreShading.axialImage(colors, 2, (6, 5))
        self.assertEqual(axial.shape, (6, 5, 4))
        self.assertEqual(len({tuple(p) for p in axial.reshape(-1, 4)}), 1)
        np.testing.assert_array_equal(axial[0, 0], colors[2])


try:
    from uilib.widgets.sliceViewWidget import _renderBinary
except ImportError:
    _renderBinary = None


@unittest.skipIf(_renderBinary is None, 'the Qt widgets are not installed')
class TestCoreRendering(unittest.TestCase):
    def test_core_colours_replace_the_core_and_leave_the_rest(self):
        regression = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        colors = np.zeros((2, 3, 4), dtype=np.uint8)
        colors[...] = (10, 20, 30, 255)
        plain = _renderBinary(regression, 0.6)
        shaded = _renderBinary(regression, 0.6, None, colors)
        core = regression <= 0.6
        self.assertTrue(np.all(shaded[core] == (10, 20, 30, 255)))
        np.testing.assert_array_equal(shaded[~core], plain[~core])

    def test_each_row_can_be_drawn_at_its_own_regression(self):
        regression = np.tile(np.array([0.0, 0.5, 1.0]), (2, 1))
        colors = np.full((2, 3, 4), 200, dtype=np.uint8)
        shaded = _renderBinary(regression, np.array([[0.1], [0.7]]), None, colors)
        self.assertEqual(int(np.all(shaded[0] == 200, axis=1).sum()), 1)
        self.assertEqual(int(np.all(shaded[1] == 200, axis=1).sum()), 2)

    def test_outside_the_boundary_stays_transparent(self):
        regression = np.zeros((2, 2))
        mask = np.array([[True, False], [False, False]])
        shaded = _renderBinary(regression, 1.0, mask, np.full((2, 2, 4), 200, dtype=np.uint8))
        self.assertEqual(tuple(shaded[0, 0]), (0, 0, 0, 0))
