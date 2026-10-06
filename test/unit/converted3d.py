import unittest
from unittest import mock

import numpy as np

import motorlib.grains
import motorlib.motor
import motorlib.propellant
from motorlib.grain import ConvertedGrain, Fmm3DGrain, getCrossSectionCoordinates

_PROPELLANT = {
    'name': 'Test', 'density': 1700,
    'tabs': [{'minPressure': 0, 'maxPressure': 1e7, 'a': 3.5e-5, 'n': 0.35, 'k': 1.2, 't': 2800, 'm': 25}],
}
_MAP_DIM = 96
_DIAMETER, _LENGTH = 0.08, 0.12


def _config(convert=False):
    config = motorlib.motor.MotorConfig()
    config.setProperties({
        'maxPressure': 1e7, 'maxMassFlux': 1e4, 'maxMachNumber': 1, 'minPortThroat': 1, 'flowSeparationWarnPercent': 0.05,
        'burnoutWebThres': 2.5e-5, 'burnoutThrustThres': 0.1, 'timestep': 0.03, 'ambPressure': 101325, 'mapDim': 750,
        '3DmapDim': _MAP_DIM, 'sepPressureRatio': 0.4, 'convertTo3D': convert,
    })
    return config


def _grain(geomName, **props):
    grain = motorlib.grains.grainTypes[geomName]()
    grain.setProperties({'diameter': _DIAMETER, 'length': _LENGTH, 'inhibitedEnds': 'Both', **props})
    return grain


def _motor(grains, convert):
    motor = motorlib.motor.Motor()
    motor.config = _config(convert)
    motor.nozzle.setProperties({'throat': 0.012, 'exit': 0.03, 'efficiency': 0.95, 'divAngle': 15, 'convAngle': 45,
                                'throatLength': 0.005, 'slagCoeff': 0, 'erosionCoeff': 0})
    motor.propellant = motorlib.propellant.Propellant(_PROPELLANT)
    motor.grains = grains
    return motor


class TestConvertTo3D(unittest.TestCase):
    def _converted(self, grain):
        converted = grain.convertTo3D()
        converted.simulationSetup(_config())
        grain.simulationSetup(_config())
        return converted

    def test_bates_geometry_matches_the_analytic_grain(self):
        grain = _grain('BATES', coreDiameter=0.03)
        converted = self._converted(grain)
        self.assertIsInstance(converted, ConvertedGrain)
        for reg in (0.004, 0.008):
            self.assertAlmostEqual(converted.getSurfaceAreaAtRegression(reg) / grain.getSurfaceAreaAtRegression(reg), 1, delta=0.04)
            self.assertAlmostEqual(converted.getVolumeAtRegression(reg) / grain.getVolumeAtRegression(reg), 1, delta=0.05)
            self.assertAlmostEqual(converted.getPortArea(reg) / grain.getPortArea(reg), 1, delta=0.04)
        self.assertAlmostEqual(converted.wallWeb / grain.wallWeb, 1, delta=0.03)

    def test_conical_core_is_narrower_at_the_aft_end(self):
        grain = _grain('Conical', forwardCoreDiameter=0.045, aftCoreDiameter=0.02)
        converted = self._converted(grain)
        aftArea = 3.14159265 * 0.02 ** 2 / 4
        self.assertAlmostEqual(converted.getPortArea(0) / aftArea, 1, delta=0.04)
        for reg in (0.004, 0.008):
            self.assertAlmostEqual(converted.getSurfaceAreaAtRegression(reg) / grain.getSurfaceAreaAtRegression(reg), 1, delta=0.05)

    def test_two_dimensional_fmm_grain_keeps_its_cross_section(self):
        grain = _grain('Finocyl', coreDiameter=0.02, numFins=4, finWidth=0.004, finLength=0.015, invertedFins=False)
        converted = self._converted(grain)
        for reg in (0.003, 0.006):
            self.assertAlmostEqual(converted.getSurfaceAreaAtRegression(reg) / grain.getSurfaceAreaAtRegression(reg), 1, delta=0.06)

    def test_conversion_does_not_modify_the_source_grain(self):
        grain = _grain('BATES', coreDiameter=0.03)
        before = grain.getProperties()
        converted = grain.convertTo3D()
        converted.simulationSetup(_config())
        self.assertEqual(grain.getProperties(), before)
        self.assertEqual(converted.geomName, 'BATES')
        self.assertEqual(converted.getDetailsString(), grain.getDetailsString())

    def test_grains_without_a_conversion_are_returned_unchanged(self):
        rodTube = _grain('Rod and Tube', coreDiameter=0.05, rodDiameter=0.03, supportDiameter=0.01)
        for grain in (_grain('End Burner'), rodTube):  # The rod's free ends need inhibitedEnds to be 'Neither'
            self.assertIs(grain.convertTo3D(), grain)
        converted = _grain('BATES', coreDiameter=0.03).convertTo3D()
        self.assertIs(converted.convertTo3D(), converted)
        self.assertIsInstance(converted, Fmm3DGrain)


class TestMotorConversion(unittest.TestCase):
    def test_flag_defaults_to_off_and_old_files_without_it_load(self):
        self.assertFalse(motorlib.motor.MotorConfig().getProperty('convertTo3D'))
        motor = motorlib.motor.Motor()
        motor.applyDict({'nozzle': {}, 'propellant': None, 'grains': [], 'config': {'timestep': 0.05}})
        self.assertFalse(motor.config.getProperty('convertTo3D'))

    def test_simulation_with_the_flag_uses_3d_grains_and_leaves_the_motor_alone(self):
        grains = [_grain('BATES', coreDiameter=0.03)]
        motor = _motor(grains, True)
        result = motor.runSimulation()
        self.assertTrue(result.success)
        self.assertIsInstance(result.motor.grains[0], ConvertedGrain)
        self.assertIs(motor.grains[0], grains[0])
        self.assertTrue(motor.config.getProperty('convertTo3D'))

    def test_converted_motor_matches_the_analytic_simulation(self):
        grains = [_grain('BATES', coreDiameter=0.03, inhibitedEnds='Neither'), _grain('BATES', coreDiameter=0.025)]
        analytic = _motor(grains, False).runSimulation()
        converted = _motor(grains, True).runSimulation()
        self.assertAlmostEqual(converted.getBurnTime() / analytic.getBurnTime(), 1, delta=0.04)
        self.assertAlmostEqual(converted.getImpulse() / analytic.getImpulse(), 1, delta=0.05)

    def test_grains_that_cannot_convert_are_simulated_normally_with_a_warning(self):
        grains = [_grain('End Burner', length=0.05), _grain('BATES', coreDiameter=0.03)]
        result = _motor(grains, True).runSimulation()
        self.assertTrue(result.success)
        self.assertNotIsInstance(result.motor.grains[0], Fmm3DGrain)
        self.assertIsInstance(result.motor.grains[1], ConvertedGrain)
        warnings = [a for a in result.alerts if 'no 3D conversion' in a.description]
        self.assertEqual([a.location for a in warnings], ['Grain 1'])


class TestMultiWebGrains(unittest.TestCase):
    def test_rod_and_tube_with_free_ends_keeps_its_masked_support(self):
        props = dict(coreDiameter=0.05, rodDiameter=0.03, supportDiameter=0.01, inhibitedEnds='Neither')
        analytic = _motor([_grain('Rod and Tube', **props)], False).runSimulation()
        motor = _motor([_grain('Rod and Tube', **props)], True)
        converted = motor.runSimulation()
        self.assertIsInstance(converted.motor.grains[0], ConvertedGrain)
        self.assertAlmostEqual(converted.getPeakKN() / analytic.getPeakKN(), 1, delta=0.05)
        self.assertAlmostEqual(converted.getImpulse() / analytic.getImpulse(), 1, delta=0.05)
        grain = converted.motor.grains[0]
        self.assertTrue(grain._maskU8[:, 48, 48].all())  # The hollow support at the centre is not propellant

    def test_rod_and_tube_with_inhibited_ends_stays_two_dimensional_with_a_note(self):
        props = dict(coreDiameter=0.05, rodDiameter=0.03, supportDiameter=0.01, inhibitedEnds='Both')
        result = _motor([_grain('Rod and Tube', **props)], True).runSimulation()
        self.assertNotIsInstance(result.motor.grains[0], Fmm3DGrain)
        self.assertTrue(any('no 3D conversion' in a.description for a in result.alerts))


class TestConversionSafetyNet(unittest.TestCase):
    def _notes(self, motor):
        return [a.description for a in motor.runSimulation().alerts if 'simulated normally' in a.description]

    def test_a_conversion_that_disagrees_with_the_original_is_rejected(self):
        grains = [_grain('BATES', coreDiameter=0.03)]
        def wrongCore(self, mapLength, mapDim):  # A bore that is too wide, so the grain is much thinner than the original
            coords = np.linspace(-1, 1, mapDim)
            face = coords[:, None] ** 2 + coords[None, :] ** 2 >= (0.05 / _DIAMETER) ** 2
            return np.broadcast_to(face, (mapLength, mapDim, mapDim)).copy()
        with mock.patch.object(motorlib.grains.BatesGrain, 'getExtrudedCoreMap', wrongCore):
            motor = _motor(grains, True)
            notes = self._notes(motor)
            result = motor.runSimulation()
        self.assertEqual(len(notes), 1)
        self.assertIn('away from the original', notes[0])
        self.assertTrue(result.success)
        self.assertIsInstance(result.motor.grains[0], motorlib.grains.BatesGrain)

    def test_a_core_map_with_no_void_falls_back_instead_of_raising(self):
        solid = lambda self, mapLength, mapDim: np.ones((mapLength, mapDim, mapDim), dtype=bool)
        with mock.patch.object(motorlib.grains.BatesGrain, 'getExtrudedCoreMap', solid):
            result = _motor([_grain('BATES', coreDiameter=0.03)], True).runSimulation()
        self.assertTrue(result.success)
        self.assertTrue(any('3D setup failed' in a.description for a in result.alerts))

    def test_a_failure_while_setting_up_the_3d_grain_falls_back_to_the_original(self):
        def fail(self, mapLength, mapDim):
            raise MemoryError('out of memory')
        with mock.patch.object(motorlib.grains.BatesGrain, 'getExtrudedCoreMap', fail):
            motor = _motor([_grain('BATES', coreDiameter=0.03)], True)
            result = motor.runSimulation()
        self.assertTrue(result.success)
        self.assertTrue(any('3D setup failed' in a.description and 'out of memory' in a.description for a in result.alerts))

    def test_a_grain_too_large_for_the_map_dimension_is_not_converted(self):
        motor = _motor([_grain('BATES', coreDiameter=0.03, length=10.0)], True)
        motor.config.setProperty('3DmapDim', 128)
        converted = motor.getWithGrainsConvertedTo3D()
        self.assertNotIsInstance(converted.grains[0], Fmm3DGrain)
        self.assertIn('million voxels', converted.conversionNotes[0][1])

    def test_grains_are_not_set_up_twice_during_a_simulation(self):
        grain = _grain('BATES', coreDiameter=0.03).convertTo3D()
        with mock.patch.object(ConvertedGrain, 'generateCoreMap', autospec=True, side_effect=ConvertedGrain.generateCoreMap) as spy:
            grain.simulationSetup(_config())
            grain.simulationSetup(_config())
        self.assertEqual(spy.call_count, 1)


class TestMassFluxDirection(unittest.TestCase):
    """Gas flows toward the aft end at index 0, so the mass flow through a slice is the mass burned at or ahead of it."""

    _DENSITY, _TIME, _RATE = 1700, 0.03, 0.005

    def _flux(self, grain, reg):
        return grain.getPeakMassFlux(0, self._TIME, reg, self._RATE * self._TIME, self._DENSITY)

    def _grains(self, forward, aft):
        grain = _grain('Conical', forwardCoreDiameter=forward, aftCoreDiameter=aft)
        converted = grain.convertTo3D()
        converted.simulationSetup(_config())
        grain.simulationSetup(_config())
        return grain, converted

    def test_peak_flux_matches_the_analytic_conical_grain_in_both_orientations(self):
        for forward, aft in ((0.045, 0.02), (0.02, 0.045)):
            grain, converted = self._grains(forward, aft)
            for reg in (0.004, 0.008):
                self.assertAlmostEqual(self._flux(converted, reg) / self._flux(grain, reg), 1, delta=0.06)

    def test_a_narrow_aft_port_carries_much_more_flux_than_a_wide_one(self):
        narrowAft = self._flux(self._grains(0.045, 0.02)[1], 0.004)
        wideAft = self._flux(self._grains(0.02, 0.045)[1], 0.004)
        self.assertGreater(narrowAft, 2.5 * wideAft)

    def test_share_of_burning_area_ahead_of_a_slice_falls_toward_the_fore_end(self):
        converted = self._grains(0.045, 0.02)[1]
        ahead = converted._getAreaFractionFore(converted.normalize(0.004))
        self.assertEqual(len(ahead), converted.regressionMap.shape[0])
        self.assertAlmostEqual(ahead[0], 1.0, delta=0.02)
        self.assertLess(ahead[-1], 0.05)
        self.assertTrue((ahead[1:] <= ahead[:-1] + 1e-9).all())

    def test_the_volume_fraction_fallback_points_the_same_way(self):
        grain, converted = self._grains(0.045, 0.02)
        converted._layerAreas = None
        self.assertAlmostEqual(self._flux(converted, 0.004) / self._flux(grain, 0.004), 1, delta=0.15)


class TestMapScale(unittest.TestCase):
    """Pixels are 1/mapDim of the diameter wide, so cell centered coordinates make the 3D map exact in size."""

    def test_cell_centered_coordinates_are_symmetric_with_pixel_spacing(self):
        coords = getCrossSectionCoordinates(96, cellCentered=True)
        self.assertAlmostEqual(coords[1] - coords[0], 2 / 96)
        self.assertAlmostEqual(coords[0], -coords[-1])
        self.assertAlmostEqual(coords[0], -1 + 1 / 96)

    def test_the_default_coordinates_keep_the_two_dimensional_layout(self):
        coords = getCrossSectionCoordinates(96)
        self.assertEqual((coords[0], coords[-1]), (-1.0, 1.0))

    def test_the_casting_tube_has_its_exact_area_at_any_resolution(self):
        for mapDim in (48, 64, 96, 128):
            mask = _grain('BATES', coreDiameter=0.03).getExtrudedMask(mapDim)
            area = np.count_nonzero(~mask) * (_DIAMETER / mapDim) ** 2
            self.assertAlmostEqual(area / (np.pi * (_DIAMETER / 2) ** 2), 1, delta=0.01, msg=str(mapDim))

    def test_converted_volume_and_web_match_the_analytic_grain(self):
        for mapDim in (64, 96):
            config = _config()
            config.setProperty('3DmapDim', mapDim)
            grain = _grain('BATES', coreDiameter=0.03)
            converted = grain.convertTo3D()
            grain.simulationSetup(config)
            converted.simulationSetup(config)
            for reg in (0, 0.004, 0.008):
                self.assertAlmostEqual(converted.getVolumeAtRegression(reg) / grain.getVolumeAtRegression(reg), 1, delta=0.012)
            self.assertAlmostEqual(converted.wallWeb / grain.wallWeb, 1, delta=0.012)


class TestRegressionCacheKeys(unittest.TestCase):
    def test_the_distance_cache_depends_on_the_mask(self):
        core = np.ones((4, 8, 8), dtype=bool)
        wide = np.broadcast_to(np.zeros((8, 8), dtype=bool)[None], core.shape)
        narrow = np.broadcast_to(np.pad(np.zeros((6, 6), dtype=bool), 1, constant_values=True)[None], core.shape)
        key = Fmm3DGrain._computeFmmCacheKey
        self.assertEqual(key(core, 0.01, wide), key(core, 0.01, wide))
        self.assertNotEqual(key(core, 0.01, wide), key(core, 0.01, narrow))

    def test_a_grain_whose_mask_changes_recomputes_its_regression_map(self):
        grain = _grain('BATES', coreDiameter=0.03).convertTo3D()
        grain.simulationSetup(_config())
        before = grain.regressionMap.copy()
        grain.mask = np.broadcast_to(grain.mask[0] | (np.arange(96)[:, None] < 10), grain.mask.shape)
        grain.generateRegressionMap()
        self.assertFalse(np.array_equal(before, grain.regressionMap))


class TestAftPortArea(unittest.TestCase):
    def test_a_solid_end_cap_is_skipped(self):
        self.assertEqual(Fmm3DGrain._getAftPortCount([0, 0, 40, 100, 100]), 100)
        self.assertEqual(Fmm3DGrain._getAftPortCount([5, 90, 100, 100]), 90)

    def test_a_tapering_core_keeps_its_true_aft_area(self):
        self.assertEqual(Fmm3DGrain._getAftPortCount([100, 104, 108, 112]), 100)
        self.assertEqual(Fmm3DGrain._getAftPortCount([]), 0)


class TestAreaCurve(unittest.TestCase):
    """Burning area comes from marching cubes areas on a signed distance field; volume is its integral."""

    @staticmethod
    def _setUp(grain, mapDim):
        config = _config()
        config.setProperty('3DmapDim', mapDim)
        reference = _grain(*grain[0], **grain[1])
        converted = reference.convertTo3D()
        reference.simulationSetup(config)
        converted.simulationSetup(config)
        return reference, converted

    def test_the_regression_map_is_zero_at_the_surface_and_negative_in_the_core(self):
        _, converted = self._setUp((('BATES',), dict(coreDiameter=0.03)), 96)
        valid = ~converted.mask
        core = ~converted.coreMap & valid
        self.assertLess(converted.regressionMap[core].max(), 0)
        self.assertGreater(converted.regressionMap[converted.coreMap & valid].min(), 0)

    def test_area_at_zero_regression_matches_the_analytic_grain(self):
        for grain, tolerance in (((('BATES',), dict(coreDiameter=0.03)), 0.05),
                                 ((('Conical',), dict(forwardCoreDiameter=0.045, aftCoreDiameter=0.02)), 0.07)):
            reference, converted = self._setUp(grain, 96)
            for reg in (0, 0.004, 0.008):
                self.assertAlmostEqual(converted.getSurfaceAreaAtRegression(reg) / reference.getSurfaceAreaAtRegression(reg),
                                       1, delta=tolerance, msg=str(reg))

    def test_volume_is_the_integral_of_the_area(self):
        _, converted = self._setUp((('Finocyl',), dict(coreDiameter=0.02, numFins=4, finWidth=0.004, finLength=0.015,
                                                         invertedFins=False)), 96)
        for reg in (0.002, 0.006, 0.010):
            step = 1e-4
            consumed = converted.getVolumeAtRegression(reg) - converted.getVolumeAtRegression(reg + step)
            self.assertAlmostEqual(consumed / (converted.getSurfaceAreaAtRegression(reg + step / 2) * step), 1, delta=0.02)

    def test_the_wall_sliver_the_surfaces_miss_stays_small_at_any_map_size(self):
        for mapDim in (64, 96, 128):
            shortfall = self._setUp((('BATES',), dict(coreDiameter=0.03)), mapDim)[1]._areaShortfall
            self.assertGreater(shortfall, 0, mapDim)
            self.assertLess(shortfall, 0.06, mapDim)

    def test_marching_cubes_areas_that_disagree_with_the_volume_are_not_trusted(self):
        _, converted = self._setUp((('BATES',), dict(coreDiameter=0.03)), 64)
        levels = np.linspace(0, 1.0, 20)
        self.assertIsNone(converted._getCurvesFromAreas(levels, np.full(20, 1e-3), 1e-3, 1.0))
        self.assertIsNone(converted._getCurvesFromAreas(levels[:3], np.ones(3), 1e-3, 1.0))
