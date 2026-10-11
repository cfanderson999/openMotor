"""
Unit tests for mathlib._edt – the parallel exact Euclidean distance transform – and its use in Fmm3DGrain.
"""

import os
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage

import motorlib.grain
import motorlib.grains
import motorlib.motor
from mathlib import _edt


class TestSquaredDistance(unittest.TestCase):
    def test_matches_scipy_on_random_sources(self):
        rng = np.random.default_rng(1)
        for shape in ((37, 41, 29), (5, 64, 64), (1, 20, 20), (64, 8, 3)):
            sources = rng.random(shape) < 0.02
            sources[0, 0, 0] = True
            expected = ndimage.distance_transform_edt(~sources) ** 2
            np.testing.assert_allclose(_edt.squared_distance(sources), expected, atol=1e-9)

    def test_no_sources_is_unbounded(self):
        self.assertTrue(np.all(_edt.squared_distance(np.zeros((4, 5, 6), dtype=bool)) > 1e19))


class TestSignedDistance(unittest.TestCase):
    def setUp(self):
        _, y, x = np.indices((40, 64, 64))
        self.radius = np.hypot(y - 31.5, x - 31.5)
        self.propellant = (self.radius > 12) & (self.radius < 30)
        self.valid = self.radius < 31

    def test_surface_sits_half_a_voxel_from_each_side(self):
        field = _edt.signed_distance(self.propellant, self.valid, 2.0)
        inside = self.valid
        self.assertAlmostEqual(float(field[self.propellant & inside].min()), 0.5)
        self.assertAlmostEqual(float(field[~self.propellant & inside].max()), -0.5)

    def test_masked_voxels_take_the_fill_value(self):
        field = _edt.signed_distance(self.propellant, self.valid, 2.0)
        self.assertTrue(np.all(field[~self.valid] == 2.0))

    def test_masked_voxels_are_not_sources(self):
        field = _edt.signed_distance(self.propellant | ~self.valid, self.valid, 2.0)
        expected = ndimage.distance_transform_edt(self.propellant | ~self.valid) - 0.5
        keep = self.propellant & self.valid
        np.testing.assert_allclose(field[keep], expected[keep], atol=1e-9)

    def test_bore_distance_tracks_radius(self):
        field = _edt.signed_distance(self.propellant, self.valid, 2.0)
        keep = self.propellant & (self.radius < 20)
        error = field[keep] - (self.radius[keep] - 12)
        self.assertLess(float(np.abs(error).max()), 0.75)


class TestFmm3DGrainDistanceMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = motorlib.motor.MotorConfig()
        cls.config.setProperties({'maxPressure': 7e7, 'maxMassFlux': 1e4, 'maxMachNumber': 1, 'minPortThroat': 1,
                                  'flowSeparationWarnPercent': 0.05, 'burnoutWebThres': 2.54e-5,
                                  'burnoutThrustThres': 0.1, 'timestep': 0.03, 'ambPressure': 101325, 'mapDim': 750,
                                  '3DmapDim': 48, 'sepPressureRatio': 0.4, 'convertTo3D': False})

    def _setUp(self, useEdt):
        grain = motorlib.grains.grainTypes['BATES']()
        grain.setProperties({'diameter': 0.08, 'length': 0.1, 'coreDiameter': 0.03, 'inhibitedEnds': 'Both'})
        converted = grain.convertTo3D()
        env = {'OPENMOTOR_EXPERIMENTAL_EDT_DISTANCE': '1' if useEdt else '0'}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(motorlib.grain.Fmm3DGrain, '_loadFmmCache', lambda self, key: None), \
                mock.patch.object(motorlib.grain.Fmm3DGrain, '_saveFmmCache', lambda self, key, data: None):
            converted.simulationSetup(self.config)
        return converted

    def test_matches_fast_marching_within_a_third_of_a_voxel(self):
        marched = self._setUp(False)
        exact = self._setUp(True)
        valid = ~np.asarray(marched.mask, dtype=bool)
        voxelsPerUnit = marched.mapDim / 2.0
        difference = np.abs(marched.regressionMap - exact.regressionMap)[valid] * voxelsPerUnit
        self.assertLess(float(difference.max()), 0.35)

    def test_burning_area_and_web_agree(self):
        marched = self._setUp(False)
        exact = self._setUp(True)
        self.assertAlmostEqual(exact.wallWeb / marched.wallWeb, 1.0, delta=0.01)
        for fraction in (0.1, 0.4, 0.7):
            distance = fraction * marched.wallWeb
            self.assertAlmostEqual(
                exact.getSurfaceAreaAtRegression(distance) / marched.getSurfaceAreaAtRegression(distance), 1.0, delta=0.03
            )


if __name__ == '__main__':
    unittest.main()
