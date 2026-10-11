"""
Unit tests for mathlib._march – input validation and marching-cubes wrapper.
"""

import unittest
import numpy as np

from mathlib._march import _validate_inputs, marching_area_pixels


class TestValidateInputs(unittest.TestCase):
    """Tests for _validate_inputs checks."""

    def _make_grid(self, shape=(10, 10, 10)):
        grid = np.random.rand(*shape)
        mask = np.zeros(shape, dtype=bool)
        return grid, mask

    def test_valid_inputs_pass(self):
        grid, mask = self._make_grid()
        g, m, l = _validate_inputs(grid, mask, 0.5)
        self.assertEqual(g.dtype, np.float64)
        self.assertEqual(m.dtype, np.bool_)
        self.assertAlmostEqual(l, 0.5)

    def test_2d_grid_raises(self):
        with self.assertRaises(ValueError):
            _validate_inputs(np.zeros((5, 5)), np.zeros((5, 5)), 0.5)

    def test_2d_mask_raises(self):
        with self.assertRaises(ValueError):
            _validate_inputs(np.zeros((5, 5, 5)), np.zeros((5, 5)), 0.5)

    def test_shape_mismatch_raises(self):
        grid = np.zeros((5, 5, 5))
        mask = np.zeros((5, 5, 6))
        with self.assertRaises(ValueError):
            _validate_inputs(grid, mask, 0.5)

    def test_nan_level_raises(self):
        grid, mask = self._make_grid()
        with self.assertRaises(ValueError):
            _validate_inputs(grid, mask, float('nan'))

    def test_inf_level_raises(self):
        grid, mask = self._make_grid()
        with self.assertRaises(ValueError):
            _validate_inputs(grid, mask, float('inf'))

    def test_returns_contiguous_arrays(self):
        grid = np.zeros((5, 5, 5), order='F')
        mask = np.zeros((5, 5, 5), dtype=bool, order='F')
        g, m, _ = _validate_inputs(grid, mask, 0.5)
        self.assertTrue(g.flags['C_CONTIGUOUS'])
        self.assertTrue(m.flags['C_CONTIGUOUS'])


class TestMarchingAreaPixels(unittest.TestCase):
    """Tests for marching_area_pixels with synthetic scalar fields."""

    def test_sphere_has_positive_area(self):
        """A sphere distance field should produce a measurable iso-surface."""
        n = 30
        x = np.linspace(-1, 1, n)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        grid = np.sqrt(X**2 + Y**2 + Z**2)
        mask = np.ones_like(grid, dtype=bool)
        area = marching_area_pixels(grid, mask, level=0.5)
        self.assertIsNotNone(area)
        self.assertGreater(area, 0.0)

    def test_no_surface_returns_none(self):
        """Level above all values → no surface → None."""
        grid = np.zeros((10, 10, 10))
        mask = np.ones_like(grid, dtype=bool)
        area = marching_area_pixels(grid, mask, level=1.0)
        self.assertIsNone(area)

    def test_step_size_reduces_cost(self):
        """Higher step_size should still produce valid (though less precise) area."""
        n = 30
        x = np.linspace(-1, 1, n)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        grid = np.sqrt(X**2 + Y**2 + Z**2)
        mask = np.ones_like(grid, dtype=bool)
        area1 = marching_area_pixels(grid, mask, level=0.5, step_size=1)
        area2 = marching_area_pixels(grid, mask, level=0.5, step_size=2)
        self.assertIsNotNone(area1)
        self.assertIsNotNone(area2)
        # Both should be positive; step_size=2 may differ but should be same order
        self.assertGreater(area2, 0.0)

    def test_mask_excludes_region(self):
        """Masking part of the volume should change the measured area."""
        n = 30
        x = np.linspace(-1, 1, n)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        grid = np.sqrt(X**2 + Y**2 + Z**2)
        full_mask = np.ones_like(grid, dtype=bool)
        half_mask = np.ones_like(grid, dtype=bool)
        half_mask[n // 2:, :, :] = False  # exclude half the volume
        area_full = marching_area_pixels(grid, full_mask, level=0.5)
        area_half = marching_area_pixels(grid, half_mask, level=0.5)
        self.assertIsNotNone(area_full)
        self.assertIsNotNone(area_half)
        self.assertLess(area_half, area_full)


if __name__ == '__main__':
    unittest.main()


class TestContourLengthBySlice(unittest.TestCase):
    """The marching squares kernel must reproduce skimage's contour lengths, saddles included."""

    @staticmethod
    def _skimageLength(plane, level=0.0):
        from skimage import measure
        if not plane.min() < level < plane.max():
            return 0.0
        return sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in measure.find_contours(plane, level))

    def testCircleLengthMatchesCircumference(self):
        from mathlib._march import contour_length_by_slice
        y, x = np.indices((1, 200, 200))
        field = np.hypot(y * 0 + 99.5 - np.indices((1, 200, 200))[1], 99.5 - x) - 60.0
        length = contour_length_by_slice(field, 0.0)[0]
        self.assertAlmostEqual(length / (2 * np.pi * 60.0), 1.0, delta=2e-3)

    def testMatchesSkimageOnNoiseWithSaddles(self):
        from mathlib._march import contour_length_by_slice
        field = np.random.default_rng(3).normal(size=(5, 30, 30))
        expected = np.array([self._skimageLength(plane) for plane in field])
        np.testing.assert_allclose(contour_length_by_slice(field, 0.0), expected, rtol=1e-12)

    def testLevelShiftsTheContour(self):
        from mathlib._march import contour_length_by_slice
        field = np.random.default_rng(4).normal(size=(3, 24, 24))
        expected = np.array([self._skimageLength(plane, 0.5) for plane in field])
        np.testing.assert_allclose(contour_length_by_slice(field, 0.5), expected, rtol=1e-12)

    def testSliceWithoutSurfaceHasZeroLength(self):
        from mathlib._march import contour_length_by_slice
        self.assertEqual(float(contour_length_by_slice(np.ones((2, 8, 8)), 0.0).sum()), 0.0)
