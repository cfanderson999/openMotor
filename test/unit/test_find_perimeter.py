"""
Unit tests for find_perimeter (marching-squares 2D perimeter finder).
"""

import unittest
import numpy as np
import mathlib


class TestFindPerimeter(unittest.TestCase):
    """Edge cases for mathlib.find_perimeter."""

    def test_gradient_field_nonzero_perimeter(self):
        """A radial distance field should produce a positive contour perimeter."""
        x = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, x)
        field = np.sqrt(X**2 + Y**2)
        p, _ = mathlib.find_perimeter(field, 0.5)
        self.assertGreater(p, 0.0)

    def test_all_zeros_yields_zero_perimeter(self):
        """Uniform array (below level) → no contour → zero perimeter."""
        a = np.zeros((5, 5))
        p, _ = mathlib.find_perimeter(a, 0.5)
        self.assertAlmostEqual(p, 0.0)

    def test_all_ones_yields_zero_perimeter(self):
        """Uniform array (above level) → no contour → zero perimeter."""
        a = np.ones((5, 5))
        p, _ = mathlib.find_perimeter(a, 0.5)
        self.assertAlmostEqual(p, 0.0)

    def test_circle_distance_perimeter_scales_with_radius(self):
        """Larger radius iso-contour should have proportionally larger perimeter."""
        x = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, x)
        field = np.sqrt(X**2 + Y**2)
        p_small, _ = mathlib.find_perimeter(field, 0.3)
        p_large, _ = mathlib.find_perimeter(field, 0.6)
        # Perimeter should roughly double when radius doubles
        ratio = p_large / p_small
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)

    def test_too_small_array_raises(self):
        """Array smaller than 2×2 should raise ValueError."""
        with self.assertRaises(ValueError):
            mathlib.find_perimeter(np.array([[1.0]]), 0.5)

    def test_3d_array_raises(self):
        with self.assertRaises(ValueError):
            mathlib.find_perimeter(np.ones((3, 3, 3)), 0.5)

    def test_contours_returned_when_requested(self):
        x = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, x)
        field = np.sqrt(X**2 + Y**2)
        p, contours = mathlib.find_perimeter(field, 0.5, including_contours=True)
        self.assertIsInstance(contours, list)
        self.assertGreater(p, 0.0)

    def test_contours_not_returned_by_default(self):
        x = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, x)
        field = np.sqrt(X**2 + Y**2)
        _, contours = mathlib.find_perimeter(field, 0.5)
        self.assertEqual(len(contours), 0)


if __name__ == '__main__':
    unittest.main()
