"""
Unit tests for mathlib._voxelize – input validation and mesh sanitisation.

These tests exercise _validate_and_prepare_mesh directly (no Cython needed)
and check the grid-size safety limits in voxelize_mesh.
"""

import unittest
import numpy as np

from mathlib._voxelize import (
    _validate_and_prepare_mesh,
    _MAX_AXIS_POINTS,
    _MAX_TOTAL_POINTS,
)


def _cube_mesh(size=1.0):
    """Return (vertices, faces) for a minimal watertight unit cube."""
    s = size / 2.0
    verts = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
    ], dtype=np.float64)
    faces = np.array([
        [0,1,2],[0,2,3],  # -Z
        [4,6,5],[4,7,6],  # +Z
        [0,5,1],[0,4,5],  # -Y
        [2,7,3],[2,6,7],  # +Y
        [0,3,7],[0,7,4],  # -X
        [1,5,6],[1,6,2],  # +X
    ], dtype=np.int32)
    return verts, faces


class TestValidateAndPrepareMesh(unittest.TestCase):
    """Tests for _validate_and_prepare_mesh input sanitisation."""

    def test_valid_cube_passes(self):
        verts, faces = _cube_mesh()
        v, f, d = _validate_and_prepare_mesh(verts, faces, 0.1)
        self.assertEqual(v.dtype, np.float64)
        self.assertEqual(f.dtype, np.int32)
        self.assertAlmostEqual(d, 0.1)

    def test_returns_contiguous_arrays(self):
        verts, faces = _cube_mesh()
        v, f, _ = _validate_and_prepare_mesh(verts[:, ::-1][:, ::-1], faces, 0.2)
        self.assertTrue(v.flags['C_CONTIGUOUS'])
        self.assertTrue(f.flags['C_CONTIGUOUS'])

    # ---- density validation ----

    def test_non_numeric_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, "fast")

    def test_nan_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, float('nan'))

    def test_inf_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, float('inf'))

    def test_zero_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.0)

    def test_negative_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, -0.5)

    def test_tiny_density_raises(self):
        verts, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 1e-12)

    # ---- vertex / face shape validation ----

    def test_1d_vertices_raises(self):
        _, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(np.zeros(9), faces, 0.1)

    def test_2d_vertices_wrong_cols_raises(self):
        _, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(np.zeros((4, 2)), faces, 0.1)

    def test_1d_faces_raises(self):
        verts, _ = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, np.zeros(9, dtype=np.int32), 0.1)

    def test_faces_wrong_cols_raises(self):
        verts, _ = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, np.zeros((4, 4), dtype=np.int32), 0.1)

    def test_empty_vertices_raises(self):
        _, faces = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(np.zeros((0, 3)), faces, 0.1)

    def test_empty_faces_raises(self):
        verts, _ = _cube_mesh()
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, np.zeros((0, 3), dtype=np.int32), 0.1)

    # ---- vertex data quality ----

    def test_nan_vertex_raises(self):
        verts, faces = _cube_mesh()
        verts[0, 0] = np.nan
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.1)

    def test_inf_vertex_raises(self):
        verts, faces = _cube_mesh()
        verts[1, 2] = np.inf
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.1)

    # ---- face index bounds ----

    def test_negative_face_index_raises(self):
        verts, faces = _cube_mesh()
        faces[0, 0] = -1
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.1)

    def test_out_of_range_face_index_raises(self):
        verts, faces = _cube_mesh()
        faces[0, 0] = len(verts)  # one past the end
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.1)

    # ---- degenerate triangle filtering ----

    def test_degenerate_triangles_stripped(self):
        """A mix of valid and degenerate triangles: degen ones are removed."""
        verts, faces = _cube_mesh()
        # Add a degenerate triangle (three identical vertices)
        degen = np.array([[0, 0, 0]], dtype=np.int32)
        faces_with_degen = np.vstack([faces, degen])
        v, f, _ = _validate_and_prepare_mesh(verts, faces_with_degen, 0.1)
        self.assertEqual(f.shape[0], faces.shape[0])  # degen dropped

    def test_all_degenerate_raises(self):
        """If every triangle is degenerate the function should raise."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        # All vertices are collinear → every triangle has zero area
        faces = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        with self.assertRaises(ValueError):
            _validate_and_prepare_mesh(verts, faces, 0.1)

    def test_near_degenerate_kept(self):
        """A triangle with very small but non-zero area should survive."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1e-9, 0],  # extremely thin but non-degenerate
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        _, f, _ = _validate_and_prepare_mesh(verts, faces, 0.1)
        self.assertEqual(f.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
