import unittest

import numpy as np

from mathlib import _raycast
from mathlib._raycast import BORE, BOTH, CORE, CUT, END, GRAIN, NOTHING, WALL
from uilib import coreShading, orbit

SLICES, SIZE_MAP, BORE_RADIUS, WALL_RADIUS = 12, 40, 6.0, 18.0


def castCore(regression, valid, *args):
    """`_raycast.castCore` with the casing given as the 3-D mask of the tube, as the tests build it."""
    return _raycast.castCore(regression, valid[0].astype(np.float32), *args)


def _tube():
    """A grain in slice order with a bore of BORE_RADIUS along its axis: the distance from the bore in propellant."""
    axis = (np.arange(SIZE_MAP) + 0.5) - 0.5 * SIZE_MAP
    radius = np.hypot(axis[:, None], axis[None, :])
    regression = np.broadcast_to((radius - BORE_RADIUS).astype(np.float32), (SLICES, SIZE_MAP, SIZE_MAP)).copy()
    valid = np.broadcast_to(radius < WALL_RADIUS, regression.shape).copy()
    return regression, valid


# In (slice, y, x) order: the camera looks along -y from +y, the long axis runs across the screen, as in the viewer.
RIGHT, UP, TOWARD = np.array([-1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0])


NO_CUT = np.full(3, 1e9)
CUT_AT_AXIS = np.array([1e9, 0.5 * SIZE_MAP, 1e9])  # the half with y above the axis is cut away


def _cast(depth=0.0, toward=TOWARD, right=RIGHT, up=UP, size=64, limits=CUT_AT_AXIS, shown=GRAIN):
    regression, valid = _tube()
    return castCore(regression, valid, np.full(SLICES, depth), right, up, toward, limits, shown, np.zeros(2), 1.0, size, size, 0.5)


class TestCastCore(unittest.TestCase):
    def test_looking_into_the_cut_the_bore_is_a_channel_in_the_cut_face(self):
        kind, _, shade = _cast()
        middle = kind.shape[0] // 2
        across = kind[:, middle]
        self.assertEqual(across[middle], BORE)
        self.assertEqual(across[middle - int(BORE_RADIUS) - 3], CUT)
        self.assertEqual(across[middle + int(BORE_RADIUS) + 3], CUT)
        self.assertGreater(float(shade[middle, middle]), 0.8)  # the cut face of the bore squarely faces the camera

    def test_the_bore_widens_as_the_core_burns(self):
        widths = [int(np.count_nonzero(_cast(depth)[0][:, 32] == BORE)) for depth in (0.0, 4.0)]
        self.assertGreater(widths[1], widths[0] + 4)

    def test_a_slice_burnt_deeper_shows_a_wider_channel(self):
        regression, valid = _tube()
        depths = np.where(np.arange(SLICES) < SLICES // 2, 0.0, 6.0)
        kind, _, _ = castCore(regression, valid, depths, RIGHT, UP, TOWARD, CUT_AT_AXIS, GRAIN, np.zeros(2), 1.0, 64, 64, 0.5)
        narrow, wide = (np.count_nonzero(kind[:, column] == BORE) for column in (32 + 3, 32 - 3))
        self.assertNotEqual(narrow, wide)

    def test_a_bore_hit_reports_the_slice_it_is_on(self):
        kind, hitSlice, _ = _cast()
        columns = np.nonzero(kind[32] == BORE)[0]
        slices = hitSlice[32, columns]
        self.assertEqual(set(slices.tolist()), set(range(SLICES)))
        self.assertTrue(np.all(np.diff(slices) <= 0))

    def test_from_the_other_side_the_outside_of_the_grain_is_seen(self):
        kind, _, _ = _cast(toward=-TOWARD, right=-RIGHT)
        self.assertEqual(np.count_nonzero(kind == BORE), 0)
        self.assertGreater(np.count_nonzero(kind == WALL), 0)

    def test_from_the_side_the_casing_wall_and_the_ends_are_seen(self):
        kind, _, _ = _cast(toward=np.array([0.0, 0.0, 1.0]), right=np.array([-1.0, 0, 0]), up=np.array([0, 1.0, 0]))
        self.assertGreater(np.count_nonzero(kind == WALL), 0)
        kind, _, _ = _cast(toward=np.array([1.0, 0.0, 0.0]), right=np.array([0, 0, 1.0]), up=np.array([0, 1.0, 0]))
        self.assertGreater(np.count_nonzero(kind == END), 0)

    def test_pixels_that_miss_the_volume_show_nothing(self):
        kind, _, _ = _cast(size=64)
        self.assertEqual(kind[0, 0], NOTHING)


class TestWhatIsShown(unittest.TestCase):
    def test_the_core_alone_is_drawn_as_a_solid_where_the_grain_has_its_bore(self):
        kind, _, _ = _cast(shown=CORE, limits=NO_CUT, toward=-TOWARD, right=-RIGHT)
        middle = kind.shape[0] // 2
        self.assertEqual(kind[middle, middle], BORE)
        self.assertEqual(kind[middle, middle + int(BORE_RADIUS) + 3], NOTHING)

    def test_the_grain_and_the_core_together_show_both_on_a_cut_face(self):
        kind, _, _ = _cast(shown=BOTH)
        middle = kind.shape[0] // 2
        self.assertEqual(kind[middle, middle], BORE)
        self.assertEqual(kind[middle - int(BORE_RADIUS) - 3, middle], CUT)

    def test_uncut_the_grain_and_the_core_together_look_like_the_grain(self):
        both = _cast(shown=BOTH, limits=NO_CUT, toward=-TOWARD, right=-RIGHT)[0]
        grain = _cast(shown=GRAIN, limits=NO_CUT, toward=-TOWARD, right=-RIGHT)[0]
        np.testing.assert_array_equal(both, grain)

    def test_a_plane_across_the_axis_cuts_the_grain_short_and_shows_its_cross_section(self):
        limits = np.array([SLICES / 2, 1e9, 1e9])
        kind, hitSlice, _ = _cast(shown=BOTH, limits=limits, toward=np.array([1.0, 0, 0]), right=np.array([0, 0, 1.0]),
                                  up=np.array([0, 1.0, 0]))
        self.assertGreater(np.count_nonzero(kind == CUT), 0)
        self.assertGreater(np.count_nonzero(kind == BORE), 0)
        self.assertLessEqual(int(hitSlice[kind == CUT].max()), SLICES // 2)


class TestSurfaceDetails(unittest.TestCase):
    def test_a_core_burned_to_the_casing_is_still_the_core_where_it_meets_the_wall(self):
        regression, valid = _tube()
        kind, _, _ = castCore(regression, valid, np.full(SLICES, WALL_RADIUS + 5.0), -RIGHT, UP, -TOWARD,
                              NO_CUT, CORE, np.zeros(2), 1.0, 64, 64, 0.5)
        self.assertGreater(np.count_nonzero(kind == BORE), 0)
        self.assertEqual(np.count_nonzero(kind == WALL), 0)

    def test_the_picture_can_be_moved_across_the_screen(self):
        regression, valid = _tube()
        args = (np.full(SLICES, 0.0), RIGHT, UP, TOWARD, CUT_AT_AXIS, GRAIN)
        centered = castCore(regression, valid, *args, np.zeros(2), 1.0, 64, 64, 0.5)[0]
        moved = castCore(regression, valid, *args, np.array([0.0, 8.0]), 1.0, 64, 64, 0.5)[0]
        rows = lambda kind: np.nonzero(np.any(kind == BORE, axis=1))[0].mean()
        self.assertAlmostEqual(rows(moved) - rows(centered), 8.0, delta=0.5)  # centred higher up, the bore sits lower

    def test_the_bore_edge_follows_the_depth_to_less_than_a_voxel(self):
        """Interpolating the map puts the surface where the depth says, not at the nearest voxel."""
        for depth in (3.0, 3.3, 3.6, 3.9):
            regression, valid = _tube()
            kind, _, _ = castCore(regression, valid, np.full(SLICES, depth), RIGHT, UP, TOWARD, CUT_AT_AXIS, GRAIN,
                                  np.zeros(2), 0.2, 320, 320, 0.5)
            radius = np.count_nonzero(kind[:, 160] == BORE) * 0.2 / 2
            self.assertAlmostEqual(radius, BORE_RADIUS + depth, delta=0.25)

class TestPictureShape(unittest.TestCase):
    def test_the_picture_can_be_wider_than_it_is_tall_and_shows_more_along_its_width(self):
        regression, valid = _tube()
        args = (np.zeros(SLICES), RIGHT, UP, TOWARD, CUT_AT_AXIS, GRAIN, np.zeros(2))
        kind, hitSlice, shade = castCore(regression, valid, *args, 1.0, 32, 96, 0.5)
        self.assertEqual(kind.shape, (32, 96))
        self.assertEqual(hitSlice.shape, shade.shape)
        square = castCore(regression, valid, *args, 1.0, 32, 32, 0.5)[0]
        self.assertEqual(np.count_nonzero(square == BORE), np.count_nonzero(kind[:, 32:64] == BORE))


class TestConstants(unittest.TestCase):
    def test_the_values_are_those_the_3d_pane_and_the_colour_table_rely_on(self):
        self.assertEqual((GRAIN, CORE, BOTH), (0, 1, 2))
        self.assertEqual((NOTHING, BORE, CUT, WALL, END), (0, 1, 2, 3, 4))


class TestComposeScene(unittest.TestCase):
    def test_the_colours_are_in_the_order_of_the_kinds_of_surface(self):
        self.assertEqual(len(coreShading.SURFACE_COLORS), END + 1)
        self.assertEqual(coreShading.SURFACE_COLORS[NOTHING], coreShading.BACKGROUND)
        self.assertEqual(coreShading.SURFACE_COLORS[CUT], coreShading.CUT_FACE)
        self.assertEqual(coreShading.SURFACE_COLORS[END], coreShading.END_FACE)
        self.assertEqual(coreShading.BORE_KIND, BORE)

    def test_the_bore_takes_the_colour_of_its_slice_and_everything_is_lit_by_its_shade(self):
        kind = np.array([[NOTHING, BORE, CUT]], dtype=np.uint8)
        hitSlice = np.array([[0, 1, 0]], dtype=np.int32)
        shade = np.array([[1.0, 0.5, 1.0]], dtype=np.float32)
        colors = np.array([[10, 20, 30, 255], [200, 100, 50, 255]], dtype=np.uint8)
        picture = coreShading.composeScene(kind, hitSlice, shade, colors)
        self.assertEqual(tuple(picture[0, 0]), coreShading.BACKGROUND)
        self.assertEqual(tuple(picture[0, 1]), (100, 50, 25, 255))
        self.assertEqual(tuple(picture[0, 2]), coreShading.CUT_FACE)

    def test_without_colours_the_bore_is_plain(self):
        picture = coreShading.composeScene(np.array([[1]], dtype=np.uint8), np.zeros((1, 1), np.int32),
                                           np.ones((1, 1), np.float32), None)
        self.assertEqual(tuple(picture[0, 0]), coreShading.BORE_PLAIN)


class TestOrbit(unittest.TestCase):
    def test_the_default_view_looks_along_y_with_the_long_axis_across_the_screen(self):
        right, up, toward = orbit.defaultView()
        np.testing.assert_array_equal(toward, [0, 1, 0])
        np.testing.assert_array_equal(right, [-1, 0, 0])

    def test_dragging_moves_the_front_of_the_scene_the_way_the_mouse_goes(self):
        view = orbit.defaultView()
        front = view[orbit.TOWARD_CAMERA]
        self.assertGreater(front @ orbit.turn(view, 0.3, 0)[orbit.SCREEN_RIGHT], 0)
        self.assertLess(front @ orbit.turn(view, 0, 0.3)[orbit.SCREEN_UP], 0)

    def test_many_turns_keep_the_view_orthonormal_and_the_same_hand(self):
        view = orbit.defaultView()
        for _ in range(200):
            view = orbit.turn(view, 0.13, -0.07)
        np.testing.assert_allclose(view @ view.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(view)), float(np.linalg.det(orbit.defaultView())), places=9)


class TestSmoothVolume(unittest.TestCase):
    def test_a_constant_volume_is_unchanged_and_the_mean_is_kept_away_from_the_ends(self):
        volume = np.full((40, 6, 6), 2.5, np.float32)
        np.testing.assert_allclose(_raycast.smoothVolume(volume, 5, 0.2), volume, atol=1e-5)

    def test_along_the_slices_it_matches_three_running_averages(self):
        from scipy import ndimage
        volume = np.random.default_rng(1).random((60, 5, 5)).astype(np.float32)
        expected = volume
        for _ in range(3):
            expected = ndimage.uniform_filter1d(expected, size=7, axis=0, mode='nearest')
        np.testing.assert_allclose(_raycast.smoothVolume(volume, 7, 0.0), expected, atol=1e-5)

    def test_a_step_along_the_slices_becomes_a_smooth_ramp_that_does_not_overshoot(self):
        volume = np.zeros((100, 4, 4), np.float32)
        volume[50:] = 1.0
        ramp = _raycast.smoothVolume(volume, 13, 0.2)[:, 2, 2]
        self.assertTrue(np.all(np.diff(ramp) >= -1e-6))
        self.assertGreater(float(ramp[50]), 0.4)
        self.assertLess(float(ramp[40]), 0.1)
        self.assertLessEqual(float(ramp.max()), 1.0 + 1e-6)

    def test_across_each_slice_a_single_voxel_is_spread_to_its_neighbours(self):
        volume = np.zeros((3, 7, 7), np.float32)
        volume[1, 3, 3] = 1.0
        out = _raycast.smoothVolume(volume, 1, 0.25)[1]
        self.assertAlmostEqual(float(out[3, 3]), 0.25, places=5)
        self.assertAlmostEqual(float(out[3, 4]), 0.125, places=5)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
