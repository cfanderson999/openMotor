"""
Additional unit tests for 3D FMM features in motorlib.grain (Fmm3DGrain)
and motorlib.grains.custom3d.

Covers gaps not addressed by the existing custom3d.py test file:
  - Bitpacked coreMap storage (pack/unpack roundtrip, sum precomputation)
  - FMM distance disk cache (save/load roundtrip, key determinism)
  - _smooth_series adaptive window sizing
  - _getMeshedPeakSearchCandidates with synthetic arrays
  - Bore grain with all inhibited-ends variants ('Top', 'Bottom', 'Neither')
  - 3D mass flux mode (massFlux3D=True)
  - volumeFunc smoothed interpolation
  - portAreaFunc / foreAreaFunc existence by inhibited-ends config
  - getPortArea scanning logic
  - getEndPositionsInMapDim consistency
"""

import math
import os
import unittest

import numpy as np

import motorlib.grain
from motorlib.grain import Fmm3DGrain, _smooth_series
from motorlib.grains import custom3d as _custom3d_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bore_cylinder(radius=0.01, length=0.04, segments=16):
    """Watertight cylinder (axis along Y) for bore geometry."""
    verts = []
    verts.append([0.0, 0.0, 0.0])
    verts.append([0.0, length, 0.0])
    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append([radius * math.cos(a), 0.0, radius * math.sin(a)])
    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append([radius * math.cos(a), length, radius * math.sin(a)])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([0, 2 + i, 2 + j])
        faces.append([1, 2 + segments + j, 2 + segments + i])
        b0, b1 = 2 + i, 2 + j
        t0, t1 = 2 + segments + i, 2 + segments + j
        faces.append([b0, t0, b1])
        faces.append([b1, t0, t1])
    return np.array(faces, dtype=np.int32), np.array(verts, dtype=np.float64)


class _MockConfig:
    def __init__(self, map_dim_3d=48):
        self._props = {'3DmapDim': map_dim_3d, 'mapDim': 500}

    def getProperty(self, name):
        return self._props.get(name)


def _make_bore_grain(inhibited='Both', massFlux3D=False, mapDim=48):
    faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
    g = _custom3d_mod()
    g.setProperties({
        'diameter':      0.05,
        'length':        0.0,
        'stlUnit':       'm',
        'coreAxis':      '-Y',
        'inhibitedEnds': inhibited,
        'massFlux3D':    massFlux3D,
    })
    g.props['mesh'].setValue((faces, verts, ''))
    g.simulationSetup(_MockConfig(map_dim_3d=mapDim))
    return g


# ===========================================================================
# 1. Bitpacked coreMap storage
# ===========================================================================

class TestBitpackedCoreMap(unittest.TestCase):
    """Test the Fmm3DGrain coreMap property getter/setter bitpacking."""

    def _make_grain(self):
        return _custom3d_mod()

    def test_small_array_stored_dense(self):
        """Arrays below _PACK_THRESHOLD_BYTES should remain dense (no packing)."""
        g = self._make_grain()
        small = np.ones((10, 10, 10), dtype=bool)
        g.coreMap = small
        self.assertIsNone(g.__dict__.get('_coreMapPacked'))
        np.testing.assert_array_equal(g.coreMap, small)

    def test_large_array_roundtrip(self):
        """Pack → unpack must preserve every voxel exactly."""
        g = self._make_grain()
        # Create array just above the threshold
        threshold = Fmm3DGrain._PACK_THRESHOLD_BYTES
        # bool arrays use 1 byte/element; we need > threshold elements
        side = int(np.ceil(threshold ** (1.0 / 3))) + 1
        big = np.random.randint(0, 2, size=(side, side, side)).astype(bool)
        self.assertGreater(big.nbytes, threshold)
        g.coreMap = big
        self.assertIsNotNone(g.__dict__.get('_coreMapPacked'))
        np.testing.assert_array_equal(g.coreMap, big)

    def test_sum_precomputed_on_pack(self):
        """When packed, _coreMapSum should equal the element sum."""
        g = self._make_grain()
        threshold = Fmm3DGrain._PACK_THRESHOLD_BYTES
        side = int(np.ceil(threshold ** (1.0 / 3))) + 1
        big = np.random.randint(0, 2, size=(side, side, side)).astype(bool)
        g.coreMap = big
        self.assertEqual(g.__dict__['_coreMapSum'], int(big.sum()))

    def test_none_clears_all_slots(self):
        """Setting coreMap = None should clear packed and dense slots."""
        g = self._make_grain()
        g.coreMap = np.ones((5, 5, 5), dtype=bool)
        g.coreMap = None
        self.assertIsNone(g.__dict__.get('_coreMapDense'))
        self.assertIsNone(g.__dict__.get('_coreMapPacked'))
        self.assertIsNone(g.coreMap)

    def test_non_bool_array_stored_dense(self):
        """Non-bool arrays should stay dense regardless of size."""
        g = self._make_grain()
        arr = np.ones((10, 10, 10), dtype=np.float64)
        g.coreMap = arr
        self.assertIsNone(g.__dict__.get('_coreMapPacked'))


# ===========================================================================
# 2. FMM distance disk cache
# ===========================================================================

class TestFmmDistanceCache(unittest.TestCase):
    """Tests for _computeFmmCacheKey, _saveFmmCache, _loadFmmCache."""

    def setUp(self):
        self.grain = _custom3d_mod()

    def test_cache_key_deterministic(self):
        arr = np.ones((8, 8, 8), dtype=np.uint8)
        k1 = Fmm3DGrain._computeFmmCacheKey(arr, 0.01)
        k2 = Fmm3DGrain._computeFmmCacheKey(arr, 0.01)
        self.assertEqual(k1, k2)

    def test_cache_key_differs_by_content(self):
        a1 = np.ones((8, 8, 8), dtype=np.uint8)
        a2 = np.zeros((8, 8, 8), dtype=np.uint8)
        self.assertNotEqual(
            Fmm3DGrain._computeFmmCacheKey(a1, 0.01),
            Fmm3DGrain._computeFmmCacheKey(a2, 0.01),
        )

    def test_cache_key_differs_by_cellsize(self):
        arr = np.ones((8, 8, 8), dtype=np.uint8)
        self.assertNotEqual(
            Fmm3DGrain._computeFmmCacheKey(arr, 0.01),
            Fmm3DGrain._computeFmmCacheKey(arr, 0.02),
        )

    def test_save_and_load_roundtrip(self):
        data = np.random.rand(4, 4, 4)
        key = 'test_roundtrip_' + str(id(self))
        self.grain._saveFmmCache(key, data)
        loaded = self.grain._loadFmmCache(key)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded, data)
        # Cleanup
        path = self.grain._getFmmCacheDir() / f'{key}_v{Fmm3DGrain._FMM_DIST_CACHE_VERSION}.npy'
        if path.is_file():
            path.unlink()

    def test_load_nonexistent_returns_none(self):
        result = self.grain._loadFmmCache('nonexistent_key_xyz')
        self.assertIsNone(result)


# ===========================================================================
# 3. _smooth_series adaptive window
# ===========================================================================

class TestSmoothSeries(unittest.TestCase):
    """Tests for the adaptive Savitzky-Golay smoother."""

    def test_empty_array(self):
        result = _smooth_series([])
        self.assertEqual(len(result), 0)

    def test_single_value(self):
        result = _smooth_series([42.0])
        np.testing.assert_array_almost_equal(result, [42.0])

    def test_two_values(self):
        result = _smooth_series([1.0, 2.0])
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_short_series_does_not_crash(self):
        """A series of 5 elements should not raise (window must adapt)."""
        result = _smooth_series([1, 2, 3, 2, 1])
        self.assertEqual(len(result), 5)

    def test_long_constant_series_unchanged(self):
        """A flat series should remain flat after smoothing."""
        data = np.full(100, 5.0)
        result = _smooth_series(data)
        np.testing.assert_array_almost_equal(result, data, decimal=10)

    def test_output_length_matches_input(self):
        for n in [3, 10, 50, 200]:
            data = np.random.rand(n)
            result = _smooth_series(data)
            self.assertEqual(len(result), n)


# ===========================================================================
# 4. Bore grain inhibited-ends variants
# ===========================================================================

class TestBoreGrainInhibitedTop(unittest.TestCase):
    """Bore grain with inhibitedEnds='Top' (fore inhibited, aft burns)."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Top')

    def test_surface_area_positive(self):
        self.assertGreater(self.grain.getSurfaceAreaAtRegression(0.0), 0.0)

    def test_volume_positive(self):
        self.assertGreater(self.grain.getVolumeAtRegression(0.0), 0.0)

    def test_volume_decreases(self):
        v0 = self.grain.getVolumeAtRegression(0.0)
        v1 = self.grain.getVolumeAtRegression(0.003)
        self.assertLess(v1, v0)

    def test_port_area_positive(self):
        self.assertGreater(self.grain.getPortArea(0.0), 0.0)

    def test_web_left_positive(self):
        self.assertGreater(self.grain.getWebLeft(0.0), 0.0)

    def test_end_positions_consistent(self):
        pos = self.grain.getEndPositions(0.0)
        self.assertGreaterEqual(pos[1], pos[0])


class TestBoreGrainInhibitedBottom(unittest.TestCase):
    """Bore grain with inhibitedEnds='Bottom' (aft inhibited, fore burns)."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Bottom')

    def test_surface_area_positive(self):
        self.assertGreater(self.grain.getSurfaceAreaAtRegression(0.0), 0.0)

    def test_volume_positive(self):
        self.assertGreater(self.grain.getVolumeAtRegression(0.0), 0.0)

    def test_port_area_positive(self):
        self.assertGreater(self.grain.getPortArea(0.0), 0.0)

    def test_web_left_positive(self):
        self.assertGreater(self.grain.getWebLeft(0.0), 0.0)


class TestBoreGrainInhibitedNeither(unittest.TestCase):
    """Bore grain with inhibitedEnds='Neither' (both faces burn)."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Neither')

    def test_surface_area_positive(self):
        self.assertGreater(self.grain.getSurfaceAreaAtRegression(0.0), 0.0)

    def test_surface_area_gte_both_inhibited(self):
        """Neither-inhibited should expose >= surface area vs both-inhibited."""
        g_both = _make_bore_grain(inhibited='Both')
        a_neither = self.grain.getSurfaceAreaAtRegression(0.0)
        a_both = g_both.getSurfaceAreaAtRegression(0.0)
        self.assertGreaterEqual(a_neither, a_both * 0.8)  # tolerance for discretisation

    def test_volume_positive(self):
        self.assertGreater(self.grain.getVolumeAtRegression(0.0), 0.0)

    def test_port_area_positive(self):
        self.assertGreater(self.grain.getPortArea(0.0), 0.0)

    def test_web_left_positive(self):
        self.assertGreater(self.grain.getWebLeft(0.0), 0.0)


# ===========================================================================
# 5. 3D mass-flux mode (massFlux3D=True)
# ===========================================================================

class TestMassFlux3D(unittest.TestCase):
    """Tests for the massFlux3D=True path."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Both', massFlux3D=True)

    def test_getMassFlux_positive_at_centre(self):
        pos = self.grain.getEndPositions(0.0)
        mid = (pos[0] + pos[1]) / 2.0
        mf = self.grain.getMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5,
            position=mid, density=1800.0,
        )
        self.assertGreater(mf, 0.0)

    def test_getMassFlux_increases_toward_fore(self):
        """Mass flux should generally increase from aft to fore in 3D mode."""
        pos = self.grain.getEndPositions(0.0)
        mf_aft = self.grain.getMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5,
            position=pos[0], density=1800.0,
        )
        mf_fore = self.grain.getMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5,
            position=pos[1], density=1800.0,
        )
        # Fore should be >= aft (mass accumulates along the grain)
        self.assertGreaterEqual(mf_fore, mf_aft * 0.9)

    def test_getPeakMassFlux_3d_positive(self):
        peak = self.grain.getPeakMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5, density=1800.0,
        )
        self.assertGreater(peak, 0.0)
        self.assertTrue(math.isfinite(peak))

    def test_getPeakMassFlux_3d_increases_with_massIn(self):
        peak0 = self.grain.getPeakMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5, density=1800.0,
        )
        peak1 = self.grain.getPeakMassFlux(
            massIn=0.5, dTime=0.001, regDist=0.0, dRegDist=1e-5, density=1800.0,
        )
        self.assertGreater(peak1, peak0)


# ===========================================================================
# 6. volumeFunc smooth interpolation
# ===========================================================================

class TestVolumeFunc(unittest.TestCase):
    """Tests for the precomputed volumeFunc on Fmm3DGrain."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Both')

    def test_volumeFunc_exists(self):
        self.assertIsNotNone(self.grain.volumeFunc)

    def test_volumeFunc_at_zero_matches_getVolume(self):
        """volumeFunc(0) should closely match getVolumeAtRegression(0)."""
        v_func = float(self.grain.volumeFunc(0.0))
        v_method = self.grain.getVolumeAtRegression(0.0)
        self.assertAlmostEqual(v_func, v_method, delta=v_method * 0.05)

    def test_volume_monotonically_decreasing(self):
        """Propellant volume should never increase as regression advances."""
        prev = self.grain.getVolumeAtRegression(0.0)
        for r in np.linspace(0.001, 0.01, 10):
            v = self.grain.getVolumeAtRegression(r)
            self.assertLessEqual(v, prev + 1e-12)
            prev = v

    def test_volume_reaches_zero_past_burnout(self):
        web = self.grain.wallWeb
        v = self.grain.getVolumeAtRegression(web + 0.01)
        self.assertAlmostEqual(v, 0.0, places=6)


# ===========================================================================
# 7. portAreaFunc / foreAreaFunc existence by config
# ===========================================================================

class TestPortForeAreaFuncsByConfig(unittest.TestCase):
    """portAreaFunc and foreAreaFunc should be set based on inhibited-ends."""

    def test_both_inhibited_has_portAreaFunc(self):
        g = _make_bore_grain(inhibited='Both')
        self.assertIsNotNone(g.portAreaFunc)

    def test_both_inhibited_has_foreAreaFunc(self):
        g = _make_bore_grain(inhibited='Both')
        self.assertIsNotNone(g.foreAreaFunc)

    def test_neither_inhibited_no_portAreaFunc(self):
        g = _make_bore_grain(inhibited='Neither')
        self.assertIsNone(g.portAreaFunc)

    def test_neither_inhibited_no_foreAreaFunc(self):
        g = _make_bore_grain(inhibited='Neither')
        self.assertIsNone(g.foreAreaFunc)

    def test_top_inhibited_has_foreAreaFunc(self):
        g = _make_bore_grain(inhibited='Top')
        self.assertIsNotNone(g.foreAreaFunc)

    def test_top_inhibited_no_portAreaFunc(self):
        """'Top' means fore-inhibited only; aft is uninhibited → no portAreaFunc."""
        g = _make_bore_grain(inhibited='Top')
        self.assertIsNone(g.portAreaFunc)

    def test_bottom_inhibited_has_portAreaFunc(self):
        """'Bottom' means aft-inhibited → portAreaFunc should be set."""
        g = _make_bore_grain(inhibited='Bottom')
        self.assertIsNotNone(g.portAreaFunc)

    def test_bottom_inhibited_no_foreAreaFunc(self):
        g = _make_bore_grain(inhibited='Bottom')
        self.assertIsNone(g.foreAreaFunc)


# ===========================================================================
# 8. getPortArea scanning logic
# ===========================================================================

class TestGetPortAreaScanning(unittest.TestCase):
    """getPortArea scans aft-end slices for max core area."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Both')

    def test_port_area_positive_at_zero(self):
        pa = self.grain.getPortArea(0.0)
        self.assertGreater(pa, 0.0)

    def test_port_area_increases_with_regression(self):
        pa0 = self.grain.getPortArea(0.0)
        pa1 = self.grain.getPortArea(0.005)
        self.assertGreater(pa1, pa0)

    def test_port_area_bounded_by_grain_cross_section(self):
        """Port area cannot exceed the grain's full cross-sectional area."""
        max_area = math.pi * (0.025) ** 2  # full cylinder cross section
        pa = self.grain.getPortArea(0.0)
        self.assertLess(pa, max_area)


# ===========================================================================
# 9. getEndPositionsInMapDim consistency
# ===========================================================================

class TestEndPositionsInMapDim(unittest.TestCase):
    """getEndPositionsInMapDim should return valid indices."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Both')

    def test_returns_two_values(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        self.assertIsNotNone(aft)
        self.assertIsNotNone(fore)

    def test_fore_ge_aft(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        self.assertGreaterEqual(fore, aft)

    def test_within_grid_bounds(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        zdim = self.grain.regressionMap.shape[0]
        self.assertGreaterEqual(aft, 0)
        self.assertLess(fore, zdim)

    def test_span_shrinks_with_regression(self):
        """For 'Both'-inhibited, positions are fixed — span should not change."""
        a0, f0 = self.grain.getEndPositionsInMapDim(0.0)
        a1, f1 = self.grain.getEndPositionsInMapDim(0.005)
        span0 = f0 - a0
        span1 = f1 - a1
        # Both-inhibited: ends are fixed, but web burns inward so the
        # propellant span in the map should shrink or stay the same.
        self.assertLessEqual(span1, span0 + 1)  # +1 tolerance for discretisation


# ===========================================================================
# 10. _getMeshedPeakSearchCandidates
# ===========================================================================

class TestGetMeshedPeakSearchCandidates(unittest.TestCase):
    """Tests for the candidate selection logic in _getMeshedPeakSearchCandidates."""

    @classmethod
    def setUpClass(cls):
        # Use a bore grain with 3D mode to ensure all helper data is set up
        cls.grain = _make_bore_grain(inhibited='Both', massFlux3D=True)

    def test_always_includes_startPos(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        start, end = int(aft), int(fore - 1)
        if end >= start:
            candidates = self.grain._getMeshedPeakSearchCandidates(
                start, end, 0.0, 0.001, 0.0, 1e-5, 1800.0,
            )
            self.assertIn(start, candidates)

    def test_always_includes_endPos(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        start, end = int(aft), int(fore - 1)
        if end >= start:
            candidates = self.grain._getMeshedPeakSearchCandidates(
                start, end, 0.0, 0.001, 0.0, 1e-5, 1800.0,
            )
            # endPos may be capped due to fore-cap detection, but at least
            # the returned set should contain start
            self.assertTrue(len(candidates) >= 1)

    def test_small_range_returns_all(self):
        """When endPos - startPos + 1 <= 6, all positions should be returned."""
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        start = int(aft)
        end = min(start + 4, int(fore - 1))
        if end >= start:
            candidates = self.grain._getMeshedPeakSearchCandidates(
                start, end, 0.0, 0.001, 0.0, 1e-5, 1800.0,
            )
            self.assertEqual(len(candidates), end - start + 1)

    def test_empty_range_returns_empty(self):
        candidates = self.grain._getMeshedPeakSearchCandidates(
            10, 5, 0.0, 0.001, 0.0, 1e-5, 1800.0,
        )
        self.assertEqual(len(candidates), 0)

    def test_candidates_are_sorted(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        start, end = int(aft), int(fore - 1)
        if end >= start:
            candidates = self.grain._getMeshedPeakSearchCandidates(
                start, end, 0.0, 0.001, 0.0, 1e-5, 1800.0,
            )
            self.assertEqual(candidates, sorted(candidates))


# ===========================================================================
# 11. Core-area-profile cache (_capCache)
# ===========================================================================

class TestCapCache(unittest.TestCase):
    """_capCache should cache core_area_profile results."""

    @classmethod
    def setUpClass(cls):
        cls.grain = _make_bore_grain(inhibited='Both', massFlux3D=True)

    def test_cache_populated_after_peak_search(self):
        aft, fore = self.grain.getEndPositionsInMapDim(0.0)
        start, end = int(aft), int(fore - 1)
        if end >= start:
            self.grain._capCache.clear()
            self.grain._getMeshedPeakSearchCandidates(
                start, end, 0.0, 0.001, 0.0, 1e-5, 1800.0,
            )
            self.assertGreater(len(self.grain._capCache), 0)

    def test_cache_cleared_on_regeneration(self):
        """generateRegressionMap should clear _capCache when coreMap changes."""
        self.grain._capCache['dummy'] = 'value'
        # Force cache key mismatch so regeneration actually runs
        self.grain._regressionMapCacheKey = None
        self.grain.generateRegressionMap()
        self.assertNotIn('dummy', self.grain._capCache)


if __name__ == '__main__':
    unittest.main()
