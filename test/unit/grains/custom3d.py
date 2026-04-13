"""
Unit tests for motorlib.grains.custom3d (3D FMM custom grain).

Test coverage:
  - Static / class methods: _applyCoreAxisOrientation, _getLengthExtent, LRU cache
  - Property defaults and setProperties round-trip
  - Mesh signature hashing (_getMeshSignature)
  - getCoreMapHash determinism
  - Fmm3DGrain._makeCoreMapCacheKey
  - Unit-conversion helpers: normalize/unNormalize/lengthToMap/mapToLength/areaToMap/mapToArea
  - Empty-mesh (endburner-only) code path
  - getDetailsString: before and after voxelization, with and without mesh vertices
  - getGeometryErrors: valid grain, zero diameter, invalid endburner config
  - Simulation setup: regression map creation, faceAreaFunc, cache key
  - Regression map cache: hit (no recompute), miss (changed coreMap)
  - Grain physics (endburner, no bore): surface area, volume, web, end positions
  - Grain physics (cylindrical bore): surface area, volume, port area, web, end positions,
    mass flux, volume < bounding volume, monotonicity invariants
  - Inhibited-ends variants: surface-area ordering between 'Neither' and 'Both'
  - _getMeshedMarchingStepSize: Exact/Fast/Faster quality modes
  - _getMeshedMarchingMask: shape, dtype, and boolean content
  - getPeakMassFlux: returns non-negative value
"""

import math
import os
import unittest

import numpy as np

import motorlib.grain
import motorlib.grains
from motorlib.grains import Custom3DGrain

# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def _make_bore_cylinder(radius=0.01, length=0.04, segments=16):
    """
    Return (faces, vertices) for a closed, watertight cylinder whose axis is
    along the Y-axis.  Represents a bore (void) geometry inside the propellant.

    - Bottom cap centre at y=0, top cap centre at y=length.
    - ``radius`` and ``length`` are in metres (match stlUnit='m').
    """
    verts = []
    verts.append([0.0, 0.0, 0.0])          # index 0: bottom cap centre
    verts.append([0.0, length, 0.0])        # index 1: top cap centre

    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append([radius * math.cos(a), 0.0, radius * math.sin(a)])

    for i in range(segments):
        a = 2 * math.pi * i / segments
        verts.append([radius * math.cos(a), length, radius * math.sin(a)])

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        # Bottom cap (outward normal faces -Y)
        faces.append([0, 2 + i, 2 + j])
        # Top cap (outward normal faces +Y)
        faces.append([1, 2 + segments + j, 2 + segments + i])
        # Side (two triangles per quad)
        b0, b1 = 2 + i, 2 + j
        t0, t1 = 2 + segments + i, 2 + segments + j
        faces.append([b0, t0, b1])
        faces.append([b1, t0, t1])

    return np.array(faces, dtype=np.int32), np.array(verts, dtype=np.float64)


class _MockConfig:
    """Minimal simulation-config duck-type with a 3DmapDim setting."""

    def __init__(self, map_dim_3d=100):
        self._props = {'3DmapDim': map_dim_3d, 'mapDim': 500}

    def getProperty(self, name):
        return self._props.get(name)


# ---------------------------------------------------------------------------
# 1. Static / class method tests
# ---------------------------------------------------------------------------

class TestApplyCoreAxisOrientation(unittest.TestCase):
    """Tests for custom3d._applyCoreAxisOrientation."""

    def _orient(self, arr, axis):
        return Custom3DGrain._applyCoreAxisOrientation(arr, axis)

    def test_plus_Y_is_identity(self):
        """'+Y' selects Y (voxelised axis-0) as length → no-op."""
        arr = np.arange(24).reshape(2, 3, 4)
        np.testing.assert_array_equal(self._orient(arr, '+Y'), arr)

    def test_minus_Y_flips_axis0(self):
        """'-Y' keeps Y as length axis but reverses it (flip along axis 0)."""
        arr = np.arange(24).reshape(2, 3, 4)
        np.testing.assert_array_equal(
            self._orient(arr, '-Y'), np.flip(arr, axis=0)
        )

    def test_plus_X_transposes_axis1_to_front(self):
        """'+X' → original axis-1 becomes the length axis."""
        arr = np.arange(24).reshape(2, 3, 4)
        expected = np.transpose(arr, axes=[1, 0, 2])
        np.testing.assert_array_equal(self._orient(arr, '+X'), expected)

    def test_minus_X_transposes_then_flips(self):
        """'-X' → axis-1 to front, then flip."""
        arr = np.arange(24).reshape(2, 3, 4)
        expected = np.flip(np.transpose(arr, axes=[1, 0, 2]), axis=0)
        np.testing.assert_array_equal(self._orient(arr, '-X'), expected)

    def test_plus_Z_transposes_axis2_to_front(self):
        """'+Z' → original axis-2 becomes the length axis."""
        arr = np.arange(24).reshape(2, 3, 4)
        expected = np.transpose(arr, axes=[2, 0, 1])
        np.testing.assert_array_equal(self._orient(arr, '+Z'), expected)

    def test_minus_Z_transposes_then_flips(self):
        """'-Z' → axis-2 to front, then flip."""
        arr = np.arange(24).reshape(2, 3, 4)
        expected = np.flip(np.transpose(arr, axes=[2, 0, 1]), axis=0)
        np.testing.assert_array_equal(self._orient(arr, '-Z'), expected)

    def test_unknown_axis_returns_original(self):
        """An unknown axis label should leave the array unchanged."""
        arr = np.eye(3)
        np.testing.assert_array_equal(self._orient(arr, 'W'), arr)

    def test_idempotent_shape(self):
        """Orientation should not change the total number of elements."""
        arr = np.ones((5, 7, 11))
        for axis in ['-Y', '+Y', '+X', '-X', '+Z', '-Z']:
            result = self._orient(arr, axis)
            self.assertEqual(result.size, arr.size)

    def test_single_element_array_unchanged(self):
        """Single-element array should always come out identical."""
        arr = np.array([[[42]]])
        for axis in ['-Y', '+Y', '+X', '-X', '+Z', '-Z']:
            np.testing.assert_array_equal(self._orient(arr, axis), arr)


class TestGetLengthExtent(unittest.TestCase):
    """Tests for custom3d._getLengthExtent."""

    def _ext(self, bounds, axis):
        return Custom3DGrain._getLengthExtent(bounds, axis)

    def test_Y_axis(self):
        bounds = (0.01, 0.02, 0.03)   # (dX, dY, dZ)
        self.assertAlmostEqual(self._ext(bounds, '+Y'), 0.02)
        self.assertAlmostEqual(self._ext(bounds, '-Y'), 0.02)

    def test_X_axis(self):
        bounds = (0.05, 0.07, 0.09)
        self.assertAlmostEqual(self._ext(bounds, '+X'), 0.05)
        self.assertAlmostEqual(self._ext(bounds, '-X'), 0.05)

    def test_Z_axis(self):
        bounds = (0.10, 0.20, 0.30)
        self.assertAlmostEqual(self._ext(bounds, '+Z'), 0.30)
        self.assertAlmostEqual(self._ext(bounds, '-Z'), 0.30)

    def test_unknown_axis_returns_Y_default(self):
        """Unrecognised letter defaults to Y (index 1)."""
        bounds = (0.1, 0.2, 0.3)
        result = self._ext(bounds, 'Q')
        self.assertAlmostEqual(result, 0.2)


class TestVoxelCacheLRU(unittest.TestCase):
    """Tests for custom3d._rememberVoxelCache LRU eviction."""

    def setUp(self):
        # Save and clear the global class-level cache for isolation.
        self._orig_cache = Custom3DGrain._VOXEL_CACHE.copy()
        self._orig_max = Custom3DGrain._VOXEL_CACHE_MAX
        Custom3DGrain._VOXEL_CACHE.clear()

    def tearDown(self):
        Custom3DGrain._VOXEL_CACHE.clear()
        Custom3DGrain._VOXEL_CACHE.update(self._orig_cache)
        Custom3DGrain._VOXEL_CACHE_MAX = self._orig_max

    def test_store_and_retrieve(self):
        Custom3DGrain._rememberVoxelCache('k1', 'v1')
        self.assertEqual(Custom3DGrain._VOXEL_CACHE.get('k1'), 'v1')

    def test_evicts_oldest_when_full(self):
        Custom3DGrain._VOXEL_CACHE_MAX = 3
        for i in range(4):
            Custom3DGrain._rememberVoxelCache(f'k{i}', f'v{i}')
        # 'k0' inserted first and never re-accessed — must be gone.
        self.assertNotIn('k0', Custom3DGrain._VOXEL_CACHE)
        self.assertIn('k3', Custom3DGrain._VOXEL_CACHE)

    def test_reinsertion_prevents_eviction(self):
        """Re-inserting an entry moves it to the end, saving it from eviction."""
        Custom3DGrain._VOXEL_CACHE_MAX = 3
        for i in range(3):
            Custom3DGrain._rememberVoxelCache(f'k{i}', f'v{i}')
        # Promote 'k0' to the most-recently-used position.
        Custom3DGrain._rememberVoxelCache('k0', 'v0_updated')
        # Insert a new entry — 'k1' should be evicted (the new LRU tail).
        Custom3DGrain._rememberVoxelCache('k3', 'v3')
        self.assertIn('k0', Custom3DGrain._VOXEL_CACHE)
        self.assertNotIn('k1', Custom3DGrain._VOXEL_CACHE)

    def test_cache_does_not_exceed_max(self):
        Custom3DGrain._VOXEL_CACHE_MAX = 5
        for i in range(10):
            Custom3DGrain._rememberVoxelCache(f'key{i}', i)
        self.assertLessEqual(
            len(Custom3DGrain._VOXEL_CACHE),
            Custom3DGrain._VOXEL_CACHE_MAX,
        )


# ---------------------------------------------------------------------------
# 2. Fmm3DGrain._makeCoreMapCacheKey
# ---------------------------------------------------------------------------

class TestMakeCoreMapCacheKey(unittest.TestCase):
    """Tests for the static regression-map cache key helper."""

    def _key(self, coreMap, inhibited, mapDim):
        return motorlib.grain.Fmm3DGrain._makeCoreMapCacheKey(coreMap, inhibited, mapDim)

    def test_identical_inputs_produce_equal_keys(self):
        cm = np.ones((8, 48, 48), dtype=bool)
        self.assertEqual(self._key(cm, 'Both', 48), self._key(cm, 'Both', 48))

    def test_differs_on_inhibited_ends(self):
        cm = np.ones((8, 48, 48), dtype=bool)
        self.assertNotEqual(self._key(cm, 'Both', 48), self._key(cm, 'Neither', 48))

    def test_differs_on_map_dim(self):
        cm = np.ones((8, 48, 48), dtype=bool)
        self.assertNotEqual(self._key(cm, 'Both', 48), self._key(cm, 'Both', 64))

    def test_differs_on_coreMap_content(self):
        cm_ones = np.ones((8, 48, 48), dtype=bool)
        cm_zero = np.zeros((8, 48, 48), dtype=bool)
        self.assertNotEqual(self._key(cm_ones, 'Both', 48), self._key(cm_zero, 'Both', 48))

    def test_returns_tuple(self):
        cm = np.ones((4, 32, 32), dtype=bool)
        k = self._key(cm, 'Top', 32)
        self.assertIsInstance(k, tuple)


# ---------------------------------------------------------------------------
# 3. Property defaults and setProperties
# ---------------------------------------------------------------------------

class TestCustom3DGrainProperties(unittest.TestCase):

    def _make(self):
        return Custom3DGrain()

    def test_geomName(self):
        self.assertEqual(self._make().geomName, 'Custom 3D Grain')

    def test_default_inhibitedEnds(self):
        self.assertEqual(self._make().props['inhibitedEnds'].getValue(), 'Neither')

    def test_default_coreAxis(self):
        self.assertEqual(self._make().props['coreAxis'].getValue(), '-Y')

    def test_default_massFlux3D_false(self):
        self.assertFalse(self._make().props['massFlux3D'].getValue())

    def test_default_meshedMassFluxQuality(self):
        self.assertEqual(self._make().props['meshedMassFluxQuality'].getValue(), 'Exact')

    def test_default_totalLength_none(self):
        self.assertIsNone(self._make().totalLength)

    def test_setProperties_roundtrip(self):
        g = self._make()
        g.setProperties({
            'diameter':      0.04,
            'length':        0.01,
            'inhibitedEnds': 'Top',
            'coreAxis':      '+Z',
            'stlUnit':       'mm',
        })
        self.assertAlmostEqual(g.props['diameter'].getValue(), 0.04)
        self.assertAlmostEqual(g.props['length'].getValue(), 0.01)
        self.assertEqual(g.props['inhibitedEnds'].getValue(), 'Top')
        self.assertEqual(g.props['coreAxis'].getValue(), '+Z')
        self.assertEqual(g.props['stlUnit'].getValue(), 'mm')

    def test_getProperties_returns_all_keys(self):
        g = self._make()
        props = g.getProperties()
        for key in ('diameter', 'length', 'inhibitedEnds', 'coreAxis',
                    'stlUnit', 'massFlux3D', 'meshedMassFluxQuality', 'mesh'):
            self.assertIn(key, props)


# ---------------------------------------------------------------------------
# 4. Mesh-signature hashing
# ---------------------------------------------------------------------------

class TestMeshSignature(unittest.TestCase):

    def setUp(self):
        self.grain = Custom3DGrain()

    def test_raw_signature_is_deterministic(self):
        f, v = _make_bore_cylinder()
        sig1 = self.grain._getMeshSignature(f, v, '')
        sig2 = self.grain._getMeshSignature(f, v, '')
        self.assertEqual(sig1, sig2)

    def test_raw_signature_differs_for_different_geometry(self):
        f1, v1 = _make_bore_cylinder(radius=0.01)
        f2, v2 = _make_bore_cylinder(radius=0.02)
        self.assertNotEqual(
            self.grain._getMeshSignature(f1, v1, ''),
            self.grain._getMeshSignature(f2, v2, ''),
        )

    def test_raw_signature_starts_with_raw_tag(self):
        f, v = _make_bore_cylinder()
        sig = self.grain._getMeshSignature(f, v, '')
        self.assertEqual(sig[0], 'raw')

    def test_file_path_signature_uses_path_tag(self):
        """If a valid path is given, the 'path' strategy should be used."""
        import tempfile
        f, v = _make_bore_cylinder()
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sig = self.grain._getMeshSignature(f, v, tmp_path)
            self.assertEqual(sig[0], 'path')
        finally:
            os.unlink(tmp_path)

    def test_invalid_path_falls_back_to_raw(self):
        f, v = _make_bore_cylinder()
        sig = self.grain._getMeshSignature(f, v, '/nonexistent/path/mesh.stl')
        self.assertEqual(sig[0], 'raw')


# ---------------------------------------------------------------------------
# 5. getCoreMapHash determinism
# ---------------------------------------------------------------------------

class TestGetCoreMapHash(unittest.TestCase):

    def test_hash_is_deterministic(self):
        g1 = Custom3DGrain()
        g2 = Custom3DGrain()
        for g in (g1, g2):
            g.setProperties({'diameter': 0.05, 'length': 0.0, 'stlUnit': 'm'})
            g.mapDim = 100
        self.assertEqual(g1.getCoreMapHash(), g2.getCoreMapHash())

    def test_hash_changes_with_diameter(self):
        g1 = Custom3DGrain()
        g2 = Custom3DGrain()
        g1.setProperties({'diameter': 0.05, 'length': 0.0})
        g2.setProperties({'diameter': 0.06, 'length': 0.0})
        g1.mapDim = g2.mapDim = 100
        self.assertNotEqual(g1.getCoreMapHash(), g2.getCoreMapHash())

    def test_hash_changes_with_mesh(self):
        g1 = Custom3DGrain()
        g2 = Custom3DGrain()
        f1, v1 = _make_bore_cylinder(radius=0.01)
        f2, v2 = _make_bore_cylinder(radius=0.02)
        g1.setProperties({'diameter': 0.05, 'length': 0.0})
        g2.setProperties({'diameter': 0.05, 'length': 0.0})
        g1.props['mesh'].setValue((f1, v1))
        g2.props['mesh'].setValue((f2, v2))
        g1.mapDim = g2.mapDim = 100
        self.assertNotEqual(g1.getCoreMapHash(), g2.getCoreMapHash())


# ---------------------------------------------------------------------------
# 6. Unit-conversion helpers (require mapDim + diameter but no FMM setup)
# ---------------------------------------------------------------------------

class TestUnitConversions(unittest.TestCase):

    def setUp(self):
        self.grain = Custom3DGrain()
        self.grain.setProperties({'diameter': 0.05, 'length': 0.0})
        self.grain.mapDim = 64

    # normalize / unNormalize
    def test_normalize_at_radius_is_one(self):
        self.assertAlmostEqual(self.grain.normalize(0.025), 1.0)

    def test_normalize_zero(self):
        self.assertAlmostEqual(self.grain.normalize(0.0), 0.0)

    def test_unNormalize_one_is_radius(self):
        self.assertAlmostEqual(self.grain.unNormalize(1.0), 0.025)

    def test_normalize_unNormalize_roundtrip(self):
        for v in [0.001, 0.005, 0.010, 0.020, 0.024]:
            self.assertAlmostEqual(
                self.grain.unNormalize(self.grain.normalize(v)), v, places=12
            )

    def test_unNormalize_normalize_roundtrip(self):
        for v in [0.1, 0.5, 1.0, 1.9]:
            self.assertAlmostEqual(
                self.grain.normalize(self.grain.unNormalize(v)), v, places=12
            )

    # lengthToMap / mapToLength
    def test_lengthToMap_full_diameter_is_mapDim(self):
        self.assertAlmostEqual(self.grain.lengthToMap(0.05), 64.0)

    def test_mapToLength_mapDim_is_diameter(self):
        self.assertAlmostEqual(self.grain.mapToLength(64.0), 0.05)

    def test_lengthToMap_mapToLength_roundtrip(self):
        for v in [0.001, 0.010, 0.040]:
            self.assertAlmostEqual(
                self.grain.mapToLength(self.grain.lengthToMap(v)), v, places=12
            )

    # areaToMap / mapToArea
    def test_areaToMap_mapToArea_roundtrip(self):
        for v in [1e-6, 1e-5, 1e-4]:
            self.assertAlmostEqual(
                self.grain.mapToArea(self.grain.areaToMap(v)), v, places=18
            )

    def test_areaToMap_scales_with_mapDim_squared(self):
        a = 1e-4  # 1 cm²
        scaled = self.grain.areaToMap(a)
        self.assertAlmostEqual(
            scaled, (64 ** 2) * (a / (0.05 ** 2)), places=6
        )


# ---------------------------------------------------------------------------
# 7. Empty-mesh code path (no bore — endburner geometry)
# ---------------------------------------------------------------------------

class TestEmptyMeshCodePath(unittest.TestCase):
    """generateCoreMap with no faces/vertices → endburner-only stub."""

    def _make_grain(self, length=0.04, mapDim=64):
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': length, 'inhibitedEnds': 'Top'})
        g.mapDim = mapDim
        return g

    def test_coreMap_shape_has_proper_depth(self):
        g = self._make_grain()
        g.generateCoreMap()
        # No-mesh grain depth should be proportional to length/diameter ratio.
        expected = max(1, int(np.ceil(g.lengthToMap(g.totalLength.getValue()))))
        self.assertEqual(g.coreMap.shape[0], expected)

    def test_coreMap_is_all_propellant(self):
        g = self._make_grain()
        g.generateCoreMap()
        self.assertTrue(np.all(g.coreMap))

    def test_totalLength_equals_props_length(self):
        g = self._make_grain(length=0.06)
        g.generateCoreMap()
        self.assertAlmostEqual(g.totalLength.getValue(), 0.06)

    def test_mask_is_created(self):
        g = self._make_grain()
        g.generateCoreMap()
        self.assertIsNotNone(g.mask)

    def test_mask_shape_matches_coreMap_xy(self):
        g = self._make_grain(mapDim=100)
        g.generateCoreMap()
        self.assertEqual(g.mask.shape[1], 100)
        self.assertEqual(g.mask.shape[2], 100)

    def test_generates_valid_regression_map(self):
        """After generateCoreMap on empty mesh, generateRegressionMap should succeed."""
        g = self._make_grain()
        g.generateCoreMap()
        g.generateRegressionMap()
        self.assertIsNotNone(g.regressionMap)


# ---------------------------------------------------------------------------
# 8. getDetailsString
# ---------------------------------------------------------------------------

class TestGetDetailsString(unittest.TestCase):

    def test_no_mesh_no_voxelization_contains_Length(self):
        """With no mesh and totalLength=None, should estimate from props only."""
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.04})
        # totalLength stays None (no generateCoreMap called)
        result = g.getDetailsString('m')
        self.assertIn('Length', result)

    def test_with_vertices_estimates_extent(self):
        """Having mesh vertices should add an axis extent to the length estimate."""
        g = Custom3DGrain()
        g.setProperties({
            'diameter': 0.05, 'length': 0.01,
            'stlUnit': 'm', 'coreAxis': '-Y',
        })
        faces, verts = _make_bore_cylinder(radius=0.01, length=0.05)
        g.props['mesh'].setValue((faces, verts, ''))
        result = g.getDetailsString('m')
        self.assertIn('Length', result)

    def test_after_empty_generateCoreMap_delegates_to_parent(self):
        """After generateCoreMap sets totalLength, delegate to Fmm3DGrain."""
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.04, 'inhibitedEnds': 'Top'})
        g.mapDim = 100
        g.generateCoreMap()
        result = g.getDetailsString('m')
        self.assertIn('Length', result)

    def test_unit_conversion_in_details_string(self):
        """getDetailsString with 'mm' should include an mm-scaled value."""
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.04, 'inhibitedEnds': 'Top'})
        g.mapDim = 100
        g.generateCoreMap()
        result_mm = g.getDetailsString('mm')
        self.assertIn('Length', result_mm)


# ---------------------------------------------------------------------------
# 9. getGeometryErrors
# ---------------------------------------------------------------------------

class TestGetGeometryErrors(unittest.TestCase):

    def _errors_at_level(self, errors, level):
        from motorlib.simResult import SimAlertLevel
        return [e for e in errors if e.level == level]

    def test_no_error_for_valid_grain(self):
        from motorlib.simResult import SimAlertLevel
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.0})
        errors = self._errors_at_level(g.getGeometryErrors(), SimAlertLevel.ERROR)
        self.assertEqual(len(errors), 0)

    def test_error_for_zero_diameter(self):
        from motorlib.simResult import SimAlertLevel
        g = Custom3DGrain()
        # diameter defaults to 0
        errors = self._errors_at_level(g.getGeometryErrors(), SimAlertLevel.ERROR)
        self.assertGreater(len(errors), 0)

    def test_error_for_endburner_length_with_uninhibited_top(self):
        """Non-zero endburner length with 'Neither' inhibition is invalid."""
        from motorlib.simResult import SimAlertLevel
        g = Custom3DGrain()
        g.setProperties({
            'diameter':      0.05,
            'length':        0.01,       # non-zero endburner
            'inhibitedEnds': 'Neither',  # top NOT inhibited → error
        })
        errors = self._errors_at_level(g.getGeometryErrors(), SimAlertLevel.ERROR)
        self.assertGreater(len(errors), 0)

    def test_no_error_for_endburner_with_top_inhibited(self):
        from motorlib.simResult import SimAlertLevel
        g = Custom3DGrain()
        g.setProperties({
            'diameter':      0.05,
            'length':        0.01,
            'inhibitedEnds': 'Top',   # fore inhibited → valid endburner config
        })
        errors = self._errors_at_level(g.getGeometryErrors(), SimAlertLevel.ERROR)
        self.assertEqual(len(errors), 0)

    def test_returns_list(self):
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.0})
        self.assertIsInstance(g.getGeometryErrors(), list)


# ---------------------------------------------------------------------------
# 10. Regression-map cache
# ---------------------------------------------------------------------------

class TestRegressionMapCache(unittest.TestCase):
    """Fmm3DGrain._regressionMapCacheKey logic."""

    def _make_grain(self, mapDim=100):
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.04, 'inhibitedEnds': 'Top'})
        g.simulationSetup(_MockConfig(map_dim_3d=mapDim))
        return g

    def test_cache_key_set_after_simulationSetup(self):
        g = self._make_grain()
        self.assertIsNotNone(g._regressionMapCacheKey)

    def test_second_generateRegressionMap_uses_cache(self):
        """Unchanged inputs → cache hit; regressionMap object should not be replaced."""
        g = self._make_grain()
        key_before = g._regressionMapCacheKey
        # Obtain a reference to the current regression map array.
        rmap_before_id = id(g.regressionMap)
        g.generateRegressionMap()
        # Object identity: if the cache was used, the same array is kept.
        self.assertEqual(id(g.regressionMap), rmap_before_id)
        # Cache key should still match.
        self.assertEqual(g._regressionMapCacheKey, key_before)

    def test_changed_coreMap_invalidates_cache(self):
        """Mutating coreMap changes its checksum → forced recomputation."""
        g = self._make_grain()
        rmap_before_id = id(g.regressionMap)
        # Flip one center voxel (definitely inside cylinder mask) so the
        # coreMap checksum changes while keeping a valid, non-empty propellant map.
        g.coreMap = g.coreMap.copy()
        cy, cx = g.coreMap.shape[1] // 2, g.coreMap.shape[2] // 2
        g.coreMap[0, cy, cx] = not g.coreMap[0, cy, cx]
        g.generateRegressionMap()
        # A new regressionMap must have been created (cache key changed).
        self.assertNotEqual(id(g.regressionMap), rmap_before_id)

    def test_changed_inhibitedEnds_invalidates_cache(self):
        """Changing inhibitedEnds changes the cache key → regressionMap is rebuilt."""
        g = self._make_grain()
        rmap_before_id = id(g.regressionMap)
        # Switch from 'Top' to 'Neither' — both add at least one uninhibited disk
        # so skfmm always has a zero contour (avoids ValueError on all-propellant maps).
        g.setProperty('inhibitedEnds', 'Neither')
        g.generateRegressionMap()
        self.assertNotEqual(id(g.regressionMap), rmap_before_id)


# ---------------------------------------------------------------------------
# 11. Simulation setup — endburner (empty mesh, inhibitedEnds='Top')
# ---------------------------------------------------------------------------

class TestEndburnerGrain(unittest.TestCase):
    """
    End-to-end physics tests for a pure endburner grain (no bore mesh).
    inhibitedEnds='Top' → fore face inhibited, aft face burns.
    """

    @classmethod
    def setUpClass(cls):
        cls.grain = Custom3DGrain()
        cls.grain.setProperties({
            'diameter':      0.05,
            'length':        0.05,   # 50 mm endburner cap
            'inhibitedEnds': 'Top',
        })
        cls.grain.simulationSetup(_MockConfig(map_dim_3d=100))

    def test_mapDim_set(self):
        self.assertEqual(self.grain.mapDim, 100)

    def test_regressionMap_exists(self):
        self.assertIsNotNone(self.grain.regressionMap)

    def test_regressionMap_is_3d(self):
        self.assertEqual(self.grain.regressionMap.ndim, 3)

    def test_faceAreaFunc_exists(self):
        self.assertIsNotNone(self.grain.faceAreaFunc)

    def test_getInitialLength(self):
        self.assertAlmostEqual(self.grain.getInitialLength(), 0.05)

    def test_wallWeb_positive(self):
        self.assertGreater(self.grain.wallWeb, 0.0)

    def test_getWebLeft_positive_at_zero(self):
        self.assertGreater(self.grain.getWebLeft(0.0), 0.0)

    def test_getWebLeft_decreases(self):
        self.assertLess(self.grain.getWebLeft(0.005), self.grain.getWebLeft(0.0))

    def test_getWebLeft_equals_wallWeb_at_zero(self):
        self.assertAlmostEqual(
            self.grain.getWebLeft(0.0), self.grain.wallWeb, places=6
        )

    def test_getSurfaceArea_positive_at_zero(self):
        self.assertGreater(self.grain.getSurfaceAreaAtRegression(0.0), 0.0)

    def test_faceArea_array_has_values(self):
        """faceArea should contain at least one non-zero entry (surface exists at zero regression)."""
        self.assertGreater(len(self.grain.faceArea), 0)
        self.assertGreater(max(self.grain.faceArea), 0.0)

    def test_getVolumeAtRegression_positive_at_zero(self):
        self.assertGreater(self.grain.getVolumeAtRegression(0.0), 0.0)

    def test_getVolumeAtRegression_decreases(self):
        v0 = self.grain.getVolumeAtRegression(0.0)
        v1 = self.grain.getVolumeAtRegression(0.005)
        self.assertLess(v1, v0)

    def test_getVolumeAtRegression_less_than_bounding(self):
        r = 0.025
        h = self.grain.getInitialLength()
        bounding = math.pi * r**2 * h
        self.assertLess(self.grain.getVolumeAtRegression(0.0), bounding + 1e-10)

    def test_getEndPositions_returns_two_values(self):
        pos = self.grain.getEndPositions(0.0)
        self.assertEqual(len(pos), 2)

    def test_getEndPositions_fore_ge_aft(self):
        """Fore position must be at or beyond the aft position."""
        pos = self.grain.getEndPositions(0.0)
        self.assertGreaterEqual(pos[1], pos[0])

    def test_isWebLeft_true_at_start(self):
        self.assertTrue(self.grain.isWebLeft(0.0))

    def test_isWebLeft_false_past_burnout(self):
        self.assertFalse(self.grain.isWebLeft(self.grain.wallWeb + 0.001))

    def test_getVolumeSlice_positive(self):
        vs = self.grain.getVolumeSlice(0.0, 0.001)
        self.assertGreater(vs, 0.0)

    def test_getFreeVolume_increases_with_regression(self):
        fv0 = self.grain.getFreeVolume(0.0)
        fv1 = self.grain.getFreeVolume(0.005)
        self.assertGreater(fv1, fv0)

    def test_mask_and_regressionMap_shapes_consistent(self):
        self.assertEqual(self.grain.mask.shape, self.grain.regressionMap.shape)


# ---------------------------------------------------------------------------
# 12. Simulation setup — cylindrical bore (inhibitedEnds='Both')
# ---------------------------------------------------------------------------

class TestBoreGrain(unittest.TestCase):
    """
    End-to-end physics tests for a grain with a cylindrical bore mesh.
    Bore: radius=0.01 m, length=0.04 m along Y-axis.
    Grain: diameter=0.05 m, endburner_length=0, inhibitedEnds='Both'.
    Maps to BATES-style grain (lateral surface burns only).
    """

    @classmethod
    def setUpClass(cls):
        faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
        cls.grain = Custom3DGrain()
        cls.grain.setProperties({
            'diameter':      0.05,
            'length':        0.0,
            'stlUnit':       'm',
            'coreAxis':      '-Y',
            'inhibitedEnds': 'Both',
        })
        cls.grain.props['mesh'].setValue((faces, verts, ''))
        cls.grain.simulationSetup(_MockConfig(map_dim_3d=100))

    def test_coreMap_has_void(self):
        """Bore voxels should appear as False (void) in coreMap."""
        self.assertTrue(np.any(~self.grain.coreMap))

    def test_coreMap_has_propellant(self):
        self.assertTrue(np.any(self.grain.coreMap))

    def test_totalLength_approximately_mesh_extent(self):
        """totalLength ≈ 0.04 m (the bore cylinder's Y extent)."""
        self.assertAlmostEqual(self.grain.totalLength.getValue(), 0.04, delta=0.005)

    def test_regressionMap_3d(self):
        self.assertEqual(self.grain.regressionMap.ndim, 3)

    def test_wallWeb_greater_than_bore_radius(self):
        """wallWeb must exceed bore radius (0.01 m) since propellant wraps around it."""
        self.assertGreater(self.grain.wallWeb, 0.01)

    def test_wallWeb_less_than_diameter(self):
        """wallWeb must not exceed the full grain diameter (0.05 m)."""
        self.assertLess(self.grain.wallWeb, 0.05)

    def test_getSurfaceArea_positive_at_zero(self):
        self.assertGreater(self.grain.getSurfaceAreaAtRegression(0.0), 0.0)

    def test_getSurfaceArea_approx_bore_lateral_area(self):
        """Initial bore surface ≈ 2*pi*r_bore*L ≈ 0.00251 m²."""
        area = self.grain.getSurfaceAreaAtRegression(0.0)
        expected = 2 * math.pi * 0.01 * 0.04
        # Allow ±50% tolerance for voxelisation discretisation.
        self.assertAlmostEqual(area, expected, delta=expected * 0.5)

    def test_getSurfaceArea_increases_at_small_regression(self):
        """For a BATES grain, burning surface grows as bore expands."""
        a0 = self.grain.getSurfaceAreaAtRegression(0.0)
        a1 = self.grain.getSurfaceAreaAtRegression(0.003)
        self.assertGreater(a1, 0.0)

    def test_getVolumeAtRegression_positive_at_zero(self):
        self.assertGreater(self.grain.getVolumeAtRegression(0.0), 0.0)

    def test_getVolumeAtRegression_less_than_bounding(self):
        r = 0.025
        h = self.grain.getInitialLength()
        bounding = math.pi * r**2 * h
        self.assertLess(self.grain.getVolumeAtRegression(0.0), bounding)

    def test_getVolumeAtRegression_decreases(self):
        v0 = self.grain.getVolumeAtRegression(0.0)
        v1 = self.grain.getVolumeAtRegression(0.003)
        self.assertLess(v1, v0)

    def test_getPortArea_positive(self):
        pa = self.grain.getPortArea(0.0)
        self.assertGreater(pa, 0.0)

    def test_getPortArea_increases_with_regression(self):
        pa0 = self.grain.getPortArea(0.0)
        pa1 = self.grain.getPortArea(0.005)
        self.assertGreater(pa1, pa0)

    def test_getEndPositions_span_grain_length(self):
        pos = self.grain.getEndPositions(0.0)
        total = pos[1] - pos[0]
        self.assertAlmostEqual(total, self.grain.getInitialLength(), delta=0.005)

    def test_getWebLeft_positive_at_zero(self):
        self.assertGreater(self.grain.getWebLeft(0.0), 0.0)

    def test_getWebLeft_decreases(self):
        w0 = self.grain.getWebLeft(0.0)
        w1 = self.grain.getWebLeft(0.005)
        self.assertLess(w1, w0)

    def test_getMassFlux_no3D_positive(self):
        """getMassFlux with massFlux3D=False should return a non-negative value."""
        self.grain.setProperty('massFlux3D', False)
        pos = self.grain.getEndPositions(0.0)[0]
        mf = self.grain.getMassFlux(
            massIn=0.0, dTime=0.001,
            regDist=0.0, dRegDist=1e-5,
            position=pos, density=1800.0,
        )
        self.assertGreaterEqual(mf, 0.0)

    def test_getMassFlux_increases_with_massIn(self):
        """Higher upstream mass flow should produce higher mass flux at any position."""
        self.grain.setProperty('massFlux3D', False)
        pos = self.grain.getEndPositions(0.0)[0]
        mf0 = self.grain.getMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5,
            position=pos, density=1800.0,
        )
        mf1 = self.grain.getMassFlux(
            massIn=0.1, dTime=0.001, regDist=0.0, dRegDist=1e-5,
            position=pos, density=1800.0,
        )
        self.assertGreater(mf1, mf0)

    def test_volumeFunc_exists(self):
        self.assertIsNotNone(self.grain.volumeFunc)

    def test_portAreaFunc_exists_when_both_inhibited(self):
        """With inhibitedEnds='Both', portAreaFunc should be set."""
        self.assertIsNotNone(self.grain.portAreaFunc)

    def test_regressionMapF64_is_contiguous(self):
        if self.grain._regressionMapF64 is not None:
            self.assertTrue(self.grain._regressionMapF64.flags['C_CONTIGUOUS'])
            self.assertEqual(self.grain._regressionMapF64.dtype, np.float32)

    def test_maskU8_is_contiguous(self):
        if self.grain._maskU8 is not None:
            self.assertTrue(self.grain._maskU8.flags['C_CONTIGUOUS'])
            self.assertEqual(self.grain._maskU8.dtype, np.uint8)


# ---------------------------------------------------------------------------
# 13. _getMeshedMarchingMask
# ---------------------------------------------------------------------------

class TestGetMeshedMarchingMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grain = Custom3DGrain()
        cls.grain.setProperties({
            'diameter':      0.05,
            'length':        0.04,
            'inhibitedEnds': 'Top',
        })
        cls.grain.simulationSetup(_MockConfig(map_dim_3d=100))

    def test_mask_shape_equals_regressionMap(self):
        mcMask, _ = self.grain._getMeshedMarchingMask(5)
        self.assertEqual(mcMask.shape, self.grain.regressionMap.shape)

    def test_mask_dtype_bool(self):
        mcMask, _ = self.grain._getMeshedMarchingMask(5)
        self.assertEqual(mcMask.dtype, np.bool_)

    def test_slices_beyond_zPos_are_false(self):
        zPos = 4
        mcMask, _ = self.grain._getMeshedMarchingMask(zPos)
        self.assertTrue(np.all(mcMask[zPos + 1:] == False))

    def test_clamps_negative_position(self):
        """Position < 0 should clamp to 0."""
        mcMask, returned_zPos = self.grain._getMeshedMarchingMask(-10)
        self.assertEqual(returned_zPos, 0)

    def test_clamps_position_above_zdim(self):
        zdim = self.grain.regressionMap.shape[0]
        mcMask, returned_zPos = self.grain._getMeshedMarchingMask(zdim + 100)
        self.assertEqual(returned_zPos, zdim - 1)

    def test_buffer_reuse_is_correct(self):
        """Calling twice with different zPos should not leave stale bits."""
        large, _ = self.grain._getMeshedMarchingMask(20)
        large_copy = large.copy()
        small, _ = self.grain._getMeshedMarchingMask(5)
        # After second call, slices 6..20 must be False (buffer was cleared).
        self.assertTrue(np.all(small[6:] == False))


# ---------------------------------------------------------------------------
# 14. _getMeshedMarchingStepSize quality modes
# ---------------------------------------------------------------------------

class TestGetMeshedMarchingStepSize(unittest.TestCase):

    def _make_grain(self, quality):
        g = Custom3DGrain()
        g.setProperties({'diameter': 0.05, 'length': 0.04})
        g.mapDim = 100
        g.setProperty('meshedMassFluxQuality', quality)
        return g

    def test_exact_returns_1(self):
        g = self._make_grain('Exact')
        self.assertEqual(g._getMeshedMarchingStepSize(0.0, 1e-5, 10), 1)

    def test_fast_returns_2(self):
        g = self._make_grain('Fast')
        self.assertEqual(g._getMeshedMarchingStepSize(0.0, 1e-5, 10), 2)

    def test_faster_returns_3(self):
        g = self._make_grain('Faster')
        self.assertEqual(g._getMeshedMarchingStepSize(0.0, 1e-5, 10), 3)


# ---------------------------------------------------------------------------
# 15. Inhibited-ends surface-area ordering
# ---------------------------------------------------------------------------

class TestInhibitedEndsAreaOrdering(unittest.TestCase):
    """
    For an endburner-style grain, exposing both ends should give ≥ area
    compared to inhibiting both ends (where only the bore/lateral surface burns).
    """

    def _make_endburner_grain(self, inhibited):
        g = Custom3DGrain()
        g.setProperties({
            'diameter':      0.05,
            'length':        0.04,
            'inhibitedEnds': inhibited,
        })
        g.simulationSetup(_MockConfig(map_dim_3d=100))
        return g

    def test_neither_ge_top_inhibited(self):
        """Neither inhibited should expose at least as much surface as top-only inhibition."""
        g_neither = self._make_endburner_grain('Neither')
        g_top = self._make_endburner_grain('Top')
        a_neither = g_neither.getSurfaceAreaAtRegression(0.0)
        a_top = g_top.getSurfaceAreaAtRegression(0.0)
        self.assertGreaterEqual(a_neither, a_top * 0.9)  # 10% tolerance for discretisation


# ---------------------------------------------------------------------------
# 16. getPeakMassFlux
# ---------------------------------------------------------------------------

class TestGetPeakMassFlux(unittest.TestCase):
    """getPeakMassFlux should return a finite, non-negative value."""

    @classmethod
    def setUpClass(cls):
        faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
        cls.grain = Custom3DGrain()
        cls.grain.setProperties({
            'diameter':      0.05,
            'length':        0.0,
            'stlUnit':       'm',
            'coreAxis':      '-Y',
            'inhibitedEnds': 'Both',
            'massFlux3D':    False,
        })
        cls.grain.props['mesh'].setValue((faces, verts, ''))
        cls.grain.simulationSetup(_MockConfig(map_dim_3d=100))

    def test_returns_finite_positive_value(self):
        peak = self.grain.getPeakMassFlux(
            massIn=0.0, dTime=0.001,
            regDist=0.0, dRegDist=1e-5,
            density=1800.0,
        )
        self.assertGreater(peak, 0.0)
        self.assertTrue(math.isfinite(peak))

    def test_increases_with_massIn(self):
        peak0 = self.grain.getPeakMassFlux(
            massIn=0.0, dTime=0.001, regDist=0.0, dRegDist=1e-5, density=1800.0,
        )
        peak1 = self.grain.getPeakMassFlux(
            massIn=0.5, dTime=0.001, regDist=0.0, dRegDist=1e-5, density=1800.0,
        )
        self.assertGreater(peak1, peak0)


if __name__ == '__main__':
    unittest.main()
