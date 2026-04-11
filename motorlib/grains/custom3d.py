"""3D Custom Grain submodule"""

import pyvista as pv
import numpy as np
import hashlib
import pickle
import os
from collections import OrderedDict

try:
    from mathlib._voxelize import voxelize_mesh as _voxelize_mesh_cy
    _HAS_VOXELIZE_CY = True
except ImportError:
    _HAS_VOXELIZE_CY = False

from ..grain import Fmm3DGrain
from ..properties import MeshProperty, EnumProperty, FloatProperty
from ..simResult import SimAlert, SimAlertLevel, SimAlertType
from ..units import getAllConversions, convert


def _env_enabled_default_true(name):
    """Return True unless env var explicitly disables the feature."""
    raw = os.environ.get(name, '')
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off')

class custom3d(Fmm3DGrain):
    """Custom grains can have any core shape. They define their geometry using a polygon property, which tracks a list
    of polygons that each consist of a number of points. The polygons are scaled according to user specified units and
    drawn onto the core map."""
    geomName = 'Custom 3D Grain'
    _VOXEL_CACHE = OrderedDict()
    _VOXEL_CACHE_MAX = 24

    def __init__(self):
        super().__init__()
        self.props['mesh'] = MeshProperty('Core geometry')
        self.props['stlUnit'] = EnumProperty('Mesh Unit', getAllConversions('m'))
        # Controls which mesh axis is aligned to motor length (map axis 0).
        # Default '-Y' preserves legacy behavior (including historical length-direction flip).
        self.props['coreAxis'] = EnumProperty('Core Axis', ['-Y', '+Y', '+X', '-X', '+Z', '-Z'])
        self.faces = []
        self.vertices = []
        self._lastVoxelBackend = 'unknown'
        self._lastVoxelFallbackReason = ''

    @staticmethod
    def _applyCoreAxisOrientation(coreArray, axisChoice):
        """Reorient voxel array so selected mesh axis maps to motor length axis."""
        # After voxelization/rot90, array axes map to mesh axes as: [Y, X, Z].
        stageAxisLabels = ['Y', 'X', 'Z']

        sign = axisChoice[0] if len(axisChoice) > 1 else '+'
        axisLetter = axisChoice[-1].upper()

        if axisLetter not in stageAxisLabels:
            return coreArray

        lengthAxis = stageAxisLabels.index(axisLetter)
        perm = [lengthAxis] + [i for i in range(3) if i != lengthAxis]
        oriented = np.transpose(coreArray, axes=perm)

        if sign == '-':
            oriented = np.flip(oriented, axis=0)

        return oriented

    @staticmethod
    def _getLengthExtent(bounds, axisChoice):
        """Return selected mesh-axis extent for total motor-length accounting."""
        axisLetter = axisChoice[-1].upper()
        meshAxisToIndex = {'X': 0, 'Y': 1, 'Z': 2}
        return bounds[meshAxisToIndex.get(axisLetter, 1)]

    @classmethod
    def _rememberVoxelCache(cls, key, value):
        cls._VOXEL_CACHE[key] = value
        cls._VOXEL_CACHE.move_to_end(key)
        if len(cls._VOXEL_CACHE) > cls._VOXEL_CACHE_MAX:
            cls._VOXEL_CACHE.popitem(last=False)  # evict least-recently used

    def _getMeshSignature(self, faces, vertices, sourcePath):
        if sourcePath:
            absPath = os.path.abspath(sourcePath)
            try:
                stat = os.stat(absPath)
                return ('path', absPath, stat.st_size, stat.st_mtime_ns)
            except OSError:
                pass

        facesArray = np.ascontiguousarray(faces)
        vertsArray = np.ascontiguousarray(vertices)
        digest = hashlib.sha256(facesArray.tobytes() + vertsArray.tobytes()).hexdigest()
        return ('raw', digest)

    def hashCoreMapInputs(self):
        mapDim = self.mapDim
        inUnit = self.props['stlUnit'].getValue()
        coreAxis = self.props['coreAxis'].getValue()
        faces = self.props['mesh'].getValue()[0]
        vertices = self.props['mesh'].getValue()[1]
        diameter = self.props['diameter'].getValue()
        length = self.props['length'].getValue()

        byte_string = pickle.dumps([mapDim, inUnit, coreAxis, faces, vertices, diameter, length])
        return hashlib.sha256(byte_string).hexdigest()

    def generateCoreMap(self):
        # newCoreMapHash = self.hashCoreMapInputs()
        # if newCoreMapHash == self.coreMapHash:
        #     return 0
        # else:
        #     self.coreMapHash = newCoreMapHash

        inUnit = self.props['stlUnit'].getValue()
        coreAxis = self.props['coreAxis'].getValue()

        meshValue = self.props['mesh'].getValue()
        self.faces, self.vertices = meshValue[0], meshValue[1]
        sourcePath = meshValue[2] if len(meshValue) > 2 else ''

        if len(self.faces) == 0 or len(self.vertices) == 0:
            self.coreMap = np.ones((1, self.mapDim, self.mapDim), dtype=bool)
            self.mapLength = 1
            _x2d, _y2d = np.meshgrid(
                np.linspace(-1, 1, self.mapDim), np.linspace(-1, 1, self.mapDim), indexing='ij'
            )
            self.mask = np.broadcast_to(
                (_x2d**2 + _y2d**2 > 1)[np.newaxis, :, :],
                (self.mapLength, self.mapDim, self.mapDim),
            )
            self.mapX = self.mapY = self.mapZ = None
            self.totalLength = FloatProperty('Length', 'm', 0, 10)
            self.totalLength.setValue(self.props['length'].getValue())
            return

        voxelDensity = self.props['diameter'].getValue() / self.mapDim
        cacheKey = (
            self._getMeshSignature(self.faces, self.vertices, sourcePath),
            inUnit,
            round(float(voxelDensity), 12),
            coreAxis,
        )

        cached = self._VOXEL_CACHE.get(cacheKey)
        if cached is not None:
            self._VOXEL_CACHE.move_to_end(cacheKey)  # mark as recently used
            coreArrayBase, bounds = cached
            coreArray = coreArrayBase.copy()
            self._lastVoxelBackend = 'cache'
            self._lastVoxelFallbackReason = ''
        else:
            verts_m = np.array(self.vertices, dtype=np.float64) * convert(1, inUnit, 'm')
            faces_i = np.array(self.faces, dtype=np.int32)
            bounds = None
            coreArray = None

            # Default-on for 3DFMM; set OPENMOTOR_EXPERIMENTAL_CY_VOXEL=0/false/no to opt out.
            useCyVoxel = _env_enabled_default_true('OPENMOTOR_EXPERIMENTAL_CY_VOXEL')

            if _HAS_VOXELIZE_CY and useCyVoxel:
                try:
                    # Fast Cython path: parallel ray-casting voxelizer.
                    coreArray, bounds = _voxelize_mesh_cy(verts_m, faces_i, voxelDensity)
                    self._lastVoxelBackend = 'cython'
                    self._lastVoxelFallbackReason = ''
                except Exception as exc:
                    # Keep UI/simulation alive even on malformed meshes; fall back below.
                    self._lastVoxelBackend = 'pyvista-fallback'
                    self._lastVoxelFallbackReason = str(exc)
                    print(f"[WARN] Native voxelizer failed; falling back to PyVista: {exc}")
            else:
                self._lastVoxelBackend = 'pyvista'
                if not _HAS_VOXELIZE_CY:
                    self._lastVoxelFallbackReason = 'mathlib._voxelize_cy unavailable'
                elif not useCyVoxel:
                    self._lastVoxelFallbackReason = 'disabled by OPENMOTOR_EXPERIMENTAL_CY_VOXEL'
                else:
                    self._lastVoxelFallbackReason = 'native path not selected'

            if coreArray is None or bounds is None:
                # Fallback: PyVista.
                mesh = pv.PolyData(self.vertices, np.insert(self.faces, 0, 3, axis=1))
                mesh = mesh.scale(3 * [convert(1, inUnit, 'm')], inplace=False)

                try:
                    # Fast/strict path first for well-formed watertight meshes.
                    voxelized = pv.voxelize_volume(mesh.extract_surface(), density=voxelDensity)
                except Exception:
                    # Relax surface checks for imperfect/partial meshes.
                    voxelized = pv.voxelize_volume(
                        mesh.extract_surface(), density=voxelDensity, check_surface=False
                    )

                try:
                    x, _y, _z = voxelized.meshgrid
                    voxelized = voxelized.cell_data_to_point_data()

                    coreArray = np.array(voxelized.point_data["InsideMesh"]).reshape(x.shape, order='F') > 0.5
                    coreArray = np.rot90(coreArray, axes=(1, 0))
                    coreArray = np.logical_not(coreArray)

                    bounds = np.array(mesh.bounds)
                    bounds = bounds[1::2] - bounds[::2]
                except Exception as exc:
                    # Last-resort fail-safe for invalid core geometry: create an empty core
                    # so geometry validation can report issues without crashing simulation setup.
                    print(f"[WARN] PyVista voxelization failed; using empty core fallback: {exc}")
                    self._lastVoxelBackend = 'empty-fallback'
                    self._lastVoxelFallbackReason = str(exc)
                    coreArray = np.ones((1, self.mapDim, self.mapDim), dtype=bool)
                    if verts_m.size:
                        mins = np.min(verts_m, axis=0)
                        maxs = np.max(verts_m, axis=0)
                        bounds = np.maximum(maxs - mins, 0.0)
                    else:
                        bounds = np.zeros(3, dtype=float)

            self._rememberVoxelCache(cacheKey, (coreArray.copy(), bounds.copy()))

        coreArray = self._applyCoreAxisOrientation(coreArray, coreAxis)

        # The following code adds the endburner, if it exists, on top of the voxelized core and then afterwards 
        # manually pads values onto the core until its dims match the initGeometry coremap dims
        # I really dont like this way of doing this, there is definitely a better solution, also if your
        # voxelized core is an odd width and mapDim is even then it will be slightly off centered

        self.totalLength = FloatProperty('Length', 'm', 0, 10)
        self.totalLength.setValue(self.props['length'].getValue() + self._getLengthExtent(bounds, coreAxis))

        self.mapLength = np.ceil(self.lengthToMap(self.totalLength.getValue())).astype(int)
        # The cylindrical mask is identical for every Z slice, so compute it from a 2D
        # slice and use broadcast_to (zero-copy view) instead of a full 3D meshgrid.
        # For a 36" grain at mapDim=128 this saves ~1.15 GB vs the old np.meshgrid approach.
        _x2d, _y2d = np.meshgrid(
            np.linspace(-1, 1, self.mapDim), np.linspace(-1, 1, self.mapDim), indexing='ij'
        )
        self.mask = np.broadcast_to(
            (_x2d**2 + _y2d**2 > 1)[np.newaxis, :, :],
            (self.mapLength, self.mapDim, self.mapDim),
        )
        self.mapX = self.mapY = self.mapZ = None

        coreBlankShape = np.array((self.mapLength, self.mapDim, self.mapDim))
        coreNegativeShape = np.array(coreArray.shape)

        # Symmetric clip: when the voxelized core is larger than the blank along
        # an axis, trim equal amounts from both ends rather than always from the
        # end.  This keeps odd-size vs even-mapDim mismatches centred instead of
        # being consistently shifted 1 voxel to one side.
        for _i in range(1, 3):   # axis 0 (length) is handled by padding below
            _excess = int(coreArray.shape[_i]) - int(coreBlankShape[_i])
            if _excess > 0:
                _trim_fore = _excess // 2
                _trim_aft  = _excess - _trim_fore
                _sl = [slice(None)] * 3
                _sl[_i] = slice(_trim_fore, coreArray.shape[_i] - _trim_aft if _trim_aft > 0 else None)
                coreArray = coreArray[tuple(_sl)]
        # Clip axis 0 from the end if still oversized (should not normally occur).
        if coreArray.shape[0] > coreBlankShape[0]:
            coreArray = coreArray[:int(coreBlankShape[0])]

        # Recompute shapes after clipping; only pad axes that are now too small.
        coreNegativeShape = np.array(coreArray.shape)
        diff = coreBlankShape - coreNegativeShape
        # Round-up split: extra pixel goes to "before" for consistent centring.
        before = (diff + 1) // 2
        after  = diff - before

        before[0] = 0
        after[0]  = int(coreBlankShape[0]) - int(coreNegativeShape[0])

        for i in range(3):
            if diff[i] <= 0:
                before[i], after[i] = 0, 0

        coreArray = np.pad(
            coreArray,
            pad_width=(
                (int(before[0]), int(after[0])),
                (int(before[1]), int(after[1])),
                (int(before[2]), int(after[2])),
            ),
            mode='constant',
            constant_values=1,
        )

        # Final safety clip along all axes (guards against any remaining mismatch).
        for _i in range(3):
            if coreArray.shape[_i] > coreBlankShape[_i]:
                coreArray = np.take(coreArray, range(int(coreBlankShape[_i])), axis=_i)

        self.coreMap = coreArray

    def getGeometryErrors(self):
        errors = super().getGeometryErrors()

        if not _HAS_VOXELIZE_CY:
            errors.append(
                SimAlert(
                    SimAlertLevel.WARNING,
                    SimAlertType.VALUE,
                    'Native 3D voxelizer unavailable; GUI will use slower PyVista fallback. '
                    'Build extensions with: python setup.py build_ext --inplace',
                )
            )
        elif not _env_enabled_default_true('OPENMOTOR_EXPERIMENTAL_CY_VOXEL'):
            errors.append(
                SimAlert(
                    SimAlertLevel.WARNING,
                    SimAlertType.VALUE,
                    'Native 3D voxelizer disabled by OPENMOTOR_EXPERIMENTAL_CY_VOXEL.',
                )
            )

        return errors

    def getDetailsString(self, lengthUnit='m'):
        """Return grain length text without forcing expensive voxelization/setup."""
        if self.totalLength is not None:
            return super().getDetailsString(lengthUnit)

        totalLengthMeters = float(self.props['length'].getValue())

        meshValue = self.props['mesh'].getValue()
        vertices = meshValue[1] if len(meshValue) > 1 else []
        if len(vertices) > 0:
            verts = np.asarray(vertices)
            axisLetter = self.props['coreAxis'].getValue()[-1].upper()
            axisMap = {'X': 0, 'Y': 1, 'Z': 2}
            axisIndex = axisMap.get(axisLetter, 1)

            axisExtent = float(np.max(verts[:, axisIndex]) - np.min(verts[:, axisIndex]))
            totalLengthMeters += abs(axisExtent) * convert(1, self.props['stlUnit'].getValue(), 'm')

        lengthProp = FloatProperty('Length', 'm', 0, 1e6)
        lengthProp.setValue(totalLengthMeters)
        return 'Length: {}'.format(lengthProp.dispFormat(lengthUnit))