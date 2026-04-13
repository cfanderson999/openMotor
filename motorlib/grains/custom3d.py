"""3D Custom Grain submodule"""

import hashlib
import os
import pickle
from collections import OrderedDict

import numpy as np

try:
    from mathlib._voxelize import voxelize_mesh as _voxelize_mesh_cy
    _hasVoxelizeCy = True
except ImportError:
    _hasVoxelizeCy = False

from ..grain import Fmm3DGrain
from ..properties import MeshProperty, EnumProperty, FloatProperty
from ..simResult import SimAlert, SimAlertLevel, SimAlertType
from ..units import getAllConversions, convert


def _envEnabledDefaultTrue(name):
    """Return True unless env var explicitly disables the feature."""
    raw = os.environ.get(name, '')
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off')

class Custom3DGrain(Fmm3DGrain):
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
    def _getCrossSectionAxes(axisChoice):
        """Return the two mesh-coordinate axis indices that form the cross-section."""
        axisLetter = axisChoice[-1].upper()
        meshAxisToIndex = {'X': 0, 'Y': 1, 'Z': 2}
        lengthIdx = meshAxisToIndex.get(axisLetter, 1)
        return [i for i in range(3) if i != lengthIdx]

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

    def getCoreMapHash(self):
        mapDim = self.mapDim
        inUnit = self.props['stlUnit'].getValue()
        coreAxis = self.props['coreAxis'].getValue()
        faces = self.props['mesh'].getValue()[0]
        vertices = self.props['mesh'].getValue()[1]
        diameter = self.props['diameter'].getValue()
        length = self.props['length'].getValue()

        byteString = pickle.dumps([mapDim, inUnit, coreAxis, faces, vertices, diameter, length])
        return hashlib.sha256(byteString).hexdigest()

    def generateCoreMap(self):
        inUnit = self.props['stlUnit'].getValue()
        coreAxis = self.props['coreAxis'].getValue()

        meshValue = self.props['mesh'].getValue()
        self.faces, self.vertices = meshValue[0], meshValue[1]
        sourcePath = meshValue[2] if len(meshValue) > 2 else ''

        if len(self.faces) == 0 or len(self.vertices) == 0:
            self.totalLength = FloatProperty('Length', 'm', 0, 10)
            self.totalLength.setValue(self.props['length'].getValue())
            self.mapLength = max(1, int(np.ceil(self.lengthToMap(self.totalLength.getValue()))))
            self.coreMap = np.ones((self.mapLength, self.mapDim, self.mapDim), dtype=bool)
            _x2d, _y2d = np.meshgrid(
                np.linspace(-1, 1, self.mapDim), np.linspace(-1, 1, self.mapDim), indexing='ij'
            )
            self.mask = np.broadcast_to(
                (_x2d**2 + _y2d**2 > 1)[np.newaxis, :, :],
                (self.mapLength, self.mapDim, self.mapDim),
            )
            self.mapX = self.mapY = self.mapZ = None
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
            vertsM = np.array(self.vertices, dtype=np.float64) * convert(1, inUnit, 'm')
            facesI = np.array(self.faces, dtype=np.int32)

            # Centre the mesh cross-section on the origin BEFORE voxelization.
            # We use the vertex centroid (mean) rather than the bounding-box
            # midpoint because meshes with odd rotational symmetry (3-fin
            # finocyl, 3-point star) have a bbox whose centre is offset from
            # the bore's rotational axis.  The centroid of an N-fold symmetric
            # vertex set always lies on the symmetry axis.
            _csAxes = self._getCrossSectionAxes(coreAxis)
            for _ax in _csAxes:
                _mid = float(np.mean(vertsM[:, _ax]))
                vertsM[:, _ax] -= _mid

            # Symmetrize cross-section bounds so the voxel grid is centred on
            # coordinate 0 (the bore axis).  After centroid centering the bbox
            # can still be asymmetric (e.g. one fin extends further than the
            # gap between the other two).  "Fence" vertices at ±max_extent
            # force the grid to span a symmetric range without affecting
            # ray-cast results (they are not referenced by any triangle).
            _meshAxisToIdx = {'X': 0, 'Y': 1, 'Z': 2}
            _lengthIdx = _meshAxisToIdx[coreAxis[-1].upper()]
            _lmid = 0.5 * (float(vertsM[:, _lengthIdx].min()) +
                            float(vertsM[:, _lengthIdx].max()))
            _fences = []
            for _ax in _csAxes:
                _ext = max(abs(float(vertsM[:, _ax].min())),
                           abs(float(vertsM[:, _ax].max())))
                for _s in (-1.0, 1.0):
                    _fv = np.zeros(3, dtype=np.float64)
                    _fv[_lengthIdx] = _lmid
                    _fv[_ax] = _s * _ext
                    _fences.append(_fv)
            if _fences:
                vertsM = np.vstack([vertsM, np.array(_fences)])

            bounds = None
            coreArray = None

            if not _hasVoxelizeCy:
                raise ImportError(
                    'mathlib._voxelize_cy is required for 3D grains. '
                    'Build extensions with: python setup.py build_ext --inplace'
                )

            # Default-on for 3DFMM; set OPENMOTOR_EXPERIMENTAL_CY_VOXEL=0/false/no to opt out.
            useCyVoxel = _envEnabledDefaultTrue('OPENMOTOR_EXPERIMENTAL_CY_VOXEL')

            if not useCyVoxel:
                raise RuntimeError(
                    'Native 3D voxelizer disabled by OPENMOTOR_EXPERIMENTAL_CY_VOXEL.'
                )

            try:
                # Fast Cython path: parallel ray-casting voxelizer.
                coreArray, bounds = _voxelize_mesh_cy(vertsM, facesI, voxelDensity)
                self._lastVoxelBackend = 'cython'
                self._lastVoxelFallbackReason = ''
            except Exception as exc:
                # Voxelization failed on a malformed mesh; create an empty core
                # so geometry validation can report issues without crashing simulation setup.
                print(f"[WARN] Native voxelizer failed; using empty core fallback: {exc}")
                self._lastVoxelBackend = 'empty-fallback'
                self._lastVoxelFallbackReason = str(exc)
                coreArray = np.ones((1, self.mapDim, self.mapDim), dtype=bool)
                if vertsM.size:
                    mins = np.min(vertsM, axis=0)
                    maxs = np.max(vertsM, axis=0)
                    bounds = np.maximum(maxs - mins, 0.0)
                else:
                    bounds = np.zeros(3, dtype=float)

            self._rememberVoxelCache(cacheKey, (coreArray.copy(), bounds.copy()))

        coreArray = self._applyCoreAxisOrientation(coreArray, coreAxis)

        # The following code adds the endburner, if it exists, on top of the voxelized core and then afterwards 
        # manually pads values onto the core until its dims match the initGeometry coremap dims

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
                _trimFore = _excess // 2
                _trimAft  = _excess - _trimFore
                _sl = [slice(None)] * 3
                _sl[_i] = slice(_trimFore, coreArray.shape[_i] - _trimAft if _trimAft > 0 else None)
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

        if not _hasVoxelizeCy:
            errors.append(
                SimAlert(
                    SimAlertLevel.ERROR,
                    SimAlertType.VALUE,
                    'Native 3D voxelizer unavailable; 3D grains cannot be simulated. '
                    'Build extensions with: python setup.py build_ext --inplace',
                )
            )
        elif not _envEnabledDefaultTrue('OPENMOTOR_EXPERIMENTAL_CY_VOXEL'):
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
