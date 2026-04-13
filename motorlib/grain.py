"""
This module includes the base classes from which all grain classes should inherit. None of these objects
should be instantiated directly.
"""

from abc import abstractmethod
import hashlib
import os
import pathlib
import struct
import tempfile
from collections import OrderedDict
from typing import Tuple, List, Union

import numpy as np
import skfmm
from scipy import interpolate
from scipy.signal import savgol_filter
from skimage import measure

import mathlib
try:
    from mathlib._fmm3d import (
        get_first_last_prop_indices,
        get_volume_count_gt_threshold,
        get_volume_count_gt_threshold_from_z,
        get_core_area_count_at_slice,
        get_core_area_profile,
        get_massflux_slice_suffix_arrays,
    )
    _hasFmm3dCy = True
except ImportError:
    _hasFmm3dCy = False

try:
    from mathlib._march import marching_area_pixels as _marchingAreaPixelsCy
    _hasMarchCy = True
except ImportError:
    _hasMarchCy = False

from . import geometry
from .properties import EnumProperty, FloatProperty, BooleanProperty, PropertyCollection
from .simResult import SimAlert, SimAlertLevel, SimAlertType
from .constants import maximumRefDiameter, maximumRefLength


def _smoothSeries(values, window: int = 31, poly: int = 5) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with safe parameters for short arrays.

    The default window=31/poly=5 is tuned for dense sweeps (≥100 levels).
    For short arrays (adaptive MC produces as few as 20–30 levels) a fixed
    window of 31 spans the entire series and effectively replaces every point
    with a global polynomial fit — discarding all local shape information.

    Adaptive strategy: scale the window down proportionally so it covers at
    most ~40% of the series, with a minimum of 5 points and a maximum of the
    caller-supplied default.  The polynomial order is kept relative to the
    window (≥3, ≤ window-1) so the filter remains well-conditioned.
    """
    arr = np.asarray(values, dtype=float)
    count = arr.size
    if count < 3:
        return arr

    # Adaptive window: target ~40% of series length, capped at caller default.
    targetWin = max(5, int(round(count * 0.4)))
    win = min(window, targetWin, count)
    # Must be odd for savgol_filter.
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return arr

    # Scale polynomial order proportionally; keep it in [1, win-1].
    adaptivePoly = max(1, min(poly, int(round(poly * win / window)), win - 1))

    return savgol_filter(arr, win, adaptivePoly)


def _marchingAreaPixels(regressionMap: np.ndarray, validMask: np.ndarray, level: float):
    """Return surface area in map-space pixels for a marching-cubes iso-level, or None on failure."""
    try:
        verts, faces, _, _ = measure.marching_cubes(
            regressionMap, level=level, mask=validMask
        )
    except (RuntimeError, ValueError):
        return None
    return measure.mesh_surface_area(verts, faces)


if _hasMarchCy:
    _marchingAreaPixels = _marchingAreaPixelsCy


class _GrainSetupCanceled(Exception):
    """Raised when a grain setup operation (voxelization / regression map build)
    is aborted by the user via a cancelCheck callback."""


class Grain(PropertyCollection):
    """
    A basic propellant grain.

    This is the class that all grains inherit from. It provides a few properties and
    composed methods but otherwise it is up to the subclass to make a functional grain.
    """

    geomName: Union[str, None] = None

    def __init__(self) -> None:
        super().__init__()
        self.props["diameter"] = FloatProperty(
            dispName="Diameter",
            unit="m",
            minValue=0,
            maxValue=maximumRefDiameter,
        )
        self.props["length"] = FloatProperty(
            dispName="Length",
            unit="m",
            minValue=0,
            maxValue=maximumRefLength,
        )

    def getVolumeSlice(self, regDist: float, dRegDist: float) -> float:
        """
        Returns the amount of propellant volume consumed as the grain regresses from a distance of 'regDist' to
        regDist + dRegDist.
        """
        return self.getVolumeAtRegression(regDist) - self.getVolumeAtRegression(
            regDist + dRegDist
        )

    @abstractmethod
    def getSurfaceAreaAtRegression(self, regDist: float) -> float:
        """Returns the surface area of the grain after it has regressed a linear distance of 'regDist'."""

    @abstractmethod
    def getVolumeAtRegression(self, regDist: float) -> float:
        """Returns the volume of propellant in the grain after it has regressed a linear distance 'regDist'."""

    @abstractmethod
    def getWebLeft(self, regDist: float) -> float:
        """Returns the shortest distance the grain has to regress to burn out."""

    def isWebLeft(self, regDist: float, burnoutThres: float = 0.00001) -> bool:
        """Returns True if the grain has propellant left to burn after it has regressed a distance of 'regDist'."""
        return self.getWebLeft(regDist) > burnoutThres

    @abstractmethod
    def getMassFlux(
        self,
        massIn: float,
        dTime: float,
        regDist: float,
        dRegDist: float,
        position: float,
        density: float,
    ) -> float:
        """
        Returns the mass flux at a point along the grain.

        Takes in the mass flow into the grain, a timestep, the distance the grain has regressed so far,
        the additional distance it will regress during the timestep, a position along the grain measured
        from the head end, and the density of the propellant.
        """

    def getPeakMassFlux(
        self,
        massIn: float,
        dTime: float,
        regDist: float,
        dRegDist: float,
        density: float,
    ) -> float:
        """
        Uses the grain's mass flux method to return the max.
        Assumes that it will be at the port of the grain!
        """
        return self.getMassFlux(
            massIn=massIn,
            dTime=dTime,
            regDist=regDist,
            dRegDist=dRegDist,
            position=self.getEndPositions(regDist)[1],
            density=density,
        )

    @abstractmethod
    def getEndPositions(self, regDist: float) -> Tuple[float, float]:
        """Returns the positions of the grain ends relative to the original (unburned) grain top."""

    @abstractmethod
    def getPortArea(self, regDist: float) -> float:
        """Returns the area of the grain's port when it has regressed a distance of 'regDist'."""

    @abstractmethod
    def getInitialLength(self) -> float:
        """Returns the length of the grain before any burning has begun"""

    def getRegressedLength(self, regDist: float) -> float:
        """
        Returns the length of the grain when it has regressed a distance of 'regDist',
        taking any possible inhibition into account.
        """
        endPos = self.getEndPositions(regDist)
        return endPos[1] - endPos[0]

    def getDetailsString(self, lengthUnit: str = "m") -> str:
        """Returns a short string describing the grain, formatted using the units that is passed in."""
        return "Length: {}".format(self.props["length"].dispFormat(lengthUnit))

    @abstractmethod
    def simulationSetup(self, config):
        """Do anything needed to prepare this grain for simulation."""

    def getGeometryErrors(self) -> List[SimAlert]:
        """
        Returns a list of simAlerts that detail any issues with the geometry of the grain.

        Errors should be used for any condition that prevents simulation of the grain,
        while warnings can be used to notify the user of possible non-fatal mistakes in their entered numbers.
        Subclasses should still call the superclass method, as it performs checks that still apply to its subclasses.
        """
        errors = []
        if self.props["diameter"].getValue() == 0:
            errors.append(
                SimAlert(
                    SimAlertLevel.ERROR, SimAlertType.GEOMETRY, "Diameter must not be 0"
                )
            )
        if self.props["length"].getValue() == 0:
            errors.append(
                SimAlert(
                    SimAlertLevel.ERROR, SimAlertType.GEOMETRY, "Length must not be 0"
                )
            )
        return errors

    def getGrainBoundingVolume(self) -> float:
        """Returns the volume of the bounding cylinder around the grain."""
        return geometry.cylinderVolume(
            dia=self.props["diameter"].getValue(),
            height=self.props["length"].getValue(),
        )

    def getFreeVolume(self, regDist: float) -> float:
        """
        Returns the amount of empty (non-propellant) volume in bounding cylinder of the grain for a given regression
        depth.
        """
        return self.getGrainBoundingVolume() - self.getVolumeAtRegression(regDist)


class PerforatedGrain(Grain):
    """
    A grain with a hole of some shape through the center.

    Adds abstract methods related to the core to the basic grain class
    """

    geomName = "perfGrain"

    def __init__(self) -> None:
        super().__init__()
        self.props["inhibitedEnds"] = EnumProperty(
            "Inhibited ends", ["Neither", "Top", "Bottom", "Both"]
        )
        self.wallWeb: float = 0  # Max distance from the core to the wall

    def getEndPositions(self, regDist: float) -> Tuple[float, float]:
        if self.props["inhibitedEnds"].getValue() == "Neither":  # Neither
            return (regDist, self.props["length"].getValue() - regDist)
        if self.props["inhibitedEnds"].getValue() == "Top":  # Top
            return (0, self.props["length"].getValue() - regDist)
        if self.props["inhibitedEnds"].getValue() == "Bottom":  # Bottom
            return (regDist, self.props["length"].getValue())
        if self.props["inhibitedEnds"].getValue() == "Both":
            return (0, self.props["length"].getValue())
        # The enum should prevent this from even being raised, but to cover the case where it somehow gets set wrong
        raise ValueError("Invalid number of faces inhibited")

    @abstractmethod
    def getCorePerimeter(self, regDist: float) -> float:
        """Returns the perimeter of the core after the grain has regressed a distance of 'regDist'."""

    @abstractmethod
    def getFaceArea(self, regDist: float) -> float:
        """
        Returns the area of the grain face after it has regressed a distance of 'regDist'.
        This is the same as the area of an equal-diameter endburning grain minus the grain's port area.
        """

    def getCoreSurfaceArea(self, regDist: float) -> float:
        """Returns the surface area of the grain's core after it has regressed a distance of 'regDist'."""
        corePerimeter = self.getCorePerimeter(regDist)
        coreArea = corePerimeter * self.getRegressedLength(regDist)
        return coreArea

    def getWebLeft(self, regDist: float) -> float:
        wallLeft = self.wallWeb - regDist
        if self.props["inhibitedEnds"].getValue() == "Both":
            return wallLeft
        lengthLeft = self.getRegressedLength(regDist)
        return min(lengthLeft, wallLeft)

    def getSurfaceAreaAtRegression(self, regDist: float) -> float:
        faceArea = self.getFaceArea(regDist)
        coreArea = self.getCoreSurfaceArea(regDist)

        exposedFaces: int = 2
        if (
            self.props["inhibitedEnds"].getValue() == "Top"
            or self.props["inhibitedEnds"].getValue() == "Bottom"
        ):
            exposedFaces = 1
        if self.props["inhibitedEnds"].getValue() == "Both":
            exposedFaces = 0

        return coreArea + (exposedFaces * faceArea)

    def getVolumeAtRegression(self, regDist: float) -> float:
        faceArea = self.getFaceArea(regDist)
        return faceArea * self.getRegressedLength(regDist)

    def getPortArea(self, regDist: float) -> float:
        faceArea = self.getFaceArea(regDist)
        uncored = geometry.circleArea(self.props["diameter"].getValue())
        return uncored - faceArea

    def getInitialLength(self) -> float:
        return self.props["length"].getValue()

    def getMassFlux(
        self,
        massIn: float,
        dTime: float,
        regDist: float,
        dRegDist: float,
        position: float,
        density: float,
    ) -> float:
        diameter = self.props["diameter"].getValue()

        endPos = self.getEndPositions(regDist)
        # If a position above the top face is queried, the mass flow is just the input mass and the
        # diameter is the casting tube
        if position < endPos[0]:
            return massIn / geometry.circleArea(diameter)
        # If a position in the grain is queried, the mass flow is the input mass, from the top face,
        # and from the tube up to the point. The diameter is the core.
        if position <= endPos[1]:
            if self.props["inhibitedEnds"].getValue() in ("Top", "Both"):
                top = 0
                countedCoreLength = position
            else:
                top = self.getFaceArea(regDist + dRegDist) * dRegDist * density
                countedCoreLength = position - (endPos[0] + dRegDist)
            # This block gets the mass of propellant the core burns in the step.
            core = (self.getPortArea(regDist + dRegDist) * countedCoreLength) - (
                self.getPortArea(regDist) * countedCoreLength
            )
            core *= density

            massFlow = massIn + ((top + core) / dTime)
            return massFlow / self.getPortArea(regDist + dRegDist)
        # A position past the grain end was specified, so the mass flow includes the input mass flow
        # and all mass produced by the grain. Diameter is the casting tube.
        massFlow = massIn + (self.getVolumeSlice(regDist, dRegDist) * density / dTime)
        return massFlow / geometry.circleArea(diameter)

    @abstractmethod
    def getFaceImage(self, mapDim: int):
        """Returns an image of the grain's cross section, with resolution (mapDim, mapDim)."""

    @abstractmethod
    def getRegressionData(
        self, mapDim: int, numContours: int = 15, coreBlack: bool = True
    ) -> Tuple:
        """
        Returns a tuple that includes a grain face image as described in 'getFaceImage', a regression map
        where color maps to regression depth, a list of contours (lists of (x,y) points in image space) of
        equal regression depth, and a list of corresponding contour lengths. The contours are equally spaced
        between 0 regression and burnout.
        """


class FmmGrain(PerforatedGrain):
    """
    A grain that uses the fast marching method to calculate its regression. All a subclass has to do is
    provide an implementation of generateCoreMap that makes an image of a cross section of the grain.
    """

    geomName = "fmmGrain"

    def __init__(self) -> None:
        super().__init__()
        self.mapDim: int = 1001
        self.mapX, self.mapY = None, None
        self.mask = None
        self.coreMap = None
        self.regressionMap = None
        self.faceArea = None
        self.faceAreaFunc = None

    def normalize(self, value: float) -> float:
        """
        Transforms real unit quantities into self.mapX, self.mapY coordinates.

        For use in indexing into the coremap.
        """
        return value / (0.5 * self.props["diameter"].getValue())

    def unNormalize(self, value: float) -> float:
        """
        Transforms self.mapX, self.mapY coordinates to real unit quantities.

        Used to determine real lengths in coremap.
        """
        return (value / 2) * self.props["diameter"].getValue()

    def lengthToMap(self, value: float) -> float:
        """
        Converts meters to pixels.

        Used to compare real distances to pixel distances in the regression map.
        """
        return self.mapDim * (value / self.props["diameter"].getValue())

    def mapToLength(self, value: float) -> float:
        """
        Converts pixels to meters.

        Used to extract real distances from pixel distances such as contour lengths.
        """
        return self.props["diameter"].getValue() * (value / self.mapDim)

    def areaToMap(self, value: float) -> float:
        """Used to convert sqm to sq pixels, like on the regression map."""
        return (self.mapDim**2) * (value / (self.props["diameter"].getValue() ** 2))

    def mapToArea(self, value: float) -> float:
        """
        Used to convert sq pixels to sqm.

        For extracting real areas from the regression map.
        """
        return (self.props["diameter"].getValue() ** 2) * (value / (self.mapDim**2))

    def initGeometry(self, mapDim: int) -> None:
        """
        Set up an empty core map and reset the regression map.

        Takes in the dimension of both maps.
        """
        if mapDim < 64:  # TODO convert int value into meaningful constant
            raise ValueError("Map dimension must be 64 or larger to get good results")
        self.mapDim = mapDim
        self.mapX, self.mapY = np.meshgrid(
            np.linspace(-1, 1, self.mapDim), np.linspace(-1, 1, self.mapDim)
        )
        self.mask = self.mapX**2 + self.mapY**2 > 1
        self.coreMap = np.ones_like(self.mapX)
        self.regressionMap = None

    @abstractmethod
    def generateCoreMap(self) -> None:
        """
        Use self.mapX and self.mapY to generate an image of the grain cross section in self.coreMap.
        A 0 in the image means propellant, and a 1 means no propellant.
        """

    def simulationSetup(self, config) -> None:
        mapSize = config.getProperty("mapDim")

        self.initGeometry(mapSize)
        self.generateCoreMap()
        self.generateRegressionMap()

    def generateRegressionMap(self) -> None:
        """
        Uses the fast marching method to generate an image of how the grain regresses from the core map.

        The map is stored under self.regressionMap.
        """
        masked = np.ma.MaskedArray(self.coreMap, self.mask)
        cellSize = 1 / self.mapDim
        self.regressionMap = skfmm.distance(masked, dx=cellSize) * 2
        maxDist = np.amax(self.regressionMap)
        self.wallWeb = self.unNormalize(maxDist)
        faceArea = []
        polled = []
        valid = np.logical_not(self.mask)
        for i in range(int(maxDist * self.mapDim) + 2):
            polled.append(i / self.mapDim)
            faceArea.append(
                self.mapToArea(
                    np.count_nonzero(
                        np.logical_and(self.regressionMap > (i / self.mapDim), valid)
                    )
                )
            )
        self.faceArea = _smoothSeries(faceArea)
        self.faceAreaFunc = interpolate.interp1d(polled, self.faceArea)

    def getCorePerimeter(self, regDist: float) -> float:
        mapDist = self.normalize(regDist)
        return self.mapToLength(mathlib.find_perimeter(self.regressionMap, mapDist)[0])

    def getFaceArea(self, regDist: float):
        mapDist = self.normalize(regDist)
        index = int(mapDist * self.mapDim)
        if index >= len(self.faceArea) - 1:
            return 0  # Past burnout
        if not self.faceAreaFunc:
            raise ValueError("faceAreaFunc is missing")
        return self.faceAreaFunc(mapDist)

    def getFaceImage(self, mapDim: int) -> np.ma.MaskedArray:
        self.initGeometry(mapDim)
        self.generateCoreMap()
        return np.ma.MaskedArray(self.coreMap, self.mask)

    def getRegressionData(
        self, mapDim: int, numContours: int = 15, coreBlack: bool = True
    ) -> Tuple:
        self.initGeometry(mapDim)
        self.generateCoreMap()

        masked = np.ma.MaskedArray(self.coreMap, self.mask)
        regressionMap = None
        contours = []
        contourLengths = {}

        try:
            self.generateRegressionMap()

            regmax = np.amax(self.regressionMap)

            regressionMap = self.regressionMap[:, :].copy()
            if coreBlack:
                regressionMap[np.where(self.coreMap == 0)] = (
                    regmax  # Make the core black
                )
            regressionMap = np.ma.MaskedArray(regressionMap, self.mask)

            for dist in np.linspace(0, regmax, numContours):
                contours.append([])
                contourLengths[dist] = 0
                layerContours = mathlib.find_perimeter(
                    self.regressionMap,
                    dist,
                    fully_connected="low",
                    including_contours=True,
                )[1]
                for contour in layerContours:
                    contours[-1].append(geometry.clean(contour, self.mapDim, 3))
                    contourLengths[dist] += geometry.length(contour, self.mapDim)

        except ValueError as exc:  # If there aren't any contours, do nothing
            print(exc)

        return (masked, regressionMap, contours, contourLengths)

class Fmm3DGrain(Grain):
    """A grain that uses a 3D version of the fast marching method to calculate its regression. All a subclass has to do
    is provide an implementation of generateCoreMap that makes a 3D model of the grain and core."""
    geomName = None

    # ---- FMM distance-field disk cache -----------------------------------
    # skfmm.distance() on a 128³ grid takes 2–10 s; a disk cache eliminates
    # the cost on repeat runs (app restart, parameter sweep with fixed geometry).
    # Cache files are stored as .npy under the system temp directory.
    _FMM_DIST_CACHE_VERSION = '2'   # bump to invalidate all cached entries
    _FMM_DIST_CACHE_DIR: 'pathlib.Path | None' = None

    @classmethod
    def _getFmmCacheDir(cls) -> pathlib.Path:
        if cls._FMM_DIST_CACHE_DIR is None:
            d = pathlib.Path(tempfile.gettempdir()) / 'openmotor_fmm_cache'
            d.mkdir(exist_ok=True)
            cls._FMM_DIST_CACHE_DIR = d
        return cls._FMM_DIST_CACHE_DIR

    @staticmethod
    def _computeFmmCacheKey(coreMapUninhib: np.ndarray, cellSize: float) -> str:
        """SHA-256 fingerprint of the FMM inputs → hex string cache key."""
        # Use buffer protocol directly — avoids a full .tobytes() copy that
        # would double memory usage on large 3-D grids (mapDim=256 → ~66 MB).
        arr = np.ascontiguousarray(coreMapUninhib, dtype=np.uint8)
        h = hashlib.sha256()
        h.update(arr.data)       # memoryview — no copy
        h.update(struct.pack('>d', cellSize))
        return h.hexdigest()

    def _loadFmmCache(self, key: str) -> 'np.ndarray | None':
        path = self._getFmmCacheDir() / f'{key}_v{self._FMM_DIST_CACHE_VERSION}.npy'
        if path.is_file():
            try:
                return np.load(str(path))
            except Exception:
                pass
        return None

    def _saveFmmCache(self, key: str, data: np.ndarray) -> None:
        """Persist FMM result to disk; failures are silently ignored."""
        try:
            path = self._getFmmCacheDir() / f'{key}_v{self._FMM_DIST_CACHE_VERSION}.npy'
            np.save(str(path), data)
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self.mapDim = 64
        self.mapLength = None

        self.mapX, self.mapY, self.mapZ = None, None, None
        self.mask = None
        self._validMask = None
        self._maskU8 = None
        self._regressionMapF64 = None
        self._mcMaskBuf = None
        self.coreMap = None
        self.regressionMap = None
        self._regressionMapCacheKey = None

        self.props['length'] = FloatProperty('Endburner Length', 'm', 0, 10)
        self.props['inhibitedEnds'] = EnumProperty('Inhibited ends', ['Neither', 'Top', 'Bottom', 'Both'])
        self.totalLength = None

        self.props['meshedMassFluxQuality'] = EnumProperty(
            '3D Mass Flux Quality (mesh only)', ['Exact', 'Fast', 'Faster']
        )
        self.props['meshedMassFluxQuality'].setValue('Exact')
        self.props['massFlux3D'] = BooleanProperty('Calculate 3D Mass Flux (slower)')
        self.portAreaFunc = None  # smooth core-area-at-aft-face vs mapDist, set by generateRegressionMap
        self.foreAreaFunc = None  # smooth core-area-at-fore-face vs mapDist, set by generateRegressionMap
        self.volumeFunc = None    # smooth propellant volume vs mapDist, set by generateRegressionMap
        # ---- core_area_profile LRU cache (P1 4.3) ---------------------------
        # get_core_area_profile is O(n²·z) and is called every timestep from
        # _getMeshedPeakSearchCandidates.  The regression map is static after
        # simulationSetup, so the result is fully determined by (mapDist, startPos, endPos).
        # Cache the most recent N calls so that repeated calls at the same regression
        # distance (common in high-step-count sims) are free.
        self._capCache: 'OrderedDict' = OrderedDict()  # (mapDist_q, startPos, endPos) → int64 array
        self._capCacheMax: int = 32

    # ---- Bitpacked coreMap storage ------------------------------------------
    # Large STL grains can yield bool coreMap arrays of 100+ MB.  Bool and uint8
    # both use 1 byte per voxel in NumPy — there is NO memory difference between
    # them.  The real savings come from np.packbits / np.unpackbits which store
    # 8 bools per byte (8× smaller at rest).  Arrays above _PACK_THRESHOLD_BYTES
    # are automatically stored packed and unpacked on demand.
    #
    # The cache-key fast-path in generateRegressionMap reads the precomputed sum
    # directly from the packed descriptor, so repeated calls on an unchanged
    # grain skip the heavyweight unpack entirely.
    #
    # Dense storage is used for arrays below the threshold (cheap to hold and
    # avoids unpack overhead on every access).
    _PACK_THRESHOLD_BYTES = 32 * 1024 * 1024  # 32 MB

    @property
    def coreMap(self):
        """Return the dense bool coreMap, unpacking from bitpacked storage when needed."""
        if self.__dict__.get('_coreMapPacked') is not None:
            shape = self.__dict__['_coreMapShape']
            n = shape[0] * shape[1] * shape[2]
            return (
                np.unpackbits(self.__dict__['_coreMapPacked'], count=n, bitorder='little')
                .view(np.bool_)
                .reshape(shape)
            )
        return self.__dict__.get('_coreMapDense')

    @coreMap.setter
    def coreMap(self, value):
        if value is None:
            self.__dict__['_coreMapDense']  = None
            self.__dict__['_coreMapPacked'] = None
            self.__dict__['_coreMapShape']  = None
            self.__dict__['_coreMapSum']    = None
        elif isinstance(value, np.ndarray) and value.dtype == np.bool_ and value.nbytes > self._PACK_THRESHOLD_BYTES:
            # Bool array above threshold: store as packed bits (8 bools per byte).
            # Precompute the voxel sum for the cache key so generateRegressionMap
            # can check for changes without unpacking the full array first.
            flat = value.ravel().view(np.uint8)
            self.__dict__['_coreMapShape']  = value.shape
            self.__dict__['_coreMapPacked'] = np.packbits(flat, bitorder='little')
            self.__dict__['_coreMapSum']    = int(flat.sum())
            self.__dict__['_coreMapDense']  = None
        else:
            self.__dict__['_coreMapDense']  = value
            self.__dict__['_coreMapPacked'] = None
            self.__dict__['_coreMapShape']  = None
            self.__dict__['_coreMapSum']    = None

    def normalize(self, value):
        """Transforms real unit quantities into self.mapX, self.mapY coordinates. For use in indexing into the
        coremap."""
        return value / (0.5 * self.props['diameter'].getValue())

    def unNormalize(self, value):
        """Transforms self.mapX, self.mapY coordinates to real unit quantities. Used to determine real lengths in
        coremap."""
        return (value / 2) * self.props['diameter'].getValue()

    def lengthToMap(self, value):
        """Converts meters to pixels. Used to compare real distances to pixel distances in the regression map."""
        return self.mapDim * (value / self.props['diameter'].getValue())

    def mapToLength(self, value):
        """Converts pixels to meters. Used to extract real distances from pixel distances such as contour lengths"""
        return self.props['diameter'].getValue() * (value / self.mapDim)

    def areaToMap(self, value):
        """Used to convert sqm to sq pixels, like on the regression map."""
        return (self.mapDim ** 2) * (value / (self.props['diameter'].getValue() ** 2))

    def mapToArea(self, value):
        """Used to convert sq pixels to sqm. For extracting real areas from the regression map."""
        return (self.props['diameter'].getValue() ** 2) * (value / (self.mapDim ** 2))

    def _getMeshedMarchingMask(self, position):
        """Build a marching-cubes mask from integer z-index without large logical chains."""
        zdim = self.regressionMap.shape[0]
        zPos = max(0, min(int(position), zdim - 1))

        if self._validMask is None or self._validMask.shape != self.mask.shape:
            self._validMask = np.logical_not(self.mask)

        if self._mcMaskBuf is None or self._mcMaskBuf.shape != self.mask.shape:
            self._mcMaskBuf = np.zeros(self.mask.shape, dtype=bool)
        else:
            self._mcMaskBuf[:] = False
        # mcMask is an alias into self._mcMaskBuf; callers must not retain the reference
        # past the next call to _getMeshedMarchingMask (the buffer is reused each call).
        mcMask = self._mcMaskBuf
        # Include only valid (inside-cylinder) voxels in slices 0..zPos.
        # Slices zPos+1..end remain False so MC stops at the measurement plane.
        # Previously mcMask[0:zPos+1] was all-True, which incorrectly included
        # outside-cylinder voxels and generated spurious iso-surface patches that
        # caused jagged mass-flux output as regDist advanced.
        mcMask[:zPos + 1] = self._validMask[:zPos + 1]

        return mcMask, zPos

    def _getMeshedMarchingStepSize(self, regDist, dRegDist, zPos):
        quality = self.props['meshedMassFluxQuality'].getValue()
        if quality == 'Fast':
            return 2
        if quality == 'Faster':
            return 3
        return 1

    def _getMeshedPeakSearchCandidates(self, startPos, endPos, massIn, dTime, regDist, dRegDist, density):
        """Return candidate z-positions for peak-flux search in meshed 3D mode.

        NOTE: ``massIn``, ``dTime``, and ``density`` are accepted for API symmetry with
        ``getMassFlux`` but are **not used** by this algorithm.  The run-collapse approach
        is purely geometry-driven (core_area_profile proxy).  For single-grain motors this
        is exact; for multi-grain stacks where upstream ``massIn`` is large relative to the
        grain's own mass generation the true peak may shift fore-ward, but the narrowest
        port still dominates and the selection remains accurate in practice.

        getMassFlux computes:
            (massIn + density · cumulativeBurnArea(0..z) · dReg) / (coreArea(z) · dTime)

        Within a run of geometrically identical slices (same core-area count), coreArea is
        constant while cumulativeBurnArea strictly increases → mass flux is strictly
        increasing through the run.  Therefore only the **last (foremost) slice** of each
        identical run can be a peak; earlier slices in the same run are dominated.

        For continuously-varying geometries (conical, tapered), each run degenerates to a
        single slice.  The voxelization of a smooth cone produces small staircase oscillations
        that look like interior local minima in the run-end profile, but these are aliasing
        artefacts — a position with core area V can only beat a narrower position with area M
        if V/M < (cumulative burn ratio), which is impossible for V >> M.  The threshold
        filter `V ≤ global_min · pctThreshold` discards these artefacts automatically.

        Algorithm:
          1. Compute core_area_profile (parallelised O(n²·z) Cython scan, <1 ms).
          2. Collapse each run of equal core counts to its foremost (last) index.
          3. From the collapsed list, select candidates by quality:
               Faster – both grain ends ± 3 + foremost end of narrowest run ± 1
               Fast   – all run-ends within 35% of global-min core area
                        + local-minimum run-ends within 35% ± 1 neighbourhood
               Exact  – all run-ends within 20% of global-min core area
                        + local-minimum run-ends within 20% ± 2 neighbourhood
          4. Both startPos (aft) and endPos (fore) are always included unconditionally.
        """
        if endPos < startPos:
            return []

        count = endPos - startPos + 1
        if count <= 6:
            return list(range(startPos, endPos + 1))

        if not _hasFmm3dCy:
            return list(range(startPos, endPos + 1))

        mapDist = self.normalize(regDist)
        regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
        maskU8 = self._maskU8 if self._maskU8 is not None else self.mask

        # Parallelised O(n²·z) — negligible vs a single MC call.
        # Quantise mapDist to 6 decimal places so floating-point rounding across
        # consecutive timesteps that land at the same physical regression depth
        # reuses the cached profile instead of recomputing.
        _mapDistQ = round(float(mapDist), 6)
        _capKey = (_mapDistQ, startPos, endPos)
        coreCounts = self._capCache.get(_capKey)
        if coreCounts is None:
            coreCounts = get_core_area_profile(regressionMap, maskU8, mapDist, startPos, endPos)
            self._capCache[_capKey] = coreCounts
            self._capCache.move_to_end(_capKey)
            if len(self._capCache) > self._capCacheMax:
                self._capCache.popitem(last=False)

        if len(coreCounts) == 0:
            return list(range(startPos, endPos + 1))

        minCore = int(np.min(coreCounts))
        if minCore == 0:
            # Some z-slices have zero core count.  Distinguish two cases:
            #   (a) Burned-through: interior slices have zero area (grain is fully
            #       consumed at that cross-section) → fall back to brute-force all.
            #   (b) Fore-cap not yet penetrated: zeros appear only at the fore end of
            #       the profile because the bore hasn't reached those slices yet.
            #       In this case trim the zero-tail and continue normally.
            nonzeroIdxs = np.nonzero(coreCounts)[0]
            if len(nonzeroIdxs) == 0:
                return list(range(startPos, endPos + 1))
            lastNonzeroI = int(nonzeroIdxs[-1])
            firstZeroI   = int(np.argmax(coreCounts == 0))
            if firstZeroI <= lastNonzeroI:
                # Zero appears before the last non-zero → interior burnthrough
                return list(range(startPos, endPos + 1))
            # Zeros are strictly at the fore end: trim the zero tail
            coreCounts = coreCounts[:lastNonzeroI + 1]

        n = len(coreCounts)

        # Detect and remove fore-cap breakthrough transition.
        # When the bore approaches the inhibited fore face, the voxelised core area
        # tapers from the interior value down to near zero over many slices.  Including
        # those positions makes them the minCore, collapsing the candidate threshold so
        # only the tiny-bore cap slices are selected — causing astronomical flux spikes.
        #
        # Detection strategy: find the interior peak cc (median of middle 50%), then
        # scan from the fore end inward until we find the first slice that exceeds 40%
        # of that peak — everything strictly fore of that is considered the fore-cap
        # transition zone and excluded from both minCore and threshold candidates.
        # This handles both sudden (>80%) and gradual multi-slice tapering.
        _mid_lo = n // 4
        _mid_hi = 3 * n // 4
        if _mid_hi > _mid_lo:
            _interiorPeak = float(np.max(coreCounts[_mid_lo:_mid_hi]))
        else:
            _interiorPeak = float(np.max(coreCounts)) if n > 0 else 0.0
        _boreThresh = 0.4 * _interiorPeak

        boreEndN = n
        if _interiorPeak > 0:
            for _ci in range(n - 1, -1, -1):
                if coreCounts[_ci] >= _boreThresh:
                    boreEndN = _ci + 1  # exclusive: keep indices 0.._ci
                    break

        quality = self.props['meshedMassFluxQuality'].getValue()

        # Collapse each constant-area run to its foremost (last) slice.
        # Within a run, flux is strictly increasing, so only the last slice can be peak.
        runEnds = []   # local indices (0-based from startPos)
        for i in range(n):
            if i == n - 1 or coreCounts[i] != coreCounts[i + 1]:
                runEnds.append(i)

        # If a cap transition was found, restrict runEnds, coreCounts, and n to
        # the established bore region (aft of boreEndN).
        _endPosCapped = endPos  # endPos for unconditional candidate (may be adjusted)
        if boreEndN < n:
            establishedRunEnds = [i for i in runEnds if i < boreEndN]
            if establishedRunEnds:
                runEnds = establishedRunEnds
                coreCounts = coreCounts[:boreEndN]
                n = boreEndN
                # Cap endPos for unconditional inclusion so it stays in the established zone
                _endPosCapped = startPos + boreEndN - 1

        minCore = int(np.min(coreCounts)) if len(coreCounts) > 0 else 0

        if minCore == 0:
            return list(range(startPos, endPos + 1))

        # Both physical grain boundaries are unconditional candidates:
        #   startPos (aft/nozzle-side) – also run-end of first run
        #   _endPosCapped (fore, trimmed to established-bore zone if transition found)
        candidates = {startPos, _endPosCapped}

        if quality == 'Faster':
            # Grain ends ± 3 + smallest-area run-end ± 1 (covers the flux peak in
            # uniform grains where the dominant run ends well before endPos)
            minRunEndIdx = min(runEnds, key=lambda i: coreCounts[i])
            for d in range(1, 4):
                if startPos + d <= _endPosCapped:
                    candidates.add(startPos + d)
                if _endPosCapped - d >= startPos:
                    candidates.add(_endPosCapped - d)
            for d in range(-1, 2):
                pos = startPos + minRunEndIdx + d
                if startPos <= pos <= _endPosCapped:
                    candidates.add(pos)
            return sorted(candidates)

        # Fast: within 35% of min + local minima ±1
        # Exact: within 20% of min + local minima ±2
        pctThreshold = 1.35 if quality == 'Fast' else 1.20
        nbr = 1 if quality == 'Fast' else 2
        coreThreshold = minCore * pctThreshold
        minIdx = int(np.argmin(coreCounts))

        # All run-ends with core area within the threshold
        for i in runEnds:
            if coreCounts[i] <= coreThreshold:
                candidates.add(startPos + i)

        # Local minima among run-ends that are also within the area threshold.
        # This excludes voxelization-aliasing artefacts on monotone profiles (e.g. conical):
        # a local dip at 2× the global-min core area cannot be the flux peak because any
        # position with half the port area has a proportionally higher flux denominator.
        reVals = [coreCounts[i] for i in runEnds]
        m = len(reVals)
        for k in range(1, m - 1):
            if (reVals[k] <= reVals[k - 1] and reVals[k] <= reVals[k + 1]
                    and reVals[k] <= coreThreshold):
                for d in range(-nbr, nbr + 1):
                    pos = startPos + runEnds[k] + d
                    if startPos <= pos <= _endPosCapped:
                        candidates.add(pos)

        # Always include the global core-area minimum with its neighbourhood
        for d in range(-nbr, nbr + 1):
            pos = startPos + minIdx + d
            if startPos <= pos <= _endPosCapped:
                candidates.add(pos)

        return sorted(candidates)
    
    def volumeToMap(self, value):
        """Used to convert cu m to cu pixels, like on the regression map."""
        return (self.mapDim ** 3) * (value / (self.props['diameter'].getValue() ** 3))

    def mapToVolume(self, value):
        """Used to convert cu pixels to cu m. For extracting real volumes from the regression map."""
        return (self.props['diameter'].getValue() ** 3) * (value / (self.mapDim ** 3))
    
    @abstractmethod
    def generateCoreMap(self):
        """Generate an image of the grain cross section in self.coreMap. A 0 in the image means propellant, and a 1 means no propellant."""

    def simulationSetup(self, config, cancelCheck=None, statusCb=None):
        self.mapDim = config.getProperty("3DmapDim")
        # mapLength = self.mapDim * np.ceil(self.lengthToMap(self.props['length'].getValue() + self.props['diameter'].getValue()) / self.lengthToMap(self.props['diameter'].getValue())).astype(int)

        if statusCb:
            statusCb("Voxelizing mesh\u2026")
        self.generateCoreMap()
        if cancelCheck and cancelCheck():
            raise _GrainSetupCanceled()
        if statusCb:
            statusCb("Computing distance field\u2026")
        self.generateRegressionMap(cancelCheck=cancelCheck, statusCb=statusCb)

    @staticmethod
    def _makeCoreMapCacheKey(coreMap, inhibitedEnds, mapDim):
        """Cheap fingerprint of coreMap inputs used to skip redundant generateRegressionMap calls."""
        # shape + voxel sum is fast and collision-free for all practical grain geometries
        v = coreMap.view(np.uint8)
        return (coreMap.shape, int(v.sum()), inhibitedEnds, mapDim)

    def generateRegressionMap(self, cancelCheck=None, statusCb=None):
        """Uses the fast marching method to generate an image of how the grain regresses from the core map. The map
        is stored under self.regressionMap."""

        # Build the cache key without unpacking large packed arrays.
        # For packed storage the precomputed sum avoids a full 100+ MB unpack on
        # every call.  For small dense arrays _makeCoreMapCacheKey is fast anyway.
        _packed = self.__dict__.get('_coreMapPacked')
        if _packed is not None:
            _shape  = self.__dict__['_coreMapShape']
            _sum    = self.__dict__.get('_coreMapSum', 0)
            cacheKey = (_shape, _sum, self.props['inhibitedEnds'].getValue(), self.mapDim)
        else:
            cacheKey = self._makeCoreMapCacheKey(
                self.coreMap, self.props['inhibitedEnds'].getValue(), self.mapDim
            )
        if cacheKey == self._regressionMapCacheKey and self.regressionMap is not None:
            return  # coreMap unchanged — reuse existing regressionMap

        # Unpack the coreMap once for all computation below (avoids repeated
        # getter calls, each of which would re-allocate the dense array).
        _coreMap = self.coreMap

        # The following lines of code are done to enable burning from the ends without adding the "uninhibited disk"
        # length to the total grain length. Should probably rework how initGeometry, generateCoreMap, and generateRegressionMap
        # work for the Fmm3DGrain as they make less sense in this context
        mask = self.mask
        # uninhibitedDisk must match the coreMap cross-section exactly.
        # Normally shape[1] == shape[2] == mapDim, but if the STL's long axis
        # doesn't match coreAxis the dims can differ; use the actual shape so
        # np.insert/append don't raise a broadcast error.
        _cm_ydim = _coreMap.shape[1]
        _cm_xdim = _coreMap.shape[2]
        uninhibitedDisk = np.zeros((1, _cm_ydim, _cm_xdim))
        coreMapUninhib = _coreMap
        if self.props['inhibitedEnds'].getValue() in ['Top', 'Neither']:# BOTTOM
            coreMapUninhib = np.insert(coreMapUninhib, 0, uninhibitedDisk, axis=0)
        if self.props['inhibitedEnds'].getValue() in ['Bottom', 'Neither']:# TOP
            coreMapUninhib = np.append(coreMapUninhib, uninhibitedDisk, axis=0)
        if self.props['inhibitedEnds'].getValue() != 'Both':
            _xm, _ym = np.meshgrid(
                np.linspace(-1, 1, _cm_ydim), np.linspace(-1, 1, _cm_xdim), indexing='ij'
            )
            mask = np.broadcast_to(
                (_xm**2 + _ym**2 > 1)[np.newaxis, :, :], coreMapUninhib.shape
            )
        valid = np.logical_not(mask)

        cellSize = 1 / self.mapDim
        _fmmKey = self._computeFmmCacheKey(coreMapUninhib, cellSize)
        regressionMapUninhib = self._loadFmmCache(_fmmKey)
        if regressionMapUninhib is None:
            # Apply the cylinder mask so skfmm skips outside-cylinder voxels.
            # This is the same pattern the 2-D PerforatedGrain path uses; without
            # it the FMM propagates through ~21 % extra corner voxels and can't
            # terminate early at the cylinder boundary.
            _maskFull = np.ascontiguousarray(mask)
            _maskedPhi = np.ma.MaskedArray(coreMapUninhib, _maskFull)
            _fmmResult = skfmm.distance(_maskedPhi, dx=cellSize)
            # Fill masked positions with max distance so marching-cubes never
            # sees a spurious zero-crossing at the cylinder boundary.
            _fill = max(float(_fmmResult.max()), 1.0)
            regressionMapUninhib = _fmmResult.filled(_fill)
            regressionMapUninhib *= 2           # in-place — avoids a full copy
            del _maskFull, _maskedPhi, _fmmResult
            self._saveFmmCache(_fmmKey, regressionMapUninhib)

        if cancelCheck and cancelCheck():
            raise _GrainSetupCanceled()

        # # FIXME:
        # plt.figure(figsize=(16,8))
        # plt.contourf(self.regressionMap[:,int(self.mapDim/2),:], cmap='viridis', aspect='equal')
        # plt.colorbar()
        # plt.gca().set_aspect("equal")
        # plt.show()

        # Only consider non-masked (inside-cylinder) positions for maxDist;
        # filled positions carry an artificial large value that would inflate
        # wallWeb and break burnout detection.
        maxDist = float(np.amax(regressionMapUninhib[valid]))
        self.wallWeb = self.unNormalize(maxDist)

        polled = []
        burningArea = []
        targetLevels = int(maxDist * self.mapDim) * 10
        maxMarchingLevels = int(
            os.environ.get('OPENMOTOR_EXPERIMENTAL_MAX_MARCHING_LEVELS', '250')
        )
        numLevels = max(2, min(targetLevels, maxMarchingLevels))

        if statusCb:
            statusCb("Computing burning area\u2026")

        # Adaptive two-pass MC sweep: coarse uniform sample + curvature-driven
        # refinement.  Reduces MC evaluations by ~4–5× vs a uniform sweep of
        # numLevels points while preserving interpolation accuracy on smooth
        # burning-area curves.  High-curvature intervals (rapid area change) get
        # extra mid-points; flat intervals are left at coarse resolution.
        _nCoarse = max(2, min(20, numLevels))
        _coarseLevels = np.linspace(0.0, maxDist, _nCoarse, endpoint=False).tolist()

        # --- Coarse pass (sequential — only ~20 calls, minimal overhead) ---
        _coarsePairs: list = []  # [(level, area_m2), ...]
        _firstFailedCoarse: float | None = None
        for _lvl in _coarseLevels:
            if cancelCheck and cancelCheck():
                raise _GrainSetupCanceled()
            _ap = _marchingAreaPixels(regressionMapUninhib, valid, _lvl)
            if _ap is None:
                _firstFailedCoarse = _lvl
                break
            _coarsePairs.append((_lvl, self.mapToArea(_ap)))

        # --- Refinement pass: bisect intervals with high normalised curvature ---
        _finePairs: list = []
        if numLevels > _nCoarse and len(_coarsePairs) >= 3:
            _clvls  = np.array([p[0] for p in _coarsePairs])
            _careas = np.array([p[1] for p in _coarsePairs])
            _areaRange = max(float(_careas.max() - _careas.min()), 1e-30)
            # Normalised second difference: proxy for |d²A/dr²| over each triplet.
            _d2 = np.abs(np.diff(_careas, n=2)) / _areaRange
            _curvThresh = float(
                os.environ.get('OPENMOTOR_ADAPTIVE_MARCH_THRESHOLD', '0.05')
            )
            _budget = numLevels - len(_coarsePairs)
            _fineSet: set = set()
            for _i, _c in enumerate(_d2):
                if _c > _curvThresh:
                    _fineSet.add((_clvls[_i]     + _clvls[_i + 1]) / 2.0)
                    _fineSet.add((_clvls[_i + 1] + _clvls[_i + 2]) / 2.0)
            _fineLevelsSorted = sorted(_fineSet)[:_budget]

            if _fineLevelsSorted:
                _fineParallel = (
                    os.environ.get(
                        'OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP', ''
                    ).lower() in ('1', 'true', 'yes')
                    or len(_fineLevelsSorted) > 10
                )
                if _fineParallel and len(_fineLevelsSorted) > 4:
                    from concurrent.futures import ThreadPoolExecutor
                    _workers = int(
                        os.environ.get(
                            'OPENMOTOR_EXPERIMENTAL_PARALLEL_SWEEP_WORKERS', '0'
                        ) or 0
                    )
                    if _workers <= 0:
                        _workers = os.cpu_count() or 1
                    with ThreadPoolExecutor(max_workers=_workers) as _exec:
                        _rawAreas = list(_exec.map(
                            lambda lv: _marchingAreaPixels(
                                regressionMapUninhib, valid, lv
                            ),
                            _fineLevelsSorted,
                        ))
                    if cancelCheck and cancelCheck():
                        raise _GrainSetupCanceled()
                else:
                    _rawAreas = []
                    for lv in _fineLevelsSorted:
                        if cancelCheck and cancelCheck():
                            raise _GrainSetupCanceled()
                        _rawAreas.append(_marchingAreaPixels(regressionMapUninhib, valid, lv))
                _finePairs = [
                    (lv, self.mapToArea(av))
                    for lv, av in zip(_fineLevelsSorted, _rawAreas)
                    if av is not None
                ]

        # Merge coarse + fine sorted by level, expand into polled/burningArea.
        for _lvl, _area in sorted(_coarsePairs + _finePairs):
            polled.append(_lvl)
            burningArea.append(_area)

        # Append zero-area sentinels so the interpolation drops to zero near
        # the actual burnout instead of linearly extrapolating non-zero area
        # across a large gap to maxDist.  The first failed coarse level is
        # where MC found no surface — area is physically zero there.  Adding
        # it (and maxDist) as explicit zeros gives the SG smoother enough
        # zero-points to keep the tail near zero.
        if _firstFailedCoarse is not None and (not polled or _firstFailedCoarse > polled[-1]):
            polled.append(_firstFailedCoarse)
            burningArea.append(0.0)
        if len(polled) == 0 or polled[-1] < maxDist:
            polled.append(maxDist)
            burningArea.append(0.0)

        self._maxPolledDist = maxDist  # used by getSurfaceAreaAtRegression / getVolumeAtRegression
        if len(polled) < 2:
            self.faceArea = np.zeros(2)
            self.faceAreaFunc = interpolate.interp1d([0.0, 1.0], [0.0, 0.0])
            self.volumeFunc = None
        else:
            # Compute volume curve and SA on a uniform grid so that
            # _smoothSeries (SG filter) and np.gradient both see even spacing.
            # The MC-derived adaptive 'polled' has non-uniform spacing that
            # produces derivative artifacts at refinement boundaries.
            _rmSorted = np.sort(regressionMapUninhib[valid].ravel())
            _nVox = len(_rmSorted)
            _nPts = max(len(polled), 50)
            _uniformPolled = np.linspace(0.0, maxDist, _nPts)
            _idxs = np.searchsorted(_rmSorted, _uniformPolled, side='right')
            _volCounts = self.mapToVolume(_nVox - _idxs.astype(float))
            _smoothVol = _smoothSeries(_volCounts)
            self.volumeFunc = interpolate.interp1d(
                _uniformPolled, _smoothVol,
                fill_value=(_smoothVol[0], 0.0),
                bounds_error=False,
            )

            # Derive surface area from the volume curve via the co-area
            # formula: SA(r) = -dV/dr.  This is exact for uniform regression
            # and automatically consistent with the volume (mass) tracking,
            # eliminating the phantom-thrust divergence that marching-cubes
            # iso-surface area exhibits on complex geometries.
            _diameter = self.props['diameter'].getValue()
            _dr = _uniformPolled[1] - _uniformPolled[0]
            _dVdMap = np.gradient(_smoothVol, _dr)
            self.faceArea = np.maximum(-_dVdMap * (2.0 / _diameter), 0.0)
            self.faceAreaFunc = interpolate.interp1d(
                _uniformPolled, self.faceArea,
                fill_value=(float(self.faceArea[0]), 0.0),
                bounds_error=False,
            )

        # Remove uninhibited disks, if necessary
        if self.props['inhibitedEnds'].getValue() in ['Top', 'Neither']:# BOTTOM
            regressionMapUninhib = regressionMapUninhib[1:]
        if self.props['inhibitedEnds'].getValue() in ['Bottom', 'Neither']:# TOP
            regressionMapUninhib = regressionMapUninhib[:-1]

        self.regressionMap = regressionMapUninhib
        self._validMask = np.logical_not(self.mask)
        self._maskU8 = np.ascontiguousarray(self.mask, dtype=np.uint8)
        self._regressionMapF64 = np.ascontiguousarray(self.regressionMap, dtype=np.float32)
        # Invalidate the core-area-profile cache whenever the regression map is regenerated
        # so stale profiles from a previous mapDim/geometry are never reused.
        self._capCache.clear()

        # Precompute smooth port-area curve (core area at the aft face, vs mapDist).
        # This mirrors faceAreaFunc but for the denominator of getMassFlux so that
        # the massFlux3D=False path has a smooth denominator without ring-event spikes.
        #
        # Only valid for grains with an INHIBITED aft end ('Both' or 'Bottom').
        # For aft-uninhibited grains ('Top', 'Neither'), the FMM end-disk effect
        # skews distances at z=0 of the stored map, and propEndPos[0] retreats
        # during the burn — making a fixed z=0 portAreaFunc unreliable.
        #
        # Also precompute foreAreaFunc for the fore face (last 5% of z-slices,
        # min core area), which smooths the denominator for massFlux3D=True
        # peak candidates near the fore end (e.g. conical with inverted taper).
        _aftInhibited  = self.props['inhibitedEnds'].getValue() in ('Both', 'Bottom')
        _foreInhibited = self.props['inhibitedEnds'].getValue() in ('Both', 'Top')
        if _hasFmm3dCy and len(polled) >= 2:
            if cancelCheck and cancelCheck():
                raise _GrainSetupCanceled()
            if statusCb:
                statusCb("Building port area lookup\u2026")
            _zdim = self._regressionMapF64.shape[0]
            _scanAft  = max(2, _zdim // 20)           # z = 0 .. _scanAft-1
            _scanFore = max(2, _zdim // 20)            # z = _zdim-_scanFore .. _zdim-1
            _portCounts = [] if _aftInhibited  else None
            _foreCounts = [] if _foreInhibited else None
            for _lvl in polled:
                if cancelCheck and cancelCheck():
                    raise _GrainSetupCanceled()
                if _aftInhibited:
                    _best = 0
                    for _z in range(_scanAft):
                        _c = get_core_area_count_at_slice(
                            self._regressionMapF64, self._maskU8, _lvl, _z
                        )
                        if _c > _best:
                            _best = _c
                    _portCounts.append(self.mapToArea(_best))
                if _foreInhibited:
                    _best = 0
                    for _z in range(_zdim - _scanFore, _zdim):
                        _c = get_core_area_count_at_slice(
                            self._regressionMapF64, self._maskU8, _lvl, _z
                        )
                        if _c > _best:
                            _best = _c
                    _foreCounts.append(self.mapToArea(_best))
            _interpKwargs = dict(bounds_error=False)
            if _aftInhibited:
                self.portAreaFunc = interpolate.interp1d(
                    polled, _smoothSeries(_portCounts),
                    fill_value=(_portCounts[0], _portCounts[-1]),
                    **_interpKwargs,
                )
            else:
                self.portAreaFunc = None
            if _foreInhibited:
                self.foreAreaFunc = interpolate.interp1d(
                    polled, _smoothSeries(_foreCounts),
                    fill_value=(_foreCounts[0], _foreCounts[-1]),
                    **_interpKwargs,
                )
            else:
                self.foreAreaFunc = None
        else:
            self.portAreaFunc = None
            self.foreAreaFunc = None

        self._regressionMapCacheKey = self._makeCoreMapCacheKey(
            self.coreMap, self.props['inhibitedEnds'].getValue(), self.mapDim
        )

    def getSurfaceAreaAtRegression(self, regDist):
        mapDist = self.normalize(regDist)
        if mapDist >= getattr(self, '_maxPolledDist', mapDist + 1):
            return 0  # Past burnout
        return self.faceAreaFunc(mapDist)

    def getFaceImage(self, mapDim):
        # if self.coreMap is None:
        #     self.generateCoreMap(self.mapDim)
        masked = np.ma.MaskedArray(self.coreMap, self.mask)
        return masked

    def getRegressionImage(self, mapDim):
        # if self.regressionMap is None:
        #     self.generateCoreMap(self.mapDim)
        #     self.generateRegressionMap()
        masked = np.ma.MaskedArray(self.regressionMap, self.mask)
        return masked

    def getVolumeAtRegression(self, regDist):
        mapDist = self.normalize(regDist)
        if mapDist >= getattr(self, '_maxPolledDist', mapDist + 1):
            return 0  # Past burnout
        if self.volumeFunc is not None:
            return max(0.0, float(self.volumeFunc(mapDist)))
        if _hasFmm3dCy:
            regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
            maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
            voxelCount = get_volume_count_gt_threshold(regressionMap, maskU8, mapDist)
            return self.mapToVolume(voxelCount)

        regressionMasked = np.ma.MaskedArray(self.regressionMap, self.mask)
        return self.mapToVolume(np.sum(regressionMasked > mapDist))
    
    def getGrainBoundingVolume(self):
        """Returns the volume of the bounding cylinder around the grain"""
        if self.totalLength is None:
            self.generateCoreMap()

        return geometry.cylinderVolume(self.props['diameter'].getValue(), self.totalLength.getValue())

    def getWebLeft(self, regDist):
        wallLeft = self.wallWeb - regDist
        return wallLeft
    
    def getMassFlux(self, massIn, dTime, regDist, dRegDist, position, density):
        mapDist = self.normalize(regDist)
        zdim = self.regressionMap.shape[0]
        zPos = max(0, min(int(position), zdim - 1))

        # Fast path: reuse precomputed burning-area curve (massFlux3D=False).
        # The precomputed curve is computed at Exact-quality MC, so using it for
        # Fast/Faster modes gives strictly more accurate results than live MC.
        if (not self.props['massFlux3D'].getValue()
                and self.faceAreaFunc is not None
                and os.environ.get('OPENMOTOR_DISABLE_FACEAREA_REUSE', '').lower()
                        not in ('1', 'true', 'yes')):
            # Use the precomputed smooth portAreaFunc for aft-face queries to avoid
            # ring-event voxelization spikes in the denominator.  Fall back to the
            # raw voxel count for positions deep inside the grain (not near aft).
            if _hasFmm3dCy:
                regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
                maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
                if self.portAreaFunc is not None and zPos < zdim // 20:
                    coreArea = float(self.portAreaFunc(mapDist))
                else:
                    coreArea = self.mapToArea(
                        get_core_area_count_at_slice(regressionMap, maskU8, mapDist, zPos)
                    )
            else:
                coreArea = self.mapToArea(np.sum(
                    np.ma.MaskedArray(self.regressionMap, self.mask)[zPos] < mapDist
                ))
            if coreArea <= 0:
                return 0
            return (massIn + density * self.getSurfaceAreaAtRegression(regDist) * dRegDist) / (coreArea * dTime)

        # 3D mass-flux path: z-fraction approach.
        # Mass flow delivered to position z:
        #   massFlow(z) = massIn + density * faceArea(reg) * dRegDist * zFraction
        # where:
        #   faceArea(reg)  = SG-filtered total burning surface area (smooth)
        #   zFraction      = V_prop(0..z, reg) / V_prop_total(reg)
        #                  = fraction of remaining propellant ahead of/at z
        # This avoids differentiating the (noisy) discrete voxel-count volume
        # function, which creates large batch-transition spikes for prismatic
        # grains where the same 2D topology event repeats across every z-slice.
        if _hasFmm3dCy:
            regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
            maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
            coreAreaVoxels = get_core_area_count_at_slice(regressionMap, maskU8, mapDist, zPos)
            coreArea = self.mapToArea(coreAreaVoxels)
            if coreArea <= 0:
                return 0
            v_total = get_volume_count_gt_threshold(regressionMap, maskU8, mapDist)
            v_after = get_volume_count_gt_threshold_from_z(regressionMap, maskU8, mapDist, zPos + 1) \
                      if zPos + 1 < zdim else 0
            zFraction = (v_total - v_after) / v_total if v_total > 0 else 0.0
            burningVolume = self.getSurfaceAreaAtRegression(regDist) * dRegDist * zFraction
            return (massIn + density * burningVolume / dTime) / coreArea

        # No-Cython fallback: MC surface area * dRegDist.
        mcMask, _ = self._getMeshedMarchingMask(position)
        coreAreaVoxels = np.sum(
            np.ma.MaskedArray(self.regressionMap, self.mask)[zPos] < mapDist
        )
        coreArea = self.mapToArea(coreAreaVoxels)
        if coreArea <= 0:
            return 0
        stepSize = self._getMeshedMarchingStepSize(regDist, dRegDist, zPos)
        areaPixels = _marchingAreaPixels(self.regressionMap, mcMask, mapDist, step_size=stepSize)
        if areaPixels is None:
            return 0
        return (massIn + density * self.mapToArea(areaPixels) * dRegDist) / (coreArea * dTime)

    def getPeakMassFlux(self, massIn, dTime, regDist, dRegDist, density):
        propEndPos = self.getEndPositionsInMapDim(regDist)

        if self.props['massFlux3D'].getValue():
            startPos = int(propEndPos[0])
            # Exclude the fore end wall (propEndPos[1]) which is the inhibited closed face;
            # its core area may be zero and it does not contribute meaningful axial flux.
            endPos = int(propEndPos[1] - 1)

            if endPos < startPos:
                return 0.0

            candidates = self._getMeshedPeakSearchCandidates(
                startPos, endPos, massIn, dTime, regDist, dRegDist, density
            )
            if not candidates:
                return 0.0

            if _hasFmm3dCy:
                regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
                maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
                mapDist = self.normalize(regDist)
                mapDist_dreg = self.normalize(regDist + dRegDist)
                zdim = regressionMap.shape[0]

                # One parallelised O(n³) scan gives per-z core counts and the
                # propellant suffix sums at reg.  Each candidate is then evaluated
                # in O(1) using the z-fraction approach:
                #   massFlow(z) = massIn + faceArea(reg)*dReg * (V_0_to_z / V_total)
                # This is smooth because faceArea is SG-filtered and V-ratios are
                # stable, avoiding the derivative-of-discrete-volume noise that
                # created large spikes for prismatic grains.
                coreCounts, propSfxReg, propSfxDreg = get_massflux_slice_suffix_arrays(
                    regressionMap, maskU8, mapDist, mapDist_dreg
                )
                totalReg = int(propSfxReg[0]) if zdim > 0 else 0

                smoothBurnVol = self.getSurfaceAreaAtRegression(regDist) * dRegDist

                peakFlux = -np.inf
                _aftThresh  = zdim // 20
                _foreThresh = zdim * 19 // 20
                mapDist_float = float(mapDist)
                for position in candidates:
                    z = position
                    coreAreaVoxels = int(coreCounts[z])
                    if coreAreaVoxels <= 0:
                        continue
                    # Use precomputed smooth area functions for near-aft / near-fore positions
                    # to avoid ring-event voxelisation spikes in the denominator.
                    if self.portAreaFunc is not None and z < _aftThresh:
                        coreArea = float(self.portAreaFunc(mapDist_float))
                    elif self.foreAreaFunc is not None and z > _foreThresh:
                        coreArea = float(self.foreAreaFunc(mapDist_float))
                    else:
                        coreArea = self.mapToArea(coreAreaVoxels)
                    if coreArea <= 0:
                        continue
                    # zFraction = V_prop(0..z) / V_total
                    sfxReg  = int(propSfxReg[z + 1]) if z + 1 < zdim else 0
                    partial  = totalReg - sfxReg
                    zFraction = partial / totalReg if totalReg > 0 else 0.0
                    massFlow = massIn + zFraction * smoothBurnVol * density / dTime
                    massFlux = massFlow / coreArea
                    if massFlux > peakFlux:
                        peakFlux = massFlux
                return peakFlux if peakFlux != -np.inf else 0.0

            # No-Cython fallback: evaluate each candidate via getMassFlux (uses MC).
            peakFlux = -np.inf
            for position in candidates:
                massFlux = self.getMassFlux(massIn, dTime, regDist, dRegDist, position, density)
                if massFlux > peakFlux:
                    peakFlux = massFlux
            return peakFlux if peakFlux != -np.inf else 0.0

        return self.getMassFlux(massIn, dTime, regDist, dRegDist, propEndPos[0], density)

    def getEndPositions(self, regDist):
        """Returns the positions of the grain ends relative to the original (unburned) grain fore in the format of (fore,aft)"""
        aft, fore = self.getEndPositionsInMapDim(regDist)
        return self.mapToLength(self.mapLength - fore - 1), self.mapToLength(self.mapLength - aft - 1)
        
    def getEndPositionsInMapDim(self, regDist):
        """Returns the aft-most, fore-most indices into regressionMap and coreMap that still have prop"""
        # if self.regressionMap is None:
        #     self.generateCoreMap(self.mapDim)
        #     self.generateRegressionMap()

        mapDist = self.normalize(regDist)
        if _hasFmm3dCy:
            regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
            maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
            firstIdx, lastIdx = get_first_last_prop_indices(regressionMap, maskU8, mapDist)
            if firstIdx < 0:
                return 0, 0
            return firstIdx, lastIdx

        mapAtReg = np.ma.MaskedArray(self.regressionMap, self.mask) > mapDist
        mapAtReg = mapAtReg.reshape((mapAtReg.shape[0], -1))

        lengthwiseProp = np.sum(mapAtReg, axis=1)
        idxHasProp = np.asarray(lengthwiseProp > 0).nonzero()[0]

        if len(idxHasProp) == 0:
            return 0, 0
        # print(self.mapLength - idxHasProp[-1] - 1, self.mapLength - idxHasProp[0] - 1)
        return idxHasProp[0], idxHasProp[-1]

    def getPortArea(self, regDist):
        """For a given regDist, gets the aft-most bore area of the grain.

        Scans the first ~5% of z-slices from the aft end and returns the
        maximum core voxel count found.  This correctly handles grains with
        inhibited solid end faces (caps): a cap slice has near-zero core area
        and would underestimate port area if used alone, so taking the maximum
        across several aft-side slices skips the cap and captures the first
        real bore opening.
        """

        mapDist = self.normalize(regDist)
        if _hasFmm3dCy:
            regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
            maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
            zdim = regressionMap.shape[0]
            # Scan the first ~5 % of aft-facing slices for the best (maximum)
            # bore area.  Solid end-cap faces (inhibitedEnds != Neither/Top)
            # appear as near-zero core and are automatically out-competed.
            scanTo = max(2, zdim // 20)
            best = get_core_area_count_at_slice(regressionMap, maskU8, mapDist, 0)
            for z in range(1, scanTo):
                c = get_core_area_count_at_slice(regressionMap, maskU8, mapDist, z)
                if c > best:
                    best = c
            return self.mapToArea(best)

        mapAtReg = np.ma.MaskedArray(self.regressionMap, self.mask) > mapDist

        mapAtReg = mapAtReg.reshape((mapAtReg.shape[0], -1))

        lengthwiseProp = np.sum(mapAtReg, axis=1)
        lengthwiseCores = np.sum(np.logical_not(mapAtReg), axis=1)

        # Use the same max-in-first-5% logic as the Cython path to handle
        # solid end caps.
        zdim = mapAtReg.shape[0]
        scanTo = max(2, zdim // 20)
        aftSliceCores = lengthwiseCores[:scanTo][lengthwiseProp[:scanTo] > 0]
        if len(aftSliceCores) == 0:
            return self.mapToArea(0)
        return self.mapToArea(int(aftSliceCores.max()))

    def getInitialLength(self):
        if self.totalLength is None:
            return self.props['length'].getValue()
        return self.totalLength.getValue()

    def getDetailsString(self, lengthUnit='m'):
        """Returns a short string describing the grain, formatted using the units that is passed in"""
        if self.totalLength is None:
            return 'Length: {}'.format(self.props['length'].dispFormat(lengthUnit))

        return 'Length: {}'.format(self.totalLength.dispFormat(lengthUnit))

    def getGeometryErrors(self):
        """Returns a list of simAlerts that detail any issues with the geometry of the grain. Errors should be
        used for any condition that prevents simulation of the grain, while warnings can be used to notify the
        user of possible non-fatal mistakes in their entered numbers. Subclasses should still call the superclass
        method, as it performs checks that still apply to its subclasses."""
        errors = []
        if self.props['diameter'].getValue() == 0:
            errors.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.GEOMETRY, 'Diameter must not be 0'))
        if self.props['length'].getValue() > 0 and self.props['inhibitedEnds'].getValue() in ['Neither', 'Bottom']:
            errors.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.GEOMETRY, 'Cannot have endburner when grain top uninhibited'))
        return errors
