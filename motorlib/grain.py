"""
This module includes the base classes from which all grain classes should inherit. None of these objects
should be instantiated directly.
"""

from abc import abstractmethod
import copy
import hashlib
import os
import pathlib
import struct
import tempfile
from collections import OrderedDict
from typing import Set, Tuple, List, Union

import numpy as np
import skfmm
from scipy import interpolate, ndimage
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

try:
    from mathlib._march import marching_area_pixels_by_layer as _marchingLayerAreas
    _hasLayerAreas = True
except ImportError:
    _hasLayerAreas = False

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


def getCrossSectionCoordinates(mapDim: int, cellCentered: bool = False) -> np.ndarray:
    """Returns the position of every pixel along one axis of a cross section image, with the casting tube wall at
    radius 1. Pixels are 1/mapDim of the diameter wide, so cell centered coordinates, (2i + 1) / mapDim - 1, give an
    exact scale. The default puts the first and last pixel on the wall, which is the layout of the 2D grains."""
    if cellCentered:
        return (2 * np.arange(mapDim) + 1) / mapDim - 1
    return np.linspace(-1, 1, mapDim)


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

    def convertTo3D(self) -> "Grain":
        """Returns a 3D FMM grain with this grain's geometry, or the grain itself if it is already 3D or has no 3D
        equivalent."""
        return self

    def _addEndburnerProperty(self) -> None:
        """Gives the grain an endburner section: solid propellant above the core at the head end, bonded to the grain, that
        burns only through the open core. It only exists in the 3D simulation, which needs the head end inhibited."""
        self.props["endburnerLength"] = FloatProperty("Endburner Length", "m", 0, 10)

    def getEndburnerLength(self) -> float:
        """The length of the grain's endburner section, 0 for a grain that has none."""
        return self.props["endburnerLength"].getValue() if "endburnerLength" in self.props else 0.0

    def getHiddenProperties(self, simulatedIn3D: bool) -> Set[str]:
        """Names of the properties an editor should leave out for a motor that is, or is not, simulated in 3D. The endburner
        section is a 3D property, so it is shown while the motor is simulated in 3D, or while it is set so that a value in
        use is never out of sight."""
        if "endburnerLength" in self.props and not simulatedIn3D and self.getEndburnerLength() == 0:
            return {"endburnerLength"}
        return set()

    def getLengthString(self, lengthUnit: str = "m") -> str:
        """Returns the grain's length for display, formatted using the units that is passed in, with the endburner section
        above the core when the grain has one."""
        text = self.props["length"].dispFormat(lengthUnit)
        if self.getEndburnerLength() > 0:
            text += " + {} endburner".format(self.props["endburnerLength"].dispFormat(lengthUnit))
        return text

    def getExtrudedMask(self, mapDim: int) -> np.ndarray:
        """Returns the cross section mask (True is outside the grain) used when this grain is converted to 3D. The
        default masks everything outside the casting tube."""
        coords = getCrossSectionCoordinates(mapDim, cellCentered=True)
        return coords[:, None] ** 2 + coords[None, :] ** 2 > 1

    def getRegressedLength(self, regDist: float) -> float:
        """
        Returns the length of the grain when it has regressed a distance of 'regDist',
        taking any possible inhibition into account.
        """
        endPos = self.getEndPositions(regDist)
        return endPos[1] - endPos[0]

    def getDetailsString(self, lengthUnit: str = "m") -> str:
        """Returns a short string describing the grain, formatted using the units that is passed in."""
        return "Length: {}".format(self.getLengthString(lengthUnit))

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
        self._addEndburnerProperty()
        self.wallWeb: float = 0  # Max distance from the core to the wall
        self.cellCenteredMap = False  # Set when a 3D conversion needs the exact pixel scale

    def _getResampledFaceImage(self, mapDim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the cross section as a (propellant, masked) pair of boolean images with resolution mapDim."""
        faceDim = max(mapDim, 64)  # Some grains cannot draw their face below this resolution
        source = copy.deepcopy(self)
        source.cellCenteredMap = True
        face = source.getFaceImage(faceDim)
        propellant, masked = np.ma.getdata(face) > 0.5, np.ma.getmaskarray(face)
        if faceDim != mapDim:
            index = ((np.arange(mapDim) + 0.5) * faceDim / mapDim).astype(int)
            propellant, masked = propellant[np.ix_(index, index)], masked[np.ix_(index, index)]
        return propellant, masked

    def getExtrudedCoreMap(self, mapLength: int, mapDim: int) -> np.ndarray:
        """Returns the grain's 3D core map (True is propellant), shape (mapLength, mapDim, mapDim), with the aft end
        at index 0. The default extrudes the cross section from 'getFaceImage' along the grain."""
        propellant, _ = self._getResampledFaceImage(mapDim)
        return np.broadcast_to(propellant, (mapLength, mapDim, mapDim)).copy()

    def getExtrudedMask(self, mapDim: int) -> np.ndarray:
        """Masks the same regions as the grain's own face image, such as the support inside a rod and tube grain."""
        return self._getResampledFaceImage(mapDim)[1]

    def convertTo3D(self) -> "Grain":
        return ConvertedGrain(self, self.props["inhibitedEnds"].getValue(), self.getEndburnerLength())

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
        coords = getCrossSectionCoordinates(self.mapDim, self.cellCenteredMap)
        self.mapX, self.mapY = np.meshgrid(coords, coords)
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
    is provide an implementation of generateCoreMap that makes a 3D model of the grain and core.

    Axis 0 of the core map, the regression map and the mask runs from the aft (nozzle) end at index 0 to the fore
    (head) end at the highest index. Gas flows toward index 0, so the mass flow through a slice is the mass burned
    at or ahead of it. 'Bottom' inhibits the end at index 0 and 'Top' the end at the highest index."""
    geomName = None

    # ---- FMM distance-field disk cache -----------------------------------
    # skfmm.distance() on a 128³ grid takes 2–10 s; a disk cache eliminates
    # the cost on repeat runs (app restart, parameter sweep with fixed geometry).
    # Cache files are stored as .npy under the system temp directory.
    AREA_CURVE_POINTS = 50  # Samples of the volume curve whose slope gives the burning area
    CURVES_FROM_AREA = True  # Burning area comes from marching cubes areas instead of the slope of the volume curve
    SIGNED_DISTANCE = True  # Regression map is a signed distance with the surface at zero, not a distance from void voxel centers
    _FMM_DIST_CACHE_VERSION = '5'   # bump to invalidate all cached entries
    _FMM_DIST_CACHE_DIR: 'pathlib.Path | None' = None

    @classmethod
    def _getFmmCacheDir(cls) -> pathlib.Path:
        if cls._FMM_DIST_CACHE_DIR is None:
            d = pathlib.Path(tempfile.gettempdir()) / 'openmotor_fmm_cache'
            d.mkdir(exist_ok=True)
            cls._FMM_DIST_CACHE_DIR = d
        return cls._FMM_DIST_CACHE_DIR

    @staticmethod
    def _computeFmmCacheKey(coreMapUninhib: np.ndarray, cellSize: float, mask: 'np.ndarray | None' = None) -> str:
        """SHA-256 fingerprint of the FMM inputs → hex string cache key. The mask is part of the key because it
        decides which voxels the distance field covers, so the same core map can need different distances."""
        # Use buffer protocol directly — avoids a full .tobytes() copy that
        # would double memory usage on large 3-D grids (mapDim=256 → ~66 MB).
        arr = np.ascontiguousarray(coreMapUninhib, dtype=np.uint8)
        h = hashlib.sha256()
        h.update(arr.data)       # memoryview — no copy
        h.update(struct.pack('>d', cellSize))
        if mask is not None:
            h.update(struct.pack('>3q', *mask.shape))
            maskBits = mask[0] if mask.ndim == 3 and mask.strides[0] == 0 else mask  # One slice stands for a prism
            h.update(np.packbits(np.ascontiguousarray(maskBits, dtype=bool)).data)
        return h.hexdigest()

    @staticmethod
    def _getWallRing(mask: np.ndarray) -> np.ndarray:
        """Masked voxels that touch a valid voxel, diagonals included. These are the voxels a marching cube reaches from a
        valid corner, so giving them real distances lets the burning surface run all the way to the casting wall."""
        step = np.ones((3, 3), dtype=bool)
        if mask.ndim == 3 and mask.strides[0] == 0:
            return np.broadcast_to(ndimage.binary_dilation(~mask[0], structure=step) & mask[0], mask.shape)
        return ndimage.binary_dilation(~mask, structure=step[np.newaxis]) & mask

    def _getMaskSignature(self):
        """Cheap fingerprint of the mask for the in-memory regression map cache."""
        if self.mask is None:
            return None
        maskBits = self.mask[0] if self.mask.strides[0] == 0 else self.mask
        return (self.mask.shape, int(np.count_nonzero(maskBits)))

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
        # Burning area of each layer of voxels between adjacent z-slices, sampled at the marching cubes levels:
        # _layerAreas[i, k] is the area in pixels^2 of layer k at regression map level _layerLevels[i].
        self._layerLevels = None
        self._layerAreas = None
        self._layerOffset = 0  # Index of the first stored slice within the layers, which include any end disks
        self._areaShortfall = None  # Share of the propellant volume the marching cubes areas miss, mostly the wall sliver

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
        # Include only valid (inside-cylinder) voxels in slices zPos..end, which are the slices whose burning surface
        # feeds the flow at the measurement plane (gas flows toward the aft end at index 0). Slices before zPos
        # remain False so MC stops at the measurement plane. Including the outside-cylinder voxels generated
        # spurious iso-surface patches that caused jagged mass-flux output as regDist advanced.
        mcMask[zPos:] = self._validMask[zPos:]

        return mcMask, zPos

    def _getMeshedMarchingStepSize(self, regDist, dRegDist, zPos):
        quality = self.props['meshedMassFluxQuality'].getValue()
        if quality == 'Fast':
            return 2
        if quality == 'Faster':
            return 3
        return 1

    def _getCurvesFromAreas(self, levels, areas, propellantVolume, maxDist):
        """Burning area and propellant volume against regression from the marching cubes areas at each level.

        The areas are resampled at half voxel spacing and lightly smoothed. Marching cubes surfaces stop where the last
        voxel centers are, up to a voxel short of the casting wall, so the integral of the area falls a little short of
        the propellant volume. That shortfall is the last sliver at the wall, so the curve is extended to burnout at
        the final burning area instead of being rescaled. The volume is the integral of the area, which keeps mass and
        burning area consistent. Returns (regression map levels, volumes, areas, shortfall fraction), or None when the
        shortfall is above 12% of the volume and the areas cannot be trusted."""
        levels, areas = np.asarray(levels, dtype=float), np.asarray(areas, dtype=float)
        order = np.argsort(levels)
        levels, areas = levels[order], areas[order]
        distinct = np.concatenate([[True], np.diff(levels) > 1e-9])
        levels, areas = levels[distinct], areas[distinct]
        if len(levels) < 4 or areas.max() <= 0 or propellantVolume <= 0:
            return None
        step = 1.0 / self.mapDim  # Half a voxel in regression map units
        grid = np.linspace(0.0, maxDist, max(int(np.ceil(maxDist / step)) + 1, 50))
        area = np.nan_to_num(interpolate.PchipInterpolator(levels, areas, extrapolate=False)(grid), nan=0.0)
        if grid.size >= 7:
            area = savgol_filter(area, 5, 3)
        area = np.maximum(area, 0.0)
        toMeters = 0.5 * self.props['diameter'].getValue()
        integral = np.trapz(area, grid * toMeters)
        shortfall = (propellantVolume - integral) / propellantVolume
        if not -0.12 < shortfall < 0.12:
            return None
        if shortfall > 0:
            burning = np.nonzero(area > 0.02 * area.max())[0]
            lastArea = float(area[burning[-1]])
            extra = (propellantVolume - integral) / lastArea / toMeters  # Extra regression, in map units
            grid = np.concatenate([grid, grid[-1] + np.arange(1, int(np.ceil(extra / step)) + 1) * step])
            area = np.concatenate([area, np.full(grid.size - area.size, lastArea)])
            area[area.size - 1] = 0.0
            self._maxPolledDist = float(grid[-1])
        regression = grid * toMeters
        consumed = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(regression))])
        volume = np.maximum(propellantVolume - consumed, 0.0)
        volume[-1] = 0.0
        return grid, volume, area, shortfall

    @staticmethod
    def _getAftPortCount(counts) -> int:
        """Core voxel count at the aft end, from the counts of the first few slices. A solid end cap has little or no
        core, so slices below half of the best one are skipped. A tapering core keeps its true area at the aft face."""
        counts = np.asarray(counts)
        if counts.size == 0 or counts.max() <= 0:
            return 0
        return int(counts[np.argmax(counts >= 0.5 * counts.max())])

    def _getAreaFractionFore(self, mapDist):
        """Returns, for each slice of the regression map, the fraction of the burning surface that lies at or ahead
        of it (toward the fore end). Gas flows toward the aft end at index 0, so this is the share of the grain's
        mass flow that has passed that slice. Returns None when the layer areas are not available."""
        if self._layerAreas is None:
            return None
        levels = self._layerLevels
        if len(levels) < 2 or mapDist >= levels[-1]:
            return None
        i = int(np.clip(np.searchsorted(levels, mapDist, side='right') - 1, 0, len(levels) - 2))
        weight = float(np.clip((mapDist - levels[i]) / (levels[i + 1] - levels[i]), 0.0, 1.0))
        layers = (1 - weight) * self._layerAreas[i] + weight * self._layerAreas[i + 1]
        total = float(layers.sum())
        if total <= 0:
            return None
        ahead = np.concatenate([np.cumsum(layers[::-1])[::-1], [0.0]])
        return ahead[self._layerOffset:self._layerOffset + self.regressionMap.shape[0]] / total

    def _getCoreCounts(self, mapDist, startPos, endPos):
        """Core voxel count of every slice from startPos to endPos, cached for repeated queries at one depth."""
        regressionMap = self._regressionMapF64 if self._regressionMapF64 is not None else self.regressionMap
        maskU8 = self._maskU8 if self._maskU8 is not None else self.mask
        key = (round(float(mapDist), 6), startPos, endPos)
        counts = self._capCache.get(key)
        if counts is None:
            counts = get_core_area_profile(regressionMap, maskU8, mapDist, startPos, endPos)
            self._capCache[key] = counts
            self._capCache.move_to_end(key)
            if len(self._capCache) > self._capCacheMax:
                self._capCache.popitem(last=False)
        return counts

    def _getPeakFluxByArea(self, massIn, dTime, regDist, dRegDist, density, startPos, endPos):
        """Peak mass flux over slices startPos..endPos using the distribution of the burning surface along the grain.
        Returns None when that distribution is unavailable."""
        mapDist = self.normalize(regDist)
        fractions = self._getAreaFractionFore(mapDist)
        if fractions is None or not _hasFmm3dCy:
            return None
        counts = self._getCoreCounts(mapDist, startPos, endPos)
        positions = np.arange(startPos, endPos + 1)
        coreArea = self.mapToArea(counts.astype(float))
        massFlow = massIn + fractions[positions] * self.getSurfaceAreaAtRegression(regDist) * dRegDist * density / dTime
        usable = coreArea > 0
        return float((massFlow[usable] / coreArea[usable]).max()) if usable.any() else 0.0

    def getCoreProfile(self, regDist):
        """The core of the grain after `regDist` of regression, slice by slice from the aft end of the regression map:
        the port area [m^2] and the burning area [m^2] at or ahead of the slice, which is the part of the surface whose gas
        has passed the slice on its way to the nozzle. Returns None when the layer areas that place the burning surface
        are not available, as when the grain has burned out."""
        mapDist = self.normalize(regDist)
        fractions = self._getAreaFractionFore(mapDist)
        if fractions is None or not _hasFmm3dCy:
            return None
        counts = self._getCoreCounts(mapDist, 0, self.regressionMap.shape[0] - 1)
        return self.mapToArea(counts.astype(float)), fractions * self.getSurfaceAreaAtRegression(regDist)

    def _getMeshedPeakSearchCandidates(self, startPos, endPos, massIn, dTime, regDist, dRegDist, density):
        """Return candidate z-positions for peak-flux search in meshed 3D mode.

        NOTE: ``massIn``, ``dTime``, and ``density`` are accepted for API symmetry with
        ``getMassFlux`` but are **not used** by this algorithm.  The run-collapse approach
        is purely geometry-driven (core_area_profile proxy).  For single-grain motors this
        is exact; for multi-grain stacks where upstream ``massIn`` is large relative to the
        grain's own mass generation the true peak may shift fore-ward, but the narrowest
        port still dominates and the selection remains accurate in practice.

        getMassFlux computes:
            (massIn + density · burnAreaAhead(z) · dReg) / (coreArea(z) · dTime)

        where burnAreaAhead(z) is the burning surface from slice z to the fore end.  Within a run of
        geometrically identical slices (same core-area count), coreArea is constant while burnAreaAhead
        strictly decreases with index → mass flux is strictly decreasing through the run.  Therefore only the
        **first (aftmost) slice** of each identical run can be a peak; later slices in the same run are dominated.

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

        # Collapse each constant-area run to its aftmost (first) slice.
        # Gas flows toward the aft end at index 0, so within a run the mass flow only grows toward index 0 while the
        # core area is constant. Only the first slice of a run can be the peak.
        runEnds = []   # local indices (0-based from startPos)
        for i in range(n):
            if i == 0 or coreCounts[i] != coreCounts[i - 1]:
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
            cacheKey = (_shape, _sum, self.props['inhibitedEnds'].getValue(), self.mapDim, self._getMaskSignature())
        else:
            cacheKey = self._makeCoreMapCacheKey(
                self.coreMap, self.props['inhibitedEnds'].getValue(), self.mapDim
            ) + (self._getMaskSignature(),)
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
            if self.mask is not None and self.mask.strides[0] == 0:
                # The mask is the same on every slice, so the disks can reuse it. This keeps any extra masked
                # region, such as the hollow support inside the rod of a rod and tube grain.
                outside = self.mask[0]
            else:
                _xm, _ym = np.meshgrid(
                    getCrossSectionCoordinates(_cm_ydim, True), getCrossSectionCoordinates(_cm_xdim, True), indexing='ij'
                )
                outside = _xm**2 + _ym**2 > 1
            mask = np.broadcast_to(outside[np.newaxis, :, :], coreMapUninhib.shape)
        valid = np.logical_not(mask)

        cellSize = 1 / self.mapDim
        # Voxels just outside the wall are propellant for the distance field. A flat fill there hides the outermost half
        # voxel of propellant from the burning surface, which delays burnout along tapered grains by tens of millimetres.
        # OPENMOTOR_DISABLE_WALL_RING restores the flat fill, for comparison with results made before the ring existed.
        _ringOn = os.environ.get('OPENMOTOR_DISABLE_WALL_RING', '').lower() not in ('1', 'true', 'yes')
        _ring = self._getWallRing(mask) if _ringOn else np.zeros(mask.shape, dtype=bool)
        # An exact Euclidean transform stands in for the fast marching distance. It runs on all cores and differs from
        # the fast marching field by under 0.35 voxel. Masked voxels are not routed around, so a mask that is not
        # convex can shorten distances.
        _useEdt = self.SIGNED_DISTANCE and os.environ.get('OPENMOTOR_EXPERIMENTAL_EDT_DISTANCE', '').lower() in ('1', 'true', 'yes')
        _fmmKey = self._computeFmmCacheKey(coreMapUninhib, cellSize, mask) + ('_edt' if _useEdt else '') + ('' if _ringOn else '_flat')
        regressionMapUninhib = self._loadFmmCache(_fmmKey)
        if regressionMapUninhib is None and _useEdt:
            from mathlib._edt import signed_distance
            regressionMapUninhib = signed_distance((coreMapUninhib > 0.5) | _ring, valid | _ring, fill=self.mapDim)
            regressionMapUninhib *= 2.0 / self.mapDim
            self._saveFmmCache(_fmmKey, regressionMapUninhib)
        if regressionMapUninhib is None:
            # Apply the cylinder mask so skfmm skips outside-cylinder voxels.
            # This is the same pattern the 2-D PerforatedGrain path uses; without
            # it the FMM propagates through ~21 % extra corner voxels and can't
            # terminate early at the cylinder boundary.
            _maskFull = np.ascontiguousarray(mask & ~_ring)
            if self.SIGNED_DISTANCE:
                # With propellant +1 and void -1 the surface lies halfway between voxel centers, so a distance of zero
                # is the real surface. Measuring from the void voxel centers instead puts it half a voxel inside the
                # propellant and makes the first marching cubes levels a staircase.
                _maskedPhi = np.ma.MaskedArray(np.where((coreMapUninhib > 0.5) | _ring, 1.0, -1.0), _maskFull)
            else:
                _maskedPhi = np.ma.MaskedArray(np.where(_ring, 1.0, coreMapUninhib), _maskFull)
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

        # Cubes start at valid voxels and reach at most one voxel into the masked ring, which holds real distances, so
        # the surface meets the casting wall without a spurious crossing from the fill value beyond the ring. Without the
        # ring, masked voxels take the value of the nearest voxel inside the tube instead.
        _mcField = regressionMapUninhib
        if not _ringOn and mask.ndim == 3 and mask.strides[0] == 0:
            _, (_nearestRow, _nearestCol) = ndimage.distance_transform_edt(mask[0], return_indices=True)
            _mcField = np.ascontiguousarray(regressionMapUninhib[:, _nearestRow, _nearestCol])

        # Each level's total area is the sum of its per-layer areas, so the same pass records how the burning
        # surface is distributed along the grain for the mass flux calculation.
        _layerRows = {}

        def _areaPixelsAt(level):
            if not _hasLayerAreas:
                return _marchingAreaPixels(_mcField, valid, level)
            layers = _marchingLayerAreas(_mcField, valid, level)
            total = float(layers.sum())
            if total <= 0:
                return None
            _layerRows[level] = layers
            return total

        # --- Coarse pass (sequential — only ~20 calls, minimal overhead) ---
        _coarsePairs: list = []  # [(level, area_m2), ...]
        _firstFailedCoarse: float | None = None
        for _lvl in _coarseLevels:
            if cancelCheck and cancelCheck():
                raise _GrainSetupCanceled()
            _ap = _areaPixelsAt(_lvl)
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
                            _areaPixelsAt,
                            _fineLevelsSorted,
                        ))
                    if cancelCheck and cancelCheck():
                        raise _GrainSetupCanceled()
                else:
                    _rawAreas = []
                    for lv in _fineLevelsSorted:
                        if cancelCheck and cancelCheck():
                            raise _GrainSetupCanceled()
                        _rawAreas.append(_areaPixelsAt(lv))
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
        self._layerLevels = self._layerAreas = None
        if _layerRows:
            levels = sorted(_layerRows)
            rows = [_layerRows[level] for level in levels]
            for burnout in (_firstFailedCoarse, maxDist):  # No surface is left from these levels on
                if burnout is not None and burnout > levels[-1]:
                    levels.append(burnout)
                    rows.append(np.zeros_like(rows[0]))
            self._layerLevels = np.array(levels)
            self._layerAreas = np.array(rows)
        self._layerOffset = 1 if self.props['inhibitedEnds'].getValue() in ('Top', 'Neither') else 0
        if len(polled) < 2:
            self.faceArea = np.zeros(2)
            self.faceAreaFunc = interpolate.interp1d([0.0, 1.0], [0.0, 0.0])
            self.volumeFunc = None
        else:
            fromAreas = None
            if self.CURVES_FROM_AREA:
                propellantVoxels = int(np.count_nonzero((regressionMapUninhib > 0) & valid))
                fromAreas = self._getCurvesFromAreas(polled, burningArea, self.mapToVolume(propellantVoxels), maxDist)
            if fromAreas is not None:
                _uniformPolled, _smoothVol, self.faceArea, self._areaShortfall = fromAreas
                self.volumeFunc = interpolate.interp1d(
                    _uniformPolled, _smoothVol, fill_value=(_smoothVol[0], 0.0), bounds_error=False
                )
                self.faceAreaFunc = interpolate.interp1d(
                    _uniformPolled, self.faceArea, fill_value=(float(self.faceArea[0]), 0.0), bounds_error=False
                )
            else:
                # Compute volume curve and SA on a uniform grid so that
                # _smoothSeries (SG filter) and np.gradient both see even spacing.
                # The MC-derived adaptive 'polled' has non-uniform spacing that
                # produces derivative artifacts at refinement boundaries.
                _rmSorted = np.sort(regressionMapUninhib[valid].ravel())
                _nVox = len(_rmSorted)
                _nPts = max(len(polled), self.AREA_CURVE_POINTS)
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
                    _portCounts.append(self.mapToArea(self._getAftPortCount([
                        get_core_area_count_at_slice(self._regressionMapF64, self._maskU8, _lvl, _z)
                        for _z in range(_scanAft)
                    ])))
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
        ) + (self._getMaskSignature(),)

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
            fractions = self._getAreaFractionFore(mapDist)
            if fractions is not None:
                zFraction = float(fractions[zPos])
            else:
                v_total = get_volume_count_gt_threshold(regressionMap, maskU8, mapDist)
                v_ahead = get_volume_count_gt_threshold_from_z(regressionMap, maskU8, mapDist, zPos)
                zFraction = v_ahead / v_total if v_total > 0 else 0.0
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

            byArea = self._getPeakFluxByArea(massIn, dTime, regDist, dRegDist, density, startPos, endPos)
            if byArea is not None:
                return byArea

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
                    # zFraction = V_prop(z..end) / V_total, the propellant at or ahead of z that feeds its flow
                    zFraction = int(propSfxReg[z]) / totalReg if totalReg > 0 else 0.0
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
            counts = [get_core_area_count_at_slice(regressionMap, maskU8, mapDist, z) for z in range(scanTo)]
            return self.mapToArea(self._getAftPortCount(counts))

        mapAtReg = np.ma.MaskedArray(self.regressionMap, self.mask) > mapDist

        mapAtReg = mapAtReg.reshape((mapAtReg.shape[0], -1))

        lengthwiseProp = np.sum(mapAtReg, axis=1)
        lengthwiseCores = np.sum(np.logical_not(mapAtReg), axis=1)

        # Use the same max-in-first-5% logic as the Cython path to handle
        # solid end caps.
        zdim = mapAtReg.shape[0]
        scanTo = max(2, zdim // 20)
        aftSliceCores = lengthwiseCores[:scanTo][lengthwiseProp[:scanTo] > 0]
        return self.mapToArea(self._getAftPortCount(aftSliceCores))

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


class ConvertedGrain(Fmm3DGrain):
    """A 3D FMM version of an existing grain. The source grain is copied and supplies the 3D core map through
    'getExtrudedCoreMap' and 'getExtrudedMask', so the original is never modified. Created with 'Grain.convertTo3D'.

    Callers should check 'getSizeIssue' before setup and 'getGeometryIssue' after it, and keep using the source grain
    when either reports a problem.

    'endburnerLength' adds a solid section of that length on top of the source grain, at its head end. The core stops where
    the section starts, so the section is bonded to the grain and burns only where the core opens onto it: the face between
    the two cannot burn because it is propellant against propellant. The head end must be inhibited. The grain's 'length'
    stays the length of the source, and the section is extra."""

    MAX_VOXELS = 250_000_000

    def __init__(self, source: Grain, inhibitedEnds: str, endburnerLength: float = 0.0) -> None:
        super().__init__()
        self.source = copy.deepcopy(source)
        self.endburnerLength = float(endburnerLength)
        if "endburnerLength" in self.source.props:  # So the source's own descriptions say what this grain is
            self.source.props["endburnerLength"].setValue(self.endburnerLength)
        self.geomName = source.geomName
        self.props["diameter"].setValue(source.props["diameter"].getValue())
        self.props["length"].setValue(source.props["length"].getValue())
        self.props["inhibitedEnds"].setValue(inhibitedEnds)
        self.props["massFlux3D"].setValue(True)
        self._setupMapDim = None

    def simulationSetup(self, config, cancelCheck=None, statusCb=None):
        mapDim = config.getProperty("3DmapDim")
        if self._setupMapDim != mapDim:  # Conversion checks set the grain up early, so skip a repeat
            super().simulationSetup(config, cancelCheck, statusCb)
            self._setupMapDim = mapDim

    def generateCoreMap(self) -> None:
        length = self.source.props["length"].getValue()
        self.totalLength = FloatProperty("Length", "m", 0, 10)
        self.totalLength.setValue(length + self.endburnerLength)
        coreSlices = max(1, int(np.ceil(self.lengthToMap(length))))
        self.mapLength = max(coreSlices, int(np.ceil(self.lengthToMap(length + self.endburnerLength))))
        shape = (self.mapLength, self.mapDim, self.mapDim)
        self.mask = np.broadcast_to(self.source.getExtrudedMask(self.mapDim)[None, :, :], shape)
        self.mapX = self.mapY = self.mapZ = None
        coreMap = self.source.getExtrudedCoreMap(coreSlices, self.mapDim)
        if self.mapLength > coreSlices:  # The end-burner section is solid propellant above the core, at the head end
            coreMap = np.concatenate([coreMap, np.ones((self.mapLength - coreSlices, self.mapDim, self.mapDim), dtype=bool)])
        self.coreMap = coreMap

    def getSizeIssue(self, mapDim: int) -> Union[str, None]:
        """Returns a description of why this grain is too large to convert at the given map dimension, or None."""
        diameter = self.props["diameter"].getValue()
        if diameter <= 0:
            return "the grain has no diameter"
        voxels = mapDim * np.ceil(mapDim * (self.props["length"].getValue() + self.endburnerLength) / diameter) * mapDim
        if voxels > self.MAX_VOXELS:
            return "it would need {:.0f} million voxels at 3D map dimension {}".format(voxels / 1e6, mapDim)
        return None

    def getGeometryIssue(self, config) -> Union[str, None]:
        """Compares the converted geometry with the source grain after setup. Returns a description of the first
        disagreement that is larger than the voxel resolution explains, or None when they agree. The end-burner section has
        no source to compare with, so the grain without it is compared instead."""
        if self.endburnerLength > 0:
            if self.props["inhibitedEnds"].getValue() in ("Neither", "Bottom"):
                return "an end-burner section needs the head end of the grain inhibited"
            core = ConvertedGrain(self.source, self.props["inhibitedEnds"].getValue())
            core.simulationSetup(config)
            return core.getGeometryIssue(config)
        mapDim = config.getProperty("3DmapDim")
        volumeTolerance = max(0.08, 6 / mapDim)
        areaTolerance = max(0.15, 12 / mapDim)
        self.source.simulationSetup(config)
        sourceWeb = self.source.getWebLeft(0)
        if sourceWeb <= 0 or self.getWebLeft(0) <= 0:
            return "its web thickness is not positive"
        sourceVolume = self.source.getVolumeAtRegression(0)
        if sourceVolume <= 0:
            return "its volume is not positive"
        fractions = (0.1, 0.3, 0.5, 0.7)  # The first sample avoids the smoothing artifact at zero regression
        areaFractions = fractions[:3]  # Area comes from the slope of a smoothed volume curve, which is too noisy near burnout
        regs = [fraction * sourceWeb for fraction in fractions]
        sourceAreas = [self.source.getSurfaceAreaAtRegression(reg) for reg in regs[:len(areaFractions)]]
        peakArea = max(sourceAreas)
        if peakArea <= 0:
            return "its burning area is not positive"
        # Differences are measured against the initial volume and the peak area, so the small amounts left near
        # burnout cannot make a harmless difference look large.
        deviations = [("web", abs(self.getWebLeft(0) / sourceWeb - 1), volumeTolerance + 0.07),
                      ("volume", abs(self.getVolumeAtRegression(0) / sourceVolume - 1), volumeTolerance)]
        for index, (fraction, reg) in enumerate(zip(fractions, regs)):
            where = "at {:.0f}% of the web".format(100 * fraction)
            if index < len(areaFractions):
                deviations.append(("burning area " + where,
                                   abs(self.getSurfaceAreaAtRegression(reg) - sourceAreas[index]) / peakArea, areaTolerance))
            deviations.append(("volume " + where, abs(self.getVolumeAtRegression(reg) - self.source.getVolumeAtRegression(reg))
                               / sourceVolume, volumeTolerance + 0.02))
        for name, deviation, tolerance in deviations:
            if not (deviation <= tolerance):  # Also catches NaN
                return "the 3D {} is {:.0f}% away from the original (limit {:.0f}%)".format(
                    name, 100 * deviation, 100 * tolerance
                )
        return None

    def getDetailsString(self, lengthUnit: str = "m") -> str:
        details = self.source.getDetailsString(lengthUnit)
        if self.endburnerLength > 0 and "endburnerLength" not in self.source.props:
            section = FloatProperty("Endburner Length", "m", 0, 10)
            section.setValue(self.endburnerLength)
            details += ", endburner {}".format(section.dispFormat(lengthUnit))
        return details

    def getGeometryErrors(self) -> List[SimAlert]:
        return self.source.getGeometryErrors()
