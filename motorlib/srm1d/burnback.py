"""Voxel level-set burnback with an axially varying regression speed, and the per-slab quantities the 1D flow needs.

The grid is cubic with spacing `voxel`; axis 0 is axial. Flow cells are the slabs between adjacent z-slices.
"""

import numpy as np
import skfmm
from scipy import ndimage

from mathlib._levelset import advance_level_set
from mathlib._march import contour_length_by_slice, extend_into_mask, marching_area_pixels_by_layer


class BurnbackGrid:
    def __init__(self, phi: np.ndarray, valid: np.ndarray, voxel: float, advanceMethod: str = 'narrow-band', wallRing=None):
        """`phi` is a signed distance-like field, positive in propellant; `valid` is True inside the casting cylinder.
        `advanceMethod` is one of `ADVANCE_METHODS`. `wallRing` is a 2D mask of the voxels just outside the wall that
        carry real distances, so the surface runs all the way to the wall; without it they hold a flat fill."""
        if advanceMethod not in self.ADVANCE_METHODS:
            raise ValueError(f"advanceMethod must be one of {self.ADVANCE_METHODS}, got {advanceMethod!r}")
        self.advanceMethod = advanceMethod
        self.wallRing = None if wallRing is None else np.asarray(wallRing, dtype=bool)
        self.voxel = float(voxel)
        self.valid = np.broadcast_to(valid, phi.shape).copy()
        self.phi = np.asarray(phi, dtype=np.float64).copy()
        self.elapsed = 0.0
        if not np.all(self.valid == self.valid[0]):
            raise ValueError("valid mask must be identical on every axial slice")
        _, (self._nearestRow, self._nearestCol) = ndimage.distance_transform_edt(~self.valid[0], return_indices=True)
        self.marchValid = self.valid if self.wallRing is None else self.valid | self.wallRing[None]
        maskedRow, maskedCol = np.nonzero(~self.valid[0])
        self._maskedFlat = maskedRow * phi.shape[2] + maskedCol
        self._sourceFlat = self._nearestRow[maskedRow, maskedCol] * phi.shape[2] + self._nearestCol[maskedRow, maskedCol]

    @classmethod
    def fromMask(cls, propellant: np.ndarray, valid: np.ndarray, voxel: float, advanceMethod: str = 'narrow-band',
                 wallRing: bool = True) -> "BurnbackGrid":
        """Build from a boolean propellant mask; the initial surface sits half a voxel from the mask edge. With `wallRing`
        (the default) the masked voxels beside the casting tube count as propellant for the distance field, so the burning
        surface runs all the way to the wall; without it they hold a flat fill and the outermost half voxel of propellant
        has no surface."""
        validFull = np.broadcast_to(valid, propellant.shape)
        ring = None
        solid = propellant & validFull
        marchable = validFull
        if wallRing:
            plane = np.asarray(valid, dtype=bool)
            plane = plane[0] if plane.ndim == 3 else plane
            ring = ndimage.binary_dilation(plane, structure=np.ones((3, 3), dtype=bool)) & ~plane
            solid = solid | np.broadcast_to(ring, propellant.shape)
            marchable = validFull | np.broadcast_to(ring, propellant.shape)
        signed = np.where(solid, 1.0, -1.0)
        masked = np.ma.MaskedArray(signed, ~marchable)
        phi = skfmm.distance(masked, dx=voxel).filled(voxel * max(signed.shape))
        return cls(phi, validFull, voxel, advanceMethod, ring)

    @property
    def nSlabs(self) -> int:
        return self.phi.shape[0] - 1

    @property
    def slabCenters(self) -> np.ndarray:
        return (np.arange(self.nSlabs) + 0.5) * self.voxel

    ADVANCE_METHODS = ('scikit-fmm', 'narrow-band')
    _EDGE_PAD = 4
    _BAND_FACTOR = 2.0

    def advance(self, slabSpeed: np.ndarray, dt: float) -> None:
        """Move the surface along its normal by slabSpeed[k] [m/s] for dt seconds.

        'scikit-fmm' solves the arrival time over the whole grid and re-derives the distance beside the surface from the
        zero crossings, which moves those voxels slightly even when dt is tiny. 'narrow-band' solves only the band
        around the surface, in parallel, and seeds it from the field itself, so it is an exact shift as dt goes to zero
        and agrees with the iso-levels of the initial field that the 3D FMM grains use. The axial ends are mirrored, so
        the solve sees a reflecting wall."""
        if not np.any((self.phi > 0) & self.marchValid):
            return
        sliceSpeed = np.interp(np.arange(self.phi.shape[0]), np.arange(self.nSlabs) + 0.5, slabSpeed)
        floor = max(1e-6 * float(sliceSpeed.max()), 1e-12)
        if self.advanceMethod == 'narrow-band':
            self.phi = advance_level_set(self.phi, self.marchValid[0], np.maximum(sliceSpeed, floor), self.voxel, dt)
        else:
            self._advanceScikitFmm(sliceSpeed, floor, dt)
        self.elapsed += dt

    def _advanceScikitFmm(self, sliceSpeed: np.ndarray, floor: float, dt: float) -> None:
        pad = self._EDGE_PAD
        padded = np.pad(self.phi, ((pad, pad), (0, 0), (0, 0)), mode='edge')
        paddedValid = np.pad(self.marchValid, ((pad, pad), (0, 0), (0, 0)), mode='edge')
        speed = np.broadcast_to(np.pad(np.maximum(sliceSpeed, floor), pad, mode='edge')[:, None, None], padded.shape)
        arrival = skfmm.travel_time(
            np.ma.MaskedArray(padded, ~paddedValid), np.ma.MaskedArray(speed, ~paddedValid), dx=self.voxel, order=2,
            narrow=self._BAND_FACTOR * dt,
        )[pad:-pad]
        local = speed[pad:-pad]
        shifted = self.phi - local * dt
        inBand = ~np.ma.getmaskarray(arrival)
        advanced = np.where(self.phi < 0, shifted, np.where(inBand, (arrival.filled(0.0) - dt) * local, shifted))
        self.phi = np.where(self.marchValid, advanced, self.phi)

    def voidArea(self) -> np.ndarray:
        """Open cross-sectional area at each z-slice [m^2]."""
        return np.count_nonzero((self.phi <= 0) & self.valid, axis=(1, 2)) * self.voxel ** 2

    def slabBurnArea(self) -> np.ndarray:
        """Iso-surface area inside each slab [m^2].

        Masked voxels take the value of the nearest valid voxel so the casting wall never reads as a surface."""
        extended = self.phi if self.wallRing is not None else extend_into_mask(self.phi, self._maskedFlat, self._sourceFlat)
        return marching_area_pixels_by_layer(extended, self.valid, 0.0) * self.voxel ** 2

    def wettedPerimeter(self) -> np.ndarray:
        """Bore contour length at each z-slice [m]."""
        return contour_length_by_slice(self.phi, 0.0) * self.voxel

    def flowGeometry(self):
        """(aPort, dHyd, burnLength) per slab, SI units. burnLength is burning area per unit axial length."""
        openArea = self.voidArea()
        perimeter = self.wettedPerimeter()
        aPort = 0.5 * (openArea[:-1] + openArea[1:])
        wetted = 0.5 * (perimeter[:-1] + perimeter[1:])
        casting = float(np.count_nonzero(self.valid[0])) * self.voxel ** 2
        castingPerimeter = 2.0 * np.sqrt(np.pi * casting)
        dHyd = np.where(wetted > 0, 4.0 * aPort / np.maximum(wetted, 1e-12), 4.0 * aPort / castingPerimeter)
        return aPort, dHyd, self.slabBurnArea() / self.voxel
