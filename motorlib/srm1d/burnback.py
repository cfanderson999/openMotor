"""Voxel level-set burnback with an axially varying regression speed, and the per-slab quantities the 1D flow needs.

The grid is cubic with spacing `voxel`; axis 0 is axial. Flow cells are the slabs between adjacent z-slices.
"""

import numpy as np
import skfmm
from scipy import ndimage
from skimage import measure

from mathlib._march import marching_area_pixels_by_layer


class BurnbackGrid:
    def __init__(self, phi: np.ndarray, valid: np.ndarray, voxel: float):
        """`phi` is a signed distance-like field, positive in propellant; `valid` is True inside the casting cylinder."""
        self.voxel = float(voxel)
        self.valid = np.broadcast_to(valid, phi.shape).copy()
        self.phi = np.asarray(phi, dtype=np.float64).copy()
        self.elapsed = 0.0
        if not np.all(self.valid == self.valid[0]):
            raise ValueError("valid mask must be identical on every axial slice")
        _, (self._nearestRow, self._nearestCol) = ndimage.distance_transform_edt(~self.valid[0], return_indices=True)

    @classmethod
    def fromMask(cls, propellant: np.ndarray, valid: np.ndarray, voxel: float) -> "BurnbackGrid":
        """Build from a boolean propellant mask; the initial surface sits half a voxel from the mask edge."""
        validFull = np.broadcast_to(valid, propellant.shape)
        signed = np.where(propellant & validFull, 1.0, -1.0)
        masked = np.ma.MaskedArray(signed, ~validFull)
        phi = skfmm.distance(masked, dx=voxel).filled(voxel * max(signed.shape))
        return cls(phi, validFull, voxel)

    @property
    def nSlabs(self) -> int:
        return self.phi.shape[0] - 1

    @property
    def slabCenters(self) -> np.ndarray:
        return (np.arange(self.nSlabs) + 0.5) * self.voxel

    _EDGE_PAD = 4
    _BAND_FACTOR = 2.0

    def advance(self, slabSpeed: np.ndarray, dt: float) -> None:
        """Move the surface along its normal by slabSpeed[k] [m/s] for dt seconds.

        The axial ends are mirrored by edge-padding so the fast marching solve sees a reflecting wall."""
        if not np.any((self.phi > 0) & self.valid):
            return
        pad = self._EDGE_PAD
        sliceSpeed = np.interp(np.arange(self.phi.shape[0]), np.arange(self.nSlabs) + 0.5, slabSpeed)
        floor = max(1e-6 * float(sliceSpeed.max()), 1e-12)
        padded = np.pad(self.phi, ((pad, pad), (0, 0), (0, 0)), mode='edge')
        paddedValid = np.pad(self.valid, ((pad, pad), (0, 0), (0, 0)), mode='edge')
        speed = np.broadcast_to(np.pad(np.maximum(sliceSpeed, floor), pad, mode='edge')[:, None, None], padded.shape)
        arrival = skfmm.travel_time(
            np.ma.MaskedArray(padded, ~paddedValid), np.ma.MaskedArray(speed, ~paddedValid), dx=self.voxel, order=2,
            narrow=self._BAND_FACTOR * dt,
        )[pad:-pad]
        local = speed[pad:-pad]
        shifted = self.phi - local * dt
        inBand = ~np.ma.getmaskarray(arrival)
        advanced = np.where(self.phi < 0, shifted, np.where(inBand, (arrival.filled(0.0) - dt) * local, shifted))
        self.phi = np.where(self.valid, advanced, self.phi)
        self.elapsed += dt

    def voidArea(self) -> np.ndarray:
        """Open cross-sectional area at each z-slice [m^2]."""
        return np.count_nonzero((self.phi <= 0) & self.valid, axis=(1, 2)) * self.voxel ** 2

    def slabBurnArea(self) -> np.ndarray:
        """Iso-surface area inside each slab [m^2].

        Masked voxels take the value of the nearest valid voxel so the casting wall never reads as a surface."""
        extended = self.phi[:, self._nearestRow, self._nearestCol]
        return marching_area_pixels_by_layer(extended, self.valid, 0.0) * self.voxel ** 2

    def wettedPerimeter(self) -> np.ndarray:
        """Bore contour length at each z-slice [m]."""
        out = np.zeros(self.phi.shape[0])
        for k, plane in enumerate(self.phi):
            if plane.min() < 0 < plane.max():
                contours = measure.find_contours(plane, 0.0)
                out[k] = sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in contours) * self.voxel
        return out

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
