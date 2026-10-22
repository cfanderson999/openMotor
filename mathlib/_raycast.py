"""Orthographic ray casting of a stack of grains, burned to a depth that may differ from slice to slice.

The volume is a stack of regression maps (slice, y, x). A point is propellant while the map, interpolated between voxel
centres, exceeds the depth the core has burned to on its slice, and core once it does not; points outside the casting tube,
whose section is given as a smooth field to be crossed at one half, are neither. Planes can cut away the part of the volume
beyond them, and the grain, the core or both can be shown as solids. A ray reports what it hit and how brightly the surface
is lit; colours are for the caller to choose.

The kernels are in `_raycast_cy`, which is built with the other extensions by `python setup.py build_ext --inplace`.
"""

import numpy as np

from ._raycast_cy import BORE, BOTH, CORE, CUT, END, GRAIN, NOTHING, WALL  # noqa: F401
from ._raycast_cy import castCore as _castCore
from ._raycast_cy import smoothVolume as _smoothVolume


def castCore(regression, casing, depths, right, up, toward, limits, shown, offset, pixelSize, rows, cols, step):
    """Cast `rows` x `cols` rays along -`toward` through the volume.

    `regression` is (slice, y, x) float32, `casing` (y, x) float32 is 1 inside the casting tube and 0 outside it, with the
    edge blurred, `depths` the depth of the core on each slice. `right`, `up` and `toward` are unit vectors in (slice, y, x)
    order that point right and up on the screen and at the camera. Material is cut away where a coordinate exceeds its entry
    of `limits`, in voxels. `shown` is GRAIN, CORE or BOTH, the solids that are drawn. The picture is centred `offset` voxels
    along `right` and `up` from the centre of the volume. A pixel covers `pixelSize` voxels and the rays advance `step` voxels
    at a time.

    Returns the kind of surface each ray hit (NOTHING, BORE, CUT, WALL or END), the slice it hit, and how brightly it is
    lit, from 0 to 1. The surface of the core, wherever it is drawn, is a BORE.
    """
    vector = lambda values: np.ascontiguousarray(values, dtype=np.float64)
    return _castCore(np.ascontiguousarray(regression, dtype=np.float32), np.ascontiguousarray(casing, dtype=np.float32),
                     vector(depths), vector(right), vector(up), vector(toward), vector(limits), int(shown), vector(offset),
                     float(pixelSize), int(rows), int(cols), float(step))


def smoothVolume(volume, width, edge):
    """`volume` (slice, y, x) blurred for drawing: along the slices by three passes of a running average of `width` slices,
    which stand in for a Gaussian, and then across each slice by the taps (`edge`, 1 - 2 * `edge`, `edge`) in y and in x.
    The outermost values are extended past the ends."""
    return _smoothVolume(np.ascontiguousarray(volume, dtype=np.float32), int(width), float(edge))
