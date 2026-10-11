"""Python wrapper for _edt_cy: parallel exact Euclidean distance transform."""

import numpy as np

try:
    from ._edt_cy import squared_distance as _squared_distance_cy
    from ._edt_cy import signed_distance as _signed_distance_cy
except ImportError as exc:
    raise ImportError(
        "mathlib._edt_cy is unavailable. Build extensions with: python setup.py build_ext --inplace"
    ) from exc


def squared_distance(sources):
    """Squared distance, in voxels, from every voxel to the nearest True voxel of the 3D array `sources`."""
    return _squared_distance_cy(np.ascontiguousarray(sources, dtype=np.uint8))


def signed_distance(propellant, valid, fill):
    """Signed distance in voxels to the surface half a voxel between propellant and void voxel centers, positive in
    propellant. Voxels outside `valid` are not distance sources and take `fill`."""
    return _signed_distance_cy(
        np.ascontiguousarray(propellant, dtype=np.uint8),
        np.ascontiguousarray(valid, dtype=np.uint8),
        float(fill),
    )
