"""Python wrapper for _levelset_cy: narrow band level set advance with a speed that varies along axis 0."""

import os

import numpy as np

try:
    from ._levelset_cy import advance as _advance_cy
except ImportError as exc:
    raise ImportError(
        "mathlib._levelset_cy is unavailable. Build extensions with: python setup.py build_ext --inplace"
    ) from exc

# Arrival times are solved out to this many time steps, which is wider than the band the new surface can reach.
BAND_STEPS = 2.0
# Voxels beside the band that a block must also see so splitting along axis 0 changes nothing: the second order
# stencil reaches two voxels past the node being solved, the rest is margin.
HALO_MARGIN = 4
# Fast marching only seeds from voxels this close to the surface, measured in voxels of distance.
SEED_VOXELS = 3.0
# Slices per block along axis 0. The block layout depends on the grid alone, never on the thread count, because the
# solutions agree between layouts only to within the rim of the band, and results must not vary between machines.
BLOCK_SLICES = 48


def advance_level_set(phi, valid, sliceSpeed, voxel, dt, threads=None):
    """Level set `phi` (positive in propellant) after burning for `dt` seconds at `sliceSpeed[k]` m/s on slice k.

    `valid` is the 2D mask of the casting tube. Blocks along axis 0 are solved in parallel on `threads` threads."""
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    speed = np.ascontiguousarray(sliceSpeed, dtype=np.float64)
    narrow = BAND_STEPS * dt
    halo = int(np.ceil(narrow * float(speed.max()) / voxel)) + HALO_MARGIN
    blocks = max(1, phi.shape[0] // max(BLOCK_SLICES, 2 * halo))
    threads = int(max(1, min(threads or os.cpu_count() or 1, blocks)))
    return _advance_cy(phi, np.ascontiguousarray(valid, dtype=np.uint8), speed, float(voxel), float(dt), narrow, blocks,
                       halo, SEED_VOXELS * voxel, threads)
