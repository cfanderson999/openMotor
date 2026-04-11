# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Parallel Marching Cubes surface area computation using OpenMP.

Uses skimage's exact CASESCLASSIC lookup table so results are numerically
identical to skimage.measure.marching_cubes + mesh_surface_area when the
same table is in effect.  Only the surface area scalar is produced (no mesh),
which allows prange over z-slices with a scalar reduction.
"""

cimport numpy as cnp
from libc.math cimport sqrt
from cython.parallel cimport prange
import numpy as np

# ---------------------------------------------------------------------------
# Module-level tables – initialised from skimage once at import time.
# Stored as plain numpy arrays; typed memoryview locals are created per call.
# ---------------------------------------------------------------------------
_TRI_TABLE  = None   # (256, 16) int32  – CASESCLASSIC from skimage
_EDGE_DZ0   = None   # (12,)      int32  – Z offset of edge start corner
_EDGE_DY0   = None   # (12,)      int32
_EDGE_DX0   = None   # (12,)      int32
_EDGE_DZ1   = None   # (12,)      int32  – Z offset of edge end corner
_EDGE_DY1   = None   # (12,)      int32
_EDGE_DX1   = None   # (12,)      int32


def _init_tables():
    global _TRI_TABLE, _EDGE_DZ0, _EDGE_DY0, _EDGE_DX0, _EDGE_DZ1, _EDGE_DY1, _EDGE_DX1
    import skimage.measure._marching_cubes_lewiner as mcl

    tri = mcl._to_array(mcl.mcluts.CASESCLASSIC).astype(np.int32)
    _TRI_TABLE = np.ascontiguousarray(tri)

    ex = np.array(mcl.EDGETORELATIVEPOSX, dtype=np.int32)  # (12, 2)
    ey = np.array(mcl.EDGETORELATIVEPOSY, dtype=np.int32)
    ez = np.array(mcl.EDGETORELATIVEPOSZ, dtype=np.int32)

    _EDGE_DX0 = np.ascontiguousarray(ex[:, 0])
    _EDGE_DX1 = np.ascontiguousarray(ex[:, 1])
    _EDGE_DY0 = np.ascontiguousarray(ey[:, 0])
    _EDGE_DY1 = np.ascontiguousarray(ey[:, 1])
    _EDGE_DZ0 = np.ascontiguousarray(ez[:, 0])
    _EDGE_DZ1 = np.ascontiguousarray(ez[:, 1])


_init_tables()


# ---------------------------------------------------------------------------
# Inner cube-area function – pure C, no GIL, no Python objects.
# ---------------------------------------------------------------------------
cdef inline double _cube_area(
    cnp.float64_t[:, :, ::1] grid,
    cnp.uint8_t[:, :, ::1]   vmask,
    double level,
    int iz, int iy, int ix,
    int[:, ::1] tri_table,
    int[::1] edz0, int[::1] edy0, int[::1] edx0,
    int[::1] edz1, int[::1] edy1, int[::1] edx1,
    int step,
) nogil:
    """Return marching-cubes area contribution for the cube at (iz,iy,ix) with
    corner spacing *step* voxels.  step=1 gives the original unit-cube result;
    step>1 matches skimage's step_size behaviour (corners are step voxels apart,
    vertex coordinates are in original voxel space, area is in original voxel²).
    """
    cdef int c_idx, t, e, vi
    cdef int edges[3]
    cdef double vx[3], vy[3], vz[3]
    cdef double area, s0, s1, tp
    cdef double ax, ay, az, bx, by, bz, cx, cy, cz
    cdef int dz0, dy0, dx0, dz1, dy1, dx1

    # skimage's marching_cubes mask operates as a per-cube enable grid.
    # A cube is considered only if mask[iz, iy, ix] is true (not all 8 corners).
    if not vmask[iz, iy, ix]:
        return 0.0

    # Build case index: bit i is set when corner i > level.
    # Corner ordering matches skimage CASESCLASSIC (Paulbourke/Lorensen-Cline):
    #  bit 0: (ix,      iy,      iz     )   bit 1: (ix+step, iy,      iz     )
    #  bit 2: (ix+step, iy+step, iz     )   bit 3: (ix,      iy+step, iz     )
    #  bit 4: (ix,      iy,      iz+step)   bit 5: (ix+step, iy,      iz+step)
    #  bit 6: (ix+step, iy+step, iz+step)   bit 7: (ix,      iy+step, iz+step)
    # grid is indexed [iz, iy, ix] → map accordingly.
    c_idx = 0
    if grid[iz,        iy,        ix       ] > level: c_idx |= 1
    if grid[iz,        iy,        ix + step] > level: c_idx |= 2
    if grid[iz,        iy + step, ix + step] > level: c_idx |= 4
    if grid[iz,        iy + step, ix       ] > level: c_idx |= 8
    if grid[iz + step, iy,        ix       ] > level: c_idx |= 16
    if grid[iz + step, iy,        ix + step] > level: c_idx |= 32
    if grid[iz + step, iy + step, ix + step] > level: c_idx |= 64
    if grid[iz + step, iy + step, ix       ] > level: c_idx |= 128

    if c_idx == 0 or c_idx == 255:
        return 0.0

    area = 0.0
    t = 0
    while t < 15:
        e = tri_table[c_idx, t]
        if e < 0:
            break

        edges[0] = e
        edges[1] = tri_table[c_idx, t + 1]
        edges[2] = tri_table[c_idx, t + 2]
        t += 3

        # Interpolate the 3 triangle vertices along their respective edges.
        # Corners are *step* voxels apart; vertex coordinates are in original voxel space.
        for vi in range(3):
            e    = edges[vi]
            dz0  = edz0[e];  dy0 = edy0[e];  dx0 = edx0[e]
            dz1  = edz1[e];  dy1 = edy1[e];  dx1 = edx1[e]
            s0   = grid[iz + dz0 * step, iy + dy0 * step, ix + dx0 * step]
            s1   = grid[iz + dz1 * step, iy + dy1 * step, ix + dx1 * step]
            tp   = (level - s0) / (s1 - s0) if s1 != s0 else 0.5
            vx[vi] = ix + dx0 * step + tp * (dx1 - dx0) * step
            vy[vi] = iy + dy0 * step + tp * (dy1 - dy0) * step
            vz[vi] = iz + dz0 * step + tp * (dz1 - dz0) * step

        # Triangle area = 0.5 * |cross(v1-v0, v2-v0)|
        ax = vx[1] - vx[0];  ay = vy[1] - vy[0];  az = vz[1] - vz[0]
        bx = vx[2] - vx[0];  by = vy[2] - vy[0];  bz = vz[2] - vz[0]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area += 0.5 * sqrt(cx*cx + cy*cy + cz*cz)

    return area


# ---------------------------------------------------------------------------
# Public Python-callable entry point.
# ---------------------------------------------------------------------------
def mc_surface_area(
    cnp.float64_t[:, :, ::1] grid      not None,
    cnp.uint8_t[:, :, ::1]   valid_mask not None,
    double level,
    int step_size = 1,
):
    """
    Compute iso-surface area for *level* on *grid* using Marching Cubes.

    Parameters
    ----------
    grid       : (nz, ny, nx) float64 C-contiguous – the scalar field.
    valid_mask : (nz, ny, nx) uint8  C-contiguous – 1 where voxel is valid.
    level      : iso-level threshold.

    Returns
    -------
    float – surface area in voxel (pixel) units, identical to
            ``skimage.measure.mesh_surface_area(*marching_cubes(grid, level, step_size=step_size)[:2])``
            (within floating-point rounding at ambiguous cube cases).
    """
    cdef int nz = grid.shape[0]
    cdef int ny = grid.shape[1]
    cdef int nx = grid.shape[2]

    if step_size < 1:
        step_size = 1

    # Need at least one full cube (accounting for step_size).
    if nz <= step_size or ny <= step_size or nx <= step_size:
        return 0.0

    # Obtain typed views of the module-level tables.
    cdef int[:, ::1] tri_t = _TRI_TABLE
    cdef int[::1]    edz0  = _EDGE_DZ0
    cdef int[::1]    edy0  = _EDGE_DY0
    cdef int[::1]    edx0  = _EDGE_DX0
    cdef int[::1]    edz1  = _EDGE_DZ1
    cdef int[::1]    edy1  = _EDGE_DY1
    cdef int[::1]    edx1  = _EDGE_DX1

    # Number of cubes in each dimension when striding by step_size.
    # Using step-count indices (iz_i, iy_i, ix_i) lets us use single-argument
    # range/prange, which Cython maps to pure C loops inside nogil blocks.
    cdef int nsteps_z = (nz - 1) // step_size
    cdef int nsteps_y = (ny - 1) // step_size
    cdef int nsteps_x = (nx - 1) // step_size
    cdef double total = 0.0
    cdef int iz_i, iy_i, ix_i

    for iz_i in prange(nsteps_z, nogil=True, schedule='static'):
        for iy_i in range(nsteps_y):
            for ix_i in range(nsteps_x):
                total += _cube_area(
                    grid, valid_mask, level,
                    iz_i * step_size, iy_i * step_size, ix_i * step_size,
                    tri_t,
                    edz0, edy0, edx0,
                    edz1, edy1, edx1,
                    step_size,
                )

    return total
