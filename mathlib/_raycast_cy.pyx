# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Orthographic ray casting of a stack of grains, burned to a depth that may differ from slice to slice, and the blur
that prepares the stack for it. See `mathlib/_raycast.py` for what the arguments mean.

Rays run in parallel over the pixels with OpenMP; every kernel works on scalars and typed memoryviews, so it holds no GIL.
"""

from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
import numpy as np

cdef enum:  # kinds of surface a ray can hit
    C_NOTHING = 0
    C_BORE = 1
    C_CUT = 2
    C_WALL = 3
    C_END = 4

cdef enum:  # solids that are drawn
    C_GRAIN = 0
    C_CORE = 1
    C_BOTH = 2

cdef enum:  # what a point of the volume is
    S_OUTSIDE = 0
    S_CASING = 1
    S_CORE = 2
    S_PROPELLANT = 3
    S_AWAY = 4  # S_AWAY + axis: beyond the cutting plane of that axis

NOTHING, BORE, CUT, WALL, END = C_NOTHING, C_BORE, C_CUT, C_WALL, C_END
GRAIN, CORE, BOTH = C_GRAIN, C_CORE, C_BOTH

cdef int REFINEMENTS = 6


cdef inline double _clamp(double value, double low, double high) noexcept nogil:
    return low if value < low else (high if value > high else value)


cdef inline double _interpolate(float[:, :, ::1] volume, double z, double y, double x) noexcept nogil:
    """`volume` at the point (z, y, x), in voxels, linearly interpolated between voxel centres."""
    cdef Py_ssize_t nz = volume.shape[0], ny = volume.shape[1], nx = volume.shape[2]
    cdef double fz = _clamp(z - 0.5, 0.0, nz - 1.0)
    cdef double fy = _clamp(y - 0.5, 0.0, ny - 1.0)
    cdef double fx = _clamp(x - 0.5, 0.0, nx - 1.0)
    cdef Py_ssize_t iz = <Py_ssize_t>fz, iy = <Py_ssize_t>fy, ix = <Py_ssize_t>fx
    cdef Py_ssize_t jz = iz + 1 if iz + 1 < nz else nz - 1
    cdef Py_ssize_t jy = iy + 1 if iy + 1 < ny else ny - 1
    cdef Py_ssize_t jx = ix + 1 if ix + 1 < nx else nx - 1
    cdef double dz = fz - iz, dy = fy - iy, dx = fx - ix
    cdef double near = (volume[iz, iy, ix] * (1 - dx) + volume[iz, iy, jx] * dx) * (1 - dy) \
        + (volume[iz, jy, ix] * (1 - dx) + volume[iz, jy, jx] * dx) * dy
    cdef double far = (volume[jz, iy, ix] * (1 - dx) + volume[jz, iy, jx] * dx) * (1 - dy) \
        + (volume[jz, jy, ix] * (1 - dx) + volume[jz, jy, jx] * dx) * dy
    return near * (1 - dz) + far * dz


cdef inline double _across(float[:, ::1] plane, double y, double x) noexcept nogil:
    """`plane` (y, x) at the point (y, x), in voxels, linearly interpolated between voxel centres."""
    cdef Py_ssize_t ny = plane.shape[0], nx = plane.shape[1]
    cdef double fy = _clamp(y - 0.5, 0.0, ny - 1.0)
    cdef double fx = _clamp(x - 0.5, 0.0, nx - 1.0)
    cdef Py_ssize_t iy = <Py_ssize_t>fy, ix = <Py_ssize_t>fx
    cdef Py_ssize_t jy = iy + 1 if iy + 1 < ny else ny - 1
    cdef Py_ssize_t jx = ix + 1 if ix + 1 < nx else nx - 1
    cdef double dy = fy - iy, dx = fx - ix
    return (plane[iy, ix] * (1 - dx) + plane[iy, jx] * dx) * (1 - dy) + (plane[jy, ix] * (1 - dx) + plane[jy, jx] * dx) * dy


cdef inline double _depth(double[::1] depths, double z) noexcept nogil:
    """The depth of the core at the point `z`, in voxels along the stack, interpolated between slice centres."""
    cdef Py_ssize_t n = depths.shape[0]
    cdef double f = _clamp(z - 0.5, 0.0, n - 1.0)
    cdef Py_ssize_t i = <Py_ssize_t>f
    cdef Py_ssize_t j = i + 1 if i + 1 < n else n - 1
    return depths[i] * (1 - (f - i)) + depths[j] * (f - i)


cdef inline int _classify(float[:, :, ::1] regression, float[:, ::1] casing, double[::1] depths, double lz, double ly,
                          double lx, double z, double y, double x) noexcept nogil:
    if z > lz:
        return S_AWAY
    if y > ly:
        return S_AWAY + 1
    if x > lx:
        return S_AWAY + 2
    if _across(casing, y, x) < 0.5:
        return S_CASING
    if _interpolate(regression, z, y, x) <= _depth(depths, z):
        return S_CORE
    return S_PROPELLANT


cdef inline bint _solid(int state, bint drawGrain, bint drawCore) noexcept nogil:
    return (state == S_PROPELLANT and drawGrain) or (state == S_CORE and drawCore)


cdef inline void _slab(double start, double direction, double size, double* tIn, double* tOut) noexcept nogil:
    """Narrow the span [tIn, tOut] of a ray to where it is between 0 and `size` along one axis."""
    cdef double t1, t2, early, late
    if fabs(direction) < 1e-9:
        if start < 0.0 or start > size:
            tOut[0] = -1.0
    else:
        t1 = -start / direction
        t2 = (size - start) / direction
        early, late = (t1, t2) if t1 < t2 else (t2, t1)
        tIn[0] = tIn[0] if tIn[0] > early else early
        tOut[0] = tOut[0] if tOut[0] < late else late


def castCore(float[:, :, ::1] regression, float[:, ::1] casing, double[::1] depths, double[::1] right, double[::1] up,
             double[::1] toward, double[::1] limits, int shown, double[::1] offset, double pixelSize, int rows, int cols,
             double step):
    cdef Py_ssize_t nz = regression.shape[0], ny = regression.shape[1], nx = regression.shape[2]
    kindArray = np.zeros((rows, cols), np.uint8)
    sliceArray = np.zeros((rows, cols), np.int32)
    shadeArray = np.zeros((rows, cols), np.float32)
    cdef unsigned char[:, ::1] kind = kindArray
    cdef int[:, ::1] hitSlice = sliceArray
    cdef float[:, ::1] shade = shadeArray

    cdef double cz = 0.5 * nz, cy = 0.5 * ny, cx = 0.5 * nx
    cdef double radius = 0.5 * sqrt(<double>nz * nz + <double>ny * ny + <double>nx * nx) + 2.0
    cdef double dz = -toward[0], dy = -toward[1], dx = -toward[2]
    cdef double kz = toward[0] + 0.7 * up[0] - 0.5 * right[0]
    cdef double ky = toward[1] + 0.7 * up[1] - 0.5 * right[1]
    cdef double kx = toward[2] + 0.7 * up[2] - 0.5 * right[2]
    cdef double keyLength = sqrt(kz * kz + ky * ky + kx * kx)
    kz, ky, kx = kz / keyLength, ky / keyLength, kx / keyLength
    cdef double lz = limits[0], ly = limits[1], lx = limits[2]
    cdef bint drawGrain = shown != C_CORE
    cdef bint drawCore = shown != C_GRAIN

    cdef Py_ssize_t p, i, j
    cdef int r, state, before, inside
    cdef bint surface
    cdef double across, down, sz, sy, sx, tIn, tOut, t, z, y, x, low, high, middle
    cdef double n0, n1, n2, g0, g1, g2, length, lit

    with nogil:
        for p in prange(<Py_ssize_t>rows * cols, schedule='dynamic', chunksize=64):
            i = p // cols
            j = p % cols
            across = (j + 0.5 - 0.5 * cols) * pixelSize + offset[0]
            down = (0.5 * rows - i - 0.5) * pixelSize + offset[1]
            sz = cz + right[0] * across + up[0] * down + toward[0] * radius
            sy = cy + right[1] * across + up[1] * down + toward[1] * radius
            sx = cx + right[2] * across + up[2] * down + toward[2] * radius
            tIn = 0.0
            tOut = 2.0 * radius
            _slab(sz, dz, <double>nz, &tIn, &tOut)
            _slab(sy, dy, <double>ny, &tIn, &tOut)
            _slab(sx, dx, <double>nx, &tIn, &tOut)
            before = S_OUTSIDE
            t = tIn + 1e-3
            while t < tOut:
                z = sz + dz * t
                y = sy + dy * t
                x = sx + dx * t
                state = _classify(regression, casing, depths, lz, ly, lx, z, y, x)
                if _solid(state, drawGrain, drawCore):
                    if before != S_OUTSIDE:
                        low = t - step
                        high = t
                        for r in range(REFINEMENTS):
                            middle = 0.5 * (low + high)
                            inside = _classify(regression, casing, depths, lz, ly, lx, sz + dz * middle,
                                               sy + dy * middle, sx + dx * middle)
                            if _solid(inside, drawGrain, drawCore):
                                high = middle
                            else:
                                low = middle
                        z = sz + dz * high
                        y = sy + dy * high
                        x = sx + dx * high
                    surface = state == S_CORE
                    n0, n1, n2 = toward[0], toward[1], toward[2]
                    if before >= S_AWAY:
                        n0 = n1 = n2 = 0.0
                        if before == S_AWAY:
                            n0 = 1.0
                        elif before == S_AWAY + 1:
                            n1 = 1.0
                        else:
                            n2 = 1.0
                        kind[i, j] = C_BORE if surface else C_CUT
                    elif before == S_OUTSIDE:
                        n0 = -1.0 if dz > 0.0 else (1.0 if dz < 0.0 else 0.0)
                        n1 = n2 = 0.0
                        kind[i, j] = C_BORE if surface else C_END
                    elif before == S_CASING:
                        n0 = 0.0
                        n1 = y - cy
                        n2 = x - cx
                        length = sqrt(n1 * n1 + n2 * n2)
                        length = length if length > 1e-9 else 1e-9
                        n1 = n1 / length
                        n2 = n2 / length
                        kind[i, j] = C_BORE if surface else C_WALL
                    else:
                        g0 = _interpolate(regression, z + 0.75, y, x) - _interpolate(regression, z - 0.75, y, x)
                        g1 = _interpolate(regression, z, y + 0.75, x) - _interpolate(regression, z, y - 0.75, x)
                        g2 = _interpolate(regression, z, y, x + 0.75) - _interpolate(regression, z, y, x - 0.75)
                        length = sqrt(g0 * g0 + g1 * g1 + g2 * g2)
                        if length > 1e-6:
                            if surface:
                                n0, n1, n2 = g0 / length, g1 / length, g2 / length
                            else:
                                n0, n1, n2 = -g0 / length, -g1 / length, -g2 / length
                        kind[i, j] = C_BORE
                    hitSlice[i, j] = <int>_clamp(<double>(<Py_ssize_t>z), 0.0, nz - 1.0)
                    lit = 0.42 + 0.38 * max(0.0, n0 * kz + n1 * ky + n2 * kx) \
                        + 0.20 * max(0.0, n0 * toward[0] + n1 * toward[1] + n2 * toward[2])
                    shade[i, j] = lit if lit < 1.0 else 1.0
                    break
                before = state
                t = t + step
    return kindArray, sliceArray, shadeArray


def smoothVolume(float[:, :, ::1] volume, int width, double edge):
    """`volume` (slice, y, x) blurred for drawing: along the slices by three passes of a running average of `width` slices,
    which stand in for a Gaussian, and then across each slice by the taps (`edge`, 1 - 2 * `edge`, `edge`) in y and in x.
    The outermost values are extended past the ends."""
    cdef Py_ssize_t nz = volume.shape[0], ny = volume.shape[1], nx = volume.shape[2]
    alongArray = np.empty((nz, ny, nx), np.float32)
    outArray = np.empty((nz, ny, nx), np.float32)
    cdef float[:, :, ::1] along = alongArray
    cdef float[:, :, ::1] out = outArray
    cdef Py_ssize_t half = width // 2
    cdef double middle = 1.0 - 2.0 * edge
    cdef Py_ssize_t p, y, x, z, i, k, lo, hi
    cdef int sweep
    cdef double total
    cdef double* first
    cdef double* second
    cdef double* source
    cdef double* target
    cdef double* rows

    with nogil, parallel():
        first = <double*>malloc(nz * sizeof(double))
        second = <double*>malloc(nz * sizeof(double))
        for p in prange(ny * nx, schedule='static'):
            y = p // nx
            x = p % nx
            for i in range(nz):
                first[i] = volume[i, y, x]
            for sweep in range(3):
                source = first if sweep % 2 == 0 else second
                target = second if sweep % 2 == 0 else first
                total = 0.0
                for k in range(-half, half + 1):
                    total = total + source[min(max(k, 0), nz - 1)]
                for i in range(nz):
                    target[i] = total / width
                    hi = i + half + 1
                    lo = i - half
                    total = total + source[hi if hi < nz - 1 else nz - 1] - source[lo if lo > 0 else 0]
            for i in range(nz):
                along[i, y, x] = <float>second[i]
        free(first)
        free(second)

    with nogil, parallel():
        rows = <double*>malloc(ny * nx * sizeof(double))
        for z in prange(nz, schedule='static'):
            for y in range(ny):
                for x in range(nx):
                    rows[y * nx + x] = edge * along[z, y, x - 1 if x > 0 else 0] + middle * along[z, y, x] \
                        + edge * along[z, y, x + 1 if x < nx - 1 else nx - 1]
            for y in range(ny):
                for x in range(nx):
                    out[z, y, x] = <float>(edge * rows[(y - 1 if y > 0 else 0) * nx + x] + middle * rows[y * nx + x]
                                           + edge * rows[(y + 1 if y < ny - 1 else ny - 1) * nx + x])
        free(rows)
    return outArray
