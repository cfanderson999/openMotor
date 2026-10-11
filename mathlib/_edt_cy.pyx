# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Parallel exact Euclidean distance transform of a voxel grid.

The squared distance is separable, so each axis is a set of independent 1D lower-envelope problems
(Felzenszwalb and Huttenlocher, 2012) that OpenMP threads take one line at a time.
"""

cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cython.parallel cimport prange, parallel
import numpy as np

cdef double INF = 1e20


cdef void _line(double* f, double* d, int* v, double* z, int n) noexcept nogil:
    """Squared distance of every position on a line to the nearest finite entry of f, with f[q] the squared cost of q."""
    cdef int k = -1
    cdef int q, p
    cdef double s
    for q in range(n):
        if f[q] >= INF:
            continue
        if k < 0:
            k = 0
            v[0] = q
            z[0] = -INF
            z[1] = INF
            continue
        s = ((f[q] + <double>q * q) - (f[v[k]] + <double>v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + <double>q * q) - (f[v[k]] + <double>v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    if k < 0:
        for q in range(n):
            d[q] = INF
        return
    p = 0
    for q in range(n):
        while z[p + 1] < q:
            p += 1
        d[q] = (<double>q - v[p]) * (<double>q - v[p]) + f[v[p]]


cdef void _transform(cnp.float64_t[:, :, ::1] g) noexcept nogil:
    cdef Py_ssize_t nz = g.shape[0]
    cdef Py_ssize_t ny = g.shape[1]
    cdef Py_ssize_t nx = g.shape[2]
    cdef Py_ssize_t nmax = nz
    cdef Py_ssize_t i, a, b, q
    cdef double* f
    cdef double* d
    cdef int* v
    cdef double* z
    if ny > nmax:
        nmax = ny
    if nx > nmax:
        nmax = nx

    with parallel():
        f = <double*>malloc(nmax * sizeof(double))
        d = <double*>malloc(nmax * sizeof(double))
        v = <int*>malloc(nmax * sizeof(int))
        z = <double*>malloc((nmax + 1) * sizeof(double))
        for i in prange(nz * ny, schedule='static'):
            a = i // ny
            b = i % ny
            for q in range(nx):
                f[q] = g[a, b, q]
            _line(f, d, v, z, <int>nx)
            for q in range(nx):
                g[a, b, q] = d[q]
        free(f)
        free(d)
        free(v)
        free(z)

    with parallel():
        f = <double*>malloc(nmax * sizeof(double))
        d = <double*>malloc(nmax * sizeof(double))
        v = <int*>malloc(nmax * sizeof(int))
        z = <double*>malloc((nmax + 1) * sizeof(double))
        for i in prange(nz * nx, schedule='static'):
            a = i // nx
            b = i % nx
            for q in range(ny):
                f[q] = g[a, q, b]
            _line(f, d, v, z, <int>ny)
            for q in range(ny):
                g[a, q, b] = d[q]
        free(f)
        free(d)
        free(v)
        free(z)

    with parallel():
        f = <double*>malloc(nmax * sizeof(double))
        d = <double*>malloc(nmax * sizeof(double))
        v = <int*>malloc(nmax * sizeof(int))
        z = <double*>malloc((nmax + 1) * sizeof(double))
        for i in prange(ny * nx, schedule='static'):
            a = i // nx
            b = i % nx
            for q in range(nz):
                f[q] = g[q, a, b]
            _line(f, d, v, z, <int>nz)
            for q in range(nz):
                g[q, a, b] = d[q]
        free(f)
        free(d)
        free(v)
        free(z)


def squared_distance(cnp.uint8_t[:, :, ::1] sources):
    """Squared distance, in voxels, from every voxel to the nearest nonzero voxel of `sources`."""
    cdef Py_ssize_t nz = sources.shape[0]
    cdef Py_ssize_t ny = sources.shape[1]
    cdef Py_ssize_t nx = sources.shape[2]
    cdef Py_ssize_t i, j, k
    out = np.empty((nz, ny, nx), dtype=np.float64)
    cdef cnp.float64_t[:, :, ::1] g = out
    for i in prange(nz, nogil=True, schedule='static'):
        for j in range(ny):
            for k in range(nx):
                g[i, j, k] = 0.0 if sources[i, j, k] else INF
    _transform(g)
    return out


def signed_distance(cnp.uint8_t[:, :, ::1] propellant, cnp.uint8_t[:, :, ::1] valid, double fill):
    """Signed distance in voxels to the surface lying half a voxel between propellant and void voxel centers.

    Propellant voxels are positive and void voxels negative. Voxels where `valid` is zero are not sources and take `fill`."""
    cdef Py_ssize_t nz = propellant.shape[0]
    cdef Py_ssize_t ny = propellant.shape[1]
    cdef Py_ssize_t nx = propellant.shape[2]
    cdef Py_ssize_t i, j, k
    sources = np.empty((nz, ny, nx), dtype=np.uint8)
    cdef cnp.uint8_t[:, :, ::1] s = sources
    for i in prange(nz, nogil=True, schedule='static'):
        for j in range(ny):
            for k in range(nx):
                s[i, j, k] = valid[i, j, k] and not propellant[i, j, k]
    toVoid = squared_distance(s)
    for i in prange(nz, nogil=True, schedule='static'):
        for j in range(ny):
            for k in range(nx):
                s[i, j, k] = valid[i, j, k] and propellant[i, j, k]
    toPropellant = squared_distance(s)
    out = np.empty((nz, ny, nx), dtype=np.float64)
    cdef cnp.float64_t[:, :, ::1] o = out
    cdef cnp.float64_t[:, :, ::1] dv = toVoid
    cdef cnp.float64_t[:, :, ::1] dp = toPropellant
    for i in prange(nz, nogil=True, schedule='static'):
        for j in range(ny):
            for k in range(nx):
                if not valid[i, j, k]:
                    o[i, j, k] = fill
                elif propellant[i, j, k]:
                    o[i, j, k] = sqrt(dv[i, j, k]) - 0.5
                else:
                    o[i, j, k] = 0.5 - sqrt(dp[i, j, k])
    return out
