# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Narrow band fast marching advance of a signed distance level set whose burn speed varies along axis 0.

Only voxels within the band around the surface are visited, so the cost follows the surface area instead of the grid
volume. Axis 0 is split into blocks that OpenMP threads solve independently. Each block carries a halo of replicated
neighbouring slices wider than the band, so every block reproduces the unsplit solution. The grid ends are mirrored by
replicating the end slice, which gives a reflecting wall.
"""

cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
from libc.math cimport sqrt, ceil
from cython.parallel cimport prange
import numpy as np

cdef double INF = 1e30


cdef struct Heap:
    double* key
    long long* idx
    long long n
    long long cap


cdef void _heap_init(Heap* h) noexcept nogil:
    h.cap = 4096
    h.n = 0
    h.key = <double*>malloc(h.cap * sizeof(double))
    h.idx = <long long*>malloc(h.cap * sizeof(long long))


cdef void _heap_free(Heap* h) noexcept nogil:
    free(h.key)
    free(h.idx)


cdef void _heap_push(Heap* h, double key, long long idx) noexcept nogil:
    cdef long long j, p
    cdef double tk
    cdef long long ti
    if h.n >= h.cap:
        h.cap *= 2
        h.key = <double*>realloc(h.key, h.cap * sizeof(double))
        h.idx = <long long*>realloc(h.idx, h.cap * sizeof(long long))
    j = h.n
    h.key[j] = key
    h.idx[j] = idx
    h.n += 1
    while j > 0:
        p = (j - 1) >> 1
        if h.key[p] > h.key[j]:
            tk = h.key[p]; h.key[p] = h.key[j]; h.key[j] = tk
            ti = h.idx[p]; h.idx[p] = h.idx[j]; h.idx[j] = ti
            j = p
        else:
            break


cdef void _heap_pop(Heap* h, double* key, long long* idx) noexcept nogil:
    cdef long long j = 0, l, r, m
    cdef double tk
    cdef long long ti
    key[0] = h.key[0]
    idx[0] = h.idx[0]
    h.n -= 1
    h.key[0] = h.key[h.n]
    h.idx[0] = h.idx[h.n]
    while True:
        l = 2 * j + 1
        if l >= h.n:
            break
        r = l + 1
        m = l if (r >= h.n or h.key[l] <= h.key[r]) else r
        if h.key[m] < h.key[j]:
            tk = h.key[m]; h.key[m] = h.key[j]; h.key[j] = tk
            ti = h.idx[m]; h.idx[m] = h.idx[j]; h.idx[j] = ti
            j = m
        else:
            break


cdef inline int _axis_candidate(double* T, unsigned char* fr, long long i, long long stride, bint hasM1, bint hasP1,
                                bint hasM2, bint hasP2, double* u, double* c, int m) noexcept nogil:
    """Adds the upwind value along one axis: second order when the node behind the nearer neighbour is also frozen
    and not later than it, first order otherwise."""
    cdef double best = INF
    cdef double best2 = INF
    cdef double t
    if hasM1 and fr[i - stride]:
        best = T[i - stride]
        if hasM2 and fr[i - 2 * stride]:
            best2 = T[i - 2 * stride]
    if hasP1 and fr[i + stride]:
        t = T[i + stride]
        if t < best:
            best = t
            best2 = INF
            if hasP2 and fr[i + 2 * stride]:
                best2 = T[i + 2 * stride]
    if best < INF:
        if best2 <= best:
            u[m] = (4.0 * best - best2) / 3.0
            c[m] = 2.25
        else:
            u[m] = best
            c[m] = 1.0
        return m + 1
    return m


cdef double _solve_node(double* T, unsigned char* fr, long long i, int z, int y, int x, int nz, int ny, int nx,
                        double h) noexcept nogil:
    """Arrival time at node i from its frozen neighbours: the root of sum c (t - u)^2 = h^2, taking only the axes
    whose upwind value lies below the root."""
    cdef double u[3]
    cdef double c[3]
    cdef int m = 0
    cdef long long xy = <long long>nx * ny
    cdef int a, j, k
    cdef double tu, tc, t, A, B, C, disc
    m = _axis_candidate(T, fr, i, xy, z >= 1, z < nz - 1, z >= 2, z < nz - 2, u, c, m)
    m = _axis_candidate(T, fr, i, nx, y >= 1, y < ny - 1, y >= 2, y < ny - 2, u, c, m)
    m = _axis_candidate(T, fr, i, 1, x >= 1, x < nx - 1, x >= 2, x < nx - 2, u, c, m)
    if m == 0:
        return INF
    for a in range(1, m):
        j = a
        while j > 0 and u[j - 1] > u[j]:
            tu = u[j - 1]; u[j - 1] = u[j]; u[j] = tu
            tc = c[j - 1]; c[j - 1] = c[j]; c[j] = tc
            j -= 1
    t = u[0] + h / sqrt(c[0])
    A = 0.0
    B = 0.0
    C = -h * h
    for k in range(m):
        if k > 0 and t <= u[k]:
            break
        A += c[k]
        B += c[k] * u[k]
        C += c[k] * u[k] * u[k]
        if k > 0:
            disc = B * B - A * C
            if disc < 0.0:
                break
            t = (B + sqrt(disc)) / A
    return t


cdef void _solve_block(double* ph, cnp.uint8_t[:, ::1] valid, double* speed, int nz, int ny, int nx, double dx,
                       double narrow, double skin, double* T, unsigned char* fr) noexcept nogil:
    """Marches the arrival time from the zero level set into the propellant side (phi > 0) of one block.

    The first layer of propellant voxels, those beside a void voxel, is seeded with its own signed distance over the
    local speed. The field is already a distance, so the seeds make the advance a pure shift as dt goes to zero."""
    cdef long long xy = <long long>nx * ny
    cdef Heap heap
    cdef double t, key, tn
    cdef long long i, j
    cdef int z, y, x, zz, yy, xx, k
    cdef bint beside
    _heap_init(&heap)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not valid[y, x]:
                    continue
                i = z * xy + y * nx + x
                if ph[i] <= 0.0 or ph[i] > skin:
                    continue
                beside = False
                if z > 0 and ph[i - xy] <= 0.0:
                    beside = True
                elif z < nz - 1 and ph[i + xy] <= 0.0:
                    beside = True
                elif y > 0 and valid[y - 1, x] and ph[i - nx] <= 0.0:
                    beside = True
                elif y < ny - 1 and valid[y + 1, x] and ph[i + nx] <= 0.0:
                    beside = True
                elif x > 0 and valid[y, x - 1] and ph[i - 1] <= 0.0:
                    beside = True
                elif x < nx - 1 and valid[y, x + 1] and ph[i + 1] <= 0.0:
                    beside = True
                if beside:
                    t = ph[i] / speed[z]
                    T[i] = t
                    _heap_push(&heap, t, i)
    while heap.n > 0:
        _heap_pop(&heap, &key, &i)
        if fr[i] or key > T[i]:
            continue
        if key > narrow:
            break
        fr[i] = 1
        x = <int>(i % nx)
        y = <int>((i // nx) % ny)
        z = <int>(i // xy)
        for k in range(6):
            zz = z
            yy = y
            xx = x
            j = i
            if k == 0:
                zz -= 1; j -= xy
            elif k == 1:
                zz += 1; j += xy
            elif k == 2:
                yy -= 1; j -= nx
            elif k == 3:
                yy += 1; j += nx
            elif k == 4:
                xx -= 1; j -= 1
            else:
                xx += 1; j += 1
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if fr[j] or not valid[yy, xx] or ph[j] <= 0.0:
                continue
            tn = _solve_node(T, fr, j, zz, yy, xx, nz, ny, nx, dx / speed[zz])
            if tn < T[j]:
                T[j] = tn
                _heap_push(&heap, tn, j)
    _heap_free(&heap)


cdef void _advance_block(cnp.float64_t[:, :, ::1] phi, cnp.uint8_t[:, ::1] valid, cnp.float64_t[::1] speed,
                         cnp.float64_t[:, :, ::1] out, int lo0, int hi0, int halo, double dx, double dt, double narrow,
                         double skin) noexcept nogil:
    cdef int Z = <int>phi.shape[0]
    cdef int ny = <int>phi.shape[1]
    cdef int nx = <int>phi.shape[2]
    cdef int lo = lo0 - halo
    cdef int nz = hi0 + halo - lo
    cdef long long xy = <long long>nx * ny
    cdef long long n = nz * xy
    cdef double* ph = <double*>malloc(n * sizeof(double))
    cdef double* sp = <double*>malloc(nz * sizeof(double))
    cdef double* T = <double*>malloc(n * sizeof(double))
    cdef unsigned char* fr = <unsigned char*>malloc(n)
    cdef int k, s, y, x, z
    cdef long long i, q
    cdef double p, f
    for k in range(nz):
        s = lo + k
        if s < 0:
            s = 0
        elif s >= Z:
            s = Z - 1
        sp[k] = speed[s]
        for y in range(ny):
            for x in range(nx):
                ph[k * xy + y * nx + x] = phi[s, y, x]
    for q in range(n):
        T[q] = INF
    memset(fr, 0, n)
    _solve_block(ph, valid, sp, nz, ny, nx, dx, narrow, skin, T, fr)
    for z in range(lo0, hi0):
        k = z - lo
        f = speed[z]
        for y in range(ny):
            for x in range(nx):
                p = phi[z, y, x]
                if not valid[y, x]:
                    out[z, y, x] = p
                else:
                    i = k * xy + y * nx + x
                    if p >= 0.0 and fr[i]:
                        out[z, y, x] = (T[i] - dt) * f
                    else:
                        out[z, y, x] = p - f * dt
    free(ph)
    free(sp)
    free(T)
    free(fr)


def advance(cnp.float64_t[:, :, ::1] phi not None, cnp.uint8_t[:, ::1] valid not None, cnp.float64_t[::1] speed not None,
            double dx, double dt, double narrow, int nblocks, int halo, double skin, int threads):
    """New level set after burning for `dt`. `speed[k]` is the regression speed of slice k. Inside the band of arrival
    times up to `narrow` the surface follows the fast marching arrival time; elsewhere it shifts by speed times dt.
    Voxels outside `valid` are unchanged."""
    cdef int Z = <int>phi.shape[0]
    out = np.empty((phi.shape[0], phi.shape[1], phi.shape[2]), dtype=np.float64)
    cdef cnp.float64_t[:, :, ::1] o = out
    cdef int b, lo0, hi0
    if nblocks < 1:
        nblocks = 1
    for b in prange(nblocks, nogil=True, schedule='dynamic', num_threads=threads):
        lo0 = <int>((<long long>Z * b) // nblocks)
        hi0 = <int>((<long long>Z * (b + 1)) // nblocks)
        _advance_block(phi, valid, speed, o, lo0, hi0, halo, dx, dt, narrow, skin)
    return out
