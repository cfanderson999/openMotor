# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Parallel mesh voxelizer using Möller-Trumbore ray casting + OpenMP.

For each (ix, iy) column of the output voxel grid a Z-directed ray is cast.
Triangle intersections are found via Möller-Trumbore; even-odd counting
determines inside/outside at each grid point along the column.

prange over ix columns gives near-linear scaling across CPU cores.
Thread-local hit buffers are indexed by openmp.omp_get_thread_num().
"""

cimport numpy as cnp
from libc.math cimport fabs
from libc.stdlib cimport qsort
from cython.parallel cimport prange, threadid
cimport openmp
import numpy as np

cdef double _EPSILON = 1e-10


# ---------------------------------------------------------------------------
# qsort comparator for doubles (ascending).
# ---------------------------------------------------------------------------
cdef int _cmp_double(const void* a, const void* b) noexcept nogil:  # noexcept: already declared
    cdef double da = (<double*>a)[0]
    cdef double db = (<double*>b)[0]
    if da < db:
        return -1
    if da > db:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Möller-Trumbore Z-ray cast for one column.
# Two variants:
#   _cast_z_ray          – iterate all n_faces (legacy, used as fallback).
#   _cast_z_ray_indexed  – iterate only the faces listed in cell_faces[0..n],
#                          used by the spatial-index path.
# Returns the number of hits written to hit_buf (capacity >= n_faces).
# ---------------------------------------------------------------------------
cdef int _cast_z_ray(
    double ox, double oy,
    cnp.float64_t* verts,
    cnp.int32_t*   faces,
    cnp.float64_t* face_bbox,
    int n_faces,
    double* hit_buf,
) noexcept nogil:
    cdef int fi, n_hits, i0, i1, i2
    cdef double v0x, v0y, v0z
    cdef double e1x, e1y, e1z, e2x, e2y, e2z
    cdef double hx, hy, a, f
    cdef double sx, sy, sz, u, v
    cdef double qx, qy, qz, t_hit

    n_hits = 0
    for fi in range(n_faces):
        if ox < face_bbox[fi * 4    ] or ox > face_bbox[fi * 4 + 1]:
            continue
        if oy < face_bbox[fi * 4 + 2] or oy > face_bbox[fi * 4 + 3]:
            continue

        i0 = faces[fi * 3];      i1 = faces[fi * 3 + 1];  i2 = faces[fi * 3 + 2]
        v0x = verts[i0 * 3];     v0y = verts[i0 * 3 + 1]; v0z = verts[i0 * 3 + 2]
        e1x = verts[i1 * 3] - v0x
        e1y = verts[i1 * 3 + 1] - v0y
        e1z = verts[i1 * 3 + 2] - v0z
        e2x = verts[i2 * 3] - v0x
        e2y = verts[i2 * 3 + 1] - v0y
        e2z = verts[i2 * 3 + 2] - v0z

        # h = cross(D=(0,0,1), e2) = (-e2y, e2x, 0)
        hx = -e2y;  hy = e2x
        a  = e1x * hx + e1y * hy
        if fabs(a) < _EPSILON:
            continue

        f  = 1.0 / a
        sx = ox - v0x;  sy = oy - v0y;  sz = 0.0 - v0z
        u  = f * (sx * hx + sy * hy)
        if u < 0.0 or u > 1.0:
            continue

        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v  = f * qz
        if v < 0.0 or u + v > 1.0:
            continue

        t_hit = f * (e2x * qx + e2y * qy + e2z * qz)
        hit_buf[n_hits] = t_hit
        n_hits += 1

    return n_hits


cdef int _cast_z_ray_indexed(
    double ox, double oy,
    cnp.float64_t* verts,
    cnp.int32_t*   faces,
    cnp.float64_t* face_bbox,
    cnp.int32_t*   cell_faces,   # face indices for this spatial-grid cell
    int            n_cell_faces,
    double*        hit_buf,
) noexcept nogil:
    """Like _cast_z_ray but only tests the faces listed in cell_faces."""
    cdef int k, fi, n_hits, i0, i1, i2
    cdef double v0x, v0y, v0z
    cdef double e1x, e1y, e1z, e2x, e2y, e2z
    cdef double hx, hy, a, f
    cdef double sx, sy, sz, u, v
    cdef double qx, qy, qz, t_hit

    n_hits = 0
    for k in range(n_cell_faces):
        fi = cell_faces[k]
        if ox < face_bbox[fi * 4    ] or ox > face_bbox[fi * 4 + 1]:
            continue
        if oy < face_bbox[fi * 4 + 2] or oy > face_bbox[fi * 4 + 3]:
            continue

        i0 = faces[fi * 3];      i1 = faces[fi * 3 + 1];  i2 = faces[fi * 3 + 2]
        v0x = verts[i0 * 3];     v0y = verts[i0 * 3 + 1]; v0z = verts[i0 * 3 + 2]
        e1x = verts[i1 * 3] - v0x
        e1y = verts[i1 * 3 + 1] - v0y
        e1z = verts[i1 * 3 + 2] - v0z
        e2x = verts[i2 * 3] - v0x
        e2y = verts[i2 * 3 + 1] - v0y
        e2z = verts[i2 * 3 + 2] - v0z

        hx = -e2y;  hy = e2x
        a  = e1x * hx + e1y * hy
        if fabs(a) < _EPSILON:
            continue

        f  = 1.0 / a
        sx = ox - v0x;  sy = oy - v0y;  sz = 0.0 - v0z
        u  = f * (sx * hx + sy * hy)
        if u < 0.0 or u > 1.0:
            continue

        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v  = f * qz
        if v < 0.0 or u + v > 1.0:
            continue

        t_hit = f * (e2x * qx + e2y * qy + e2z * qz)
        hit_buf[n_hits] = t_hit
        n_hits += 1

    return n_hits


# ---------------------------------------------------------------------------
# Process one (ix, iy) column: ray, sort hits, even-odd fill.
# Two variants matching the two ray-cast variants above.
# ---------------------------------------------------------------------------
cdef void _fill_column(
    double ox, double oy,
    cnp.float64_t* verts,
    cnp.int32_t*   faces,
    cnp.float64_t* face_bbox,
    int n_faces,
    cnp.float64_t* z_coords,
    int nz,
    cnp.uint8_t* out_col,
    cnp.float64_t* hit_buf,
) noexcept nogil:
    cdef int n_hits, hi, inside, iz, r, w
    n_hits = _cast_z_ray(ox, oy, verts, faces, face_bbox, n_faces, hit_buf)
    if n_hits == 0:
        return
    qsort(hit_buf, n_hits, sizeof(double), _cmp_double)

    # Collapse duplicate/near-duplicate intersections at shared edges/vertices.
    w = 0
    for r in range(n_hits):
        if w == 0 or fabs(hit_buf[r] - hit_buf[w - 1]) > 1e-9:
            hit_buf[w] = hit_buf[r]
            w += 1
    n_hits = w

    hi     = 0
    inside = 0
    for iz in range(nz):
        while hi < n_hits and hit_buf[hi] < z_coords[iz]:
            inside ^= 1
            hi     += 1
        out_col[iz] = <cnp.uint8_t>inside


cdef void _fill_column_indexed(
    double ox, double oy,
    cnp.float64_t* verts,
    cnp.int32_t*   faces,
    cnp.float64_t* face_bbox,
    cnp.int32_t*   sg_starts,
    cnp.int32_t*   sg_face_list,
    int cell,
    cnp.float64_t* z_coords,
    int nz,
    cnp.uint8_t*   out_col,
    cnp.float64_t* hit_buf,
) noexcept nogil:
    """Indexed variant: only ray-tests faces belonging to spatial-grid cell."""
    cdef int n_hits, hi, inside, iz, r, w
    cdef int cell_start   = sg_starts[cell]
    cdef int n_cell_faces = sg_starts[cell + 1] - cell_start
    n_hits = _cast_z_ray_indexed(
        ox, oy, verts, faces, face_bbox,
        sg_face_list + cell_start, n_cell_faces,
        hit_buf,
    )
    if n_hits == 0:
        return
    qsort(hit_buf, n_hits, sizeof(double), _cmp_double)

    w = 0
    for r in range(n_hits):
        if w == 0 or fabs(hit_buf[r] - hit_buf[w - 1]) > 1e-9:
            hit_buf[w] = hit_buf[r]
            w += 1
    n_hits = w

    hi     = 0
    inside = 0
    for iz in range(nz):
        while hi < n_hits and hit_buf[hi] < z_coords[iz]:
            inside ^= 1
            hi     += 1
        out_col[iz] = <cnp.uint8_t>inside


# ---------------------------------------------------------------------------
# Public Python-callable voxelizer.
# ---------------------------------------------------------------------------
def voxelize_mesh(
    cnp.float64_t[:, ::1] vertices  not None,
    cnp.int32_t[:,   ::1] faces     not None,
    cnp.float64_t[::1]    x_coords  not None,
    cnp.float64_t[::1]    y_coords  not None,
    cnp.float64_t[::1]    z_coords  not None,
):
    """
    Voxelize a closed triangle mesh using Z-directed ray casting.

    Returns (nx, ny, nz) uint8 array – 1 where the grid point is inside the mesh.
    """
    cdef int nx      = x_coords.shape[0]
    cdef int ny      = y_coords.shape[0]
    cdef int nz      = z_coords.shape[0]
    cdef int n_faces = faces.shape[0]
    cdef int n_verts = vertices.shape[0]

    if n_faces == 0 or n_verts == 0:
        return np.zeros((nx, ny, nz), dtype=np.uint8)

    # Per-face XY AABB [xmin, xmax, ymin, ymax].
    face_bbox_np = np.empty((n_faces, 4), dtype=np.float64)
    cdef cnp.float64_t[:, ::1] face_bbox = face_bbox_np
    cdef int fi, i0c, i1c, i2c
    cdef double xmin, xmax, ymin, ymax
    for fi in range(n_faces):
        i0c = faces[fi, 0];  i1c = faces[fi, 1];  i2c = faces[fi, 2]
        xmin = vertices[i0c, 0]
        xmax = xmin
        if vertices[i1c, 0] < xmin: xmin = vertices[i1c, 0]
        if vertices[i2c, 0] < xmin: xmin = vertices[i2c, 0]
        if vertices[i1c, 0] > xmax: xmax = vertices[i1c, 0]
        if vertices[i2c, 0] > xmax: xmax = vertices[i2c, 0]
        ymin = vertices[i0c, 1]
        ymax = ymin
        if vertices[i1c, 1] < ymin: ymin = vertices[i1c, 1]
        if vertices[i2c, 1] < ymin: ymin = vertices[i2c, 1]
        if vertices[i1c, 1] > ymax: ymax = vertices[i1c, 1]
        if vertices[i2c, 1] > ymax: ymax = vertices[i2c, 1]
        face_bbox[fi, 0] = xmin;  face_bbox[fi, 1] = xmax
        face_bbox[fi, 2] = ymin;  face_bbox[fi, 3] = ymax

    # Per-thread hit buffers.
    cdef int max_threads
    with nogil:
        max_threads = openmp.omp_get_max_threads()
    hit_bufs_np = np.zeros((max_threads, n_faces), dtype=np.float64)
    cdef cnp.float64_t[:, ::1] hit_bufs = hit_bufs_np

    # Output.
    out_np = np.zeros((nx, ny, nz), dtype=np.uint8)
    cdef cnp.uint8_t[:, :, ::1] out = out_np

    # Raw pointers for the nogil inner call.
    cdef cnp.float64_t* verts_ptr     = &vertices[0, 0]
    cdef cnp.int32_t*   faces_ptr     = &faces[0, 0]
    cdef cnp.float64_t* fbbox_ptr     = &face_bbox[0, 0]
    cdef cnp.float64_t* xp            = &x_coords[0]
    cdef cnp.float64_t* yp            = &y_coords[0]
    cdef cnp.float64_t* zp            = &z_coords[0]
    cdef cnp.float64_t* hbufs_ptr     = &hit_bufs[0, 0]

    cdef int ix, iy, tid

    # -----------------------------------------------------------------------
    # 2D spatial index: bucket faces into an SG×SG grid over the mesh XY
    # extent.  Each column then only ray-tests the ~n_faces/SG² faces in its
    # cell instead of all n_faces → 3–10× speedup for meshes with >500 faces.
    # Below the threshold the index-build overhead exceeds the scan savings.
    # -----------------------------------------------------------------------
    cdef int SG = 32
    cdef int _SI_THRESHOLD = 500
    cdef bint use_si = n_faces >= _SI_THRESHOLD

    # Typed-memoryview handles for spatial-index arrays (null when not used).
    cdef cnp.int32_t[::1] _sg_starts_view
    cdef cnp.int32_t[::1] _sg_face_list_view
    cdef cnp.int32_t[::1] _col_x_cell
    cdef cnp.int32_t[::1] _col_y_cell

    # Raw pointers for nogil dispatch (set only when use_si).
    cdef cnp.int32_t* sg_starts_ptr    = NULL
    cdef cnp.int32_t* sg_face_list_ptr = NULL

    # Loop variables for index build.
    cdef int gx, gy, gx1, gx2, gy1, gy2, k
    cdef double sg_x_min, sg_y_min, sg_x_cell, sg_y_cell

    if use_si:
        sg_x_min  = float(face_bbox_np[:, 0].min())
        sg_y_min  = float(face_bbox_np[:, 2].min())
        sg_x_cell = (float(face_bbox_np[:, 1].max()) - sg_x_min + 1e-10) / SG
        sg_y_cell = (float(face_bbox_np[:, 3].max()) - sg_y_min + 1e-10) / SG

        # Count pass: how many faces touch each cell?
        counts_np = np.zeros(SG * SG, dtype=np.int32)
        for fi in range(n_faces):
            gx1 = max(0,      <int>((face_bbox[fi, 0] - sg_x_min) / sg_x_cell))
            gx2 = min(SG - 1, <int>((face_bbox[fi, 1] - sg_x_min) / sg_x_cell))
            gy1 = max(0,      <int>((face_bbox[fi, 2] - sg_y_min) / sg_y_cell))
            gy2 = min(SG - 1, <int>((face_bbox[fi, 3] - sg_y_min) / sg_y_cell))
            for gx in range(gx1, gx2 + 1):
                for gy in range(gy1, gy2 + 1):
                    counts_np[gx * SG + gy] += 1

        # Prefix-sum → CSR starts array.
        sg_starts_np = np.zeros(SG * SG + 1, dtype=np.int32)
        sg_starts_np[1:] = np.cumsum(counts_np)

        # Fill pass: insert face indices into the CSR list.
        sg_face_list_np = np.empty(int(sg_starts_np[SG * SG]), dtype=np.int32)
        fill_idx_np = sg_starts_np[:SG * SG].copy()
        for fi in range(n_faces):
            gx1 = max(0,      <int>((face_bbox[fi, 0] - sg_x_min) / sg_x_cell))
            gx2 = min(SG - 1, <int>((face_bbox[fi, 1] - sg_x_min) / sg_x_cell))
            gy1 = max(0,      <int>((face_bbox[fi, 2] - sg_y_min) / sg_y_cell))
            gy2 = min(SG - 1, <int>((face_bbox[fi, 3] - sg_y_min) / sg_y_cell))
            for gx in range(gx1, gx2 + 1):
                for gy in range(gy1, gy2 + 1):
                    k = gx * SG + gy
                    sg_face_list_np[fill_idx_np[k]] = fi
                    fill_idx_np[k] += 1

        # Map each axis coord to its spatial-grid bucket index.
        _x_arr = np.asarray(x_coords)
        _y_arr = np.asarray(y_coords)
        _col_x_cell = np.ascontiguousarray(
            np.clip(((_x_arr - sg_x_min) / sg_x_cell).astype(np.int32), 0, SG - 1),
            dtype=np.int32,
        )
        _col_y_cell = np.ascontiguousarray(
            np.clip(((_y_arr - sg_y_min) / sg_y_cell).astype(np.int32), 0, SG - 1),
            dtype=np.int32,
        )

        # Bind typed memoryviews and capture raw pointers for the prange region.
        _sg_starts_view    = sg_starts_np
        _sg_face_list_view = sg_face_list_np
        sg_starts_ptr    = &_sg_starts_view[0]
        sg_face_list_ptr = &_sg_face_list_view[0]

    if use_si:
        for ix in prange(nx, nogil=True, schedule='static'):
            tid = threadid()
            for iy in range(ny):
                _fill_column_indexed(
                    xp[ix], yp[iy],
                    verts_ptr, faces_ptr, fbbox_ptr,
                    sg_starts_ptr, sg_face_list_ptr,
                    _col_x_cell[ix] * SG + _col_y_cell[iy],
                    zp, nz,
                    &out[ix, iy, 0],
                    hbufs_ptr + tid * n_faces,
                )
    else:
        for ix in prange(nx, nogil=True, schedule='static'):
            tid = threadid()
            for iy in range(ny):
                _fill_column(
                    xp[ix], yp[iy],
                    verts_ptr, faces_ptr, fbbox_ptr, n_faces,
                    zp, nz,
                    &out[ix, iy, 0],
                    hbufs_ptr + tid * n_faces,
                )

    return out_np
