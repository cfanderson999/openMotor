# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

cimport numpy as cnp
from cython.parallel cimport prange
import numpy as np


def first_last_prop_indices(cnp.float32_t[:, :, ::1] regression,
                            cnp.uint8_t[:, :, ::1] mask,
                            cnp.float64_t threshold):
    """Return first and last z-indices containing any valid voxel > threshold."""
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef bint found = False
    cdef Py_ssize_t first_idx = -1
    cdef Py_ssize_t last_idx = -1

    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x] == 0 and regression[z, y, x] > threshold:
                    if not found:
                        first_idx = z
                        found = True
                    last_idx = z
                    break
            if found and last_idx == z:
                break

    if not found:
        return -1, -1
    return first_idx, last_idx


def first_port_core_count(cnp.float32_t[:, :, ::1] regression,
                          cnp.uint8_t[:, :, ::1] mask,
                          cnp.float64_t threshold):
    """Return core-area voxel count at first z-slice that still has propellant."""
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef Py_ssize_t prop_count
    cdef Py_ssize_t valid_count

    for z in range(zdim):
        prop_count = 0
        valid_count = 0
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x] == 0:
                    valid_count += 1
                    if regression[z, y, x] > threshold:
                        prop_count += 1
        if prop_count > 0:
            return valid_count - prop_count

    return 0


def volume_count_gt_threshold(cnp.float32_t[:, :, ::1] regression,
                              cnp.uint8_t[:, :, ::1] mask,
                              cnp.float64_t threshold):
    """Count valid voxels where regression > threshold."""
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef Py_ssize_t count = 0

    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x] == 0 and regression[z, y, x] > threshold:
                    count += 1

    return count


def volume_count_gt_threshold_from_z(cnp.float32_t[:, :, ::1] regression,
                                     cnp.uint8_t[:, :, ::1] mask,
                                     cnp.float64_t threshold,
                                     Py_ssize_t z_start):
    """Count valid voxels where regression > threshold for z in [z_start, end)."""
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef Py_ssize_t count = 0

    if z_start < 0:
        z_start = 0
    if z_start >= zdim:
        return 0

    for z in range(z_start, zdim):
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x] == 0 and regression[z, y, x] > threshold:
                    count += 1

    return count


def core_area_count_at_slice(cnp.float32_t[:, :, ::1] regression,
                             cnp.uint8_t[:, :, ::1] mask,
                             cnp.float64_t threshold,
                             Py_ssize_t z):
    """Count valid voxels at z-slice where regression <= threshold."""
    cdef Py_ssize_t y, x
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef Py_ssize_t count = 0

    if z < 0 or z >= regression.shape[0]:
        return 0

    for y in range(ydim):
        for x in range(xdim):
            if mask[z, y, x] == 0 and regression[z, y, x] <= threshold:
                count += 1

    return count


def core_area_profile(cnp.float32_t[:, :, ::1] regression,
                      cnp.uint8_t[:, :, ::1] mask,
                      cnp.float64_t threshold,
                      Py_ssize_t z_start,
                      Py_ssize_t z_end):
    """Return core-area voxel counts for slices z_start..z_end (inclusive).

    out[i] = number of valid unmasked voxels at slice (z_start + i) where
    regression <= threshold.  Used as a cheap proxy for peak-flux position
    selection before running expensive marching-cubes calls.
    """
    cdef Py_ssize_t i, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]

    if z_start < 0:
        z_start = 0
    if z_end >= zdim:
        z_end = zdim - 1
    if z_end < z_start:
        return np.zeros(0, dtype=np.int64)

    cdef Py_ssize_t n = z_end - z_start + 1
    cdef cnp.ndarray[cnp.int64_t, ndim=1] counts_np = np.zeros(n, dtype=np.int64)
    cdef cnp.int64_t[::1] counts = counts_np

    for i in prange(n, nogil=True, schedule='static'):
        for y in range(ydim):
            for x in range(xdim):
                if mask[i + z_start, y, x] == 0 and regression[i + z_start, y, x] <= threshold:
                    counts[i] += 1

    return counts_np


def massflux_slice_suffix_arrays(cnp.float32_t[:, :, ::1] regression,
                                 cnp.uint8_t[:, :, ::1] mask,
                                 cnp.float64_t threshold_start,
                                 cnp.float64_t threshold_end):
    """Return (core_counts, prop_suffix_start, prop_suffix_end) over z.

    core_counts[z] is the number of valid voxels at slice z where value <= threshold_start.
    prop_suffix_start[z] is the propellant voxel count (> threshold_start) in slices [z, end).
    prop_suffix_end[z] is the propellant voxel count (> threshold_end) in slices [z, end).
    """
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t zdim = regression.shape[0]
    cdef Py_ssize_t ydim = regression.shape[1]
    cdef Py_ssize_t xdim = regression.shape[2]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] core_np = np.zeros(zdim, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] prop_start_np = np.zeros(zdim, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] prop_end_np = np.zeros(zdim, dtype=np.int64)
    cdef cnp.int64_t[::1] core = core_np
    cdef cnp.int64_t[::1] prop_start = prop_start_np
    cdef cnp.int64_t[::1] prop_end = prop_end_np
    cdef cnp.int64_t run_start = 0
    cdef cnp.int64_t run_end = 0
    cdef cnp.float32_t val

    for z in prange(zdim, nogil=True, schedule='static'):
        core[z] = 0
        prop_start[z] = 0
        prop_end[z] = 0
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x] == 0:
                    val = regression[z, y, x]
                    if val <= threshold_start:
                        core[z] += 1
                    if val > threshold_start:
                        prop_start[z] += 1
                    if val > threshold_end:
                        prop_end[z] += 1

    # In-place suffix sums for [z, end) lookups.
    for z in range(zdim - 1, -1, -1):
        run_start += prop_start[z]
        run_end += prop_end[z]
        prop_start[z] = run_start
        prop_end[z] = run_end

    return core_np, prop_start_np, prop_end_np
