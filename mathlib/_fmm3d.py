import numpy as np

try:
    from ._fmm3d_cy import (
        first_last_prop_indices,
        first_port_core_count,
        volume_count_gt_threshold,
        volume_count_gt_threshold_from_z,
        core_area_count_at_slice,
        core_area_profile,
        massflux_slice_suffix_arrays,
    )
except ImportError as exc:
    raise ImportError(
        "mathlib._fmm3d_cy is unavailable. Build extensions with: "
        "python setup.py build_ext --inplace"
    ) from exc


def _as_float32_3d(arr):
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_float64_3d(arr):
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_u8_3d(mask):
    # OpenMotor masks use True for outside/invalid, False for valid voxels.
    return np.ascontiguousarray(mask, dtype=np.uint8)


def get_first_last_prop_indices(regression, mask, threshold):
    return first_last_prop_indices(_as_float32_3d(regression), _as_u8_3d(mask), float(threshold))


def get_first_port_core_count(regression, mask, threshold):
    return first_port_core_count(_as_float32_3d(regression), _as_u8_3d(mask), float(threshold))


def get_volume_count_gt_threshold(regression, mask, threshold):
    return volume_count_gt_threshold(_as_float32_3d(regression), _as_u8_3d(mask), float(threshold))


def get_volume_count_gt_threshold_from_z(regression, mask, threshold, z_start):
    return volume_count_gt_threshold_from_z(
        _as_float32_3d(regression), _as_u8_3d(mask), float(threshold), int(z_start)
    )


def get_core_area_count_at_slice(regression, mask, threshold, z):
    return core_area_count_at_slice(_as_float32_3d(regression), _as_u8_3d(mask), float(threshold), int(z))


def get_core_area_profile(regression, mask, threshold, z_start, z_end):
    return core_area_profile(
        _as_float32_3d(regression), _as_u8_3d(mask), float(threshold), int(z_start), int(z_end)
    )


def get_massflux_slice_suffix_arrays(regression, mask, threshold_start, threshold_end):
    return massflux_slice_suffix_arrays(
        _as_float32_3d(regression),
        _as_u8_3d(mask),
        float(threshold_start),
        float(threshold_end),
    )
