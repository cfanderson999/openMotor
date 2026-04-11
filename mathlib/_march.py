"""Python wrapper for _march_cy – parallel MC area computation."""

import numpy as np
import os

try:
    from ._march_cy import mc_surface_area as _mc_surface_area_cy  # type: ignore[import-not-found]
    _CY_AVAILABLE = True
except ImportError:
    _CY_AVAILABLE = False


def _env_enabled_default_true(name):
    """Return True unless env var explicitly disables the feature."""
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _validate_inputs(regression_map, valid_mask, level):
    grid = np.asarray(regression_map, dtype=np.float64)
    mask = np.asarray(valid_mask)

    if grid.ndim != 3:
        raise ValueError(f"regression_map must be 3-D, got shape {grid.shape!r}")
    if mask.ndim != 3:
        raise ValueError(f"valid_mask must be 3-D, got shape {mask.shape!r}")
    if grid.shape != mask.shape:
        raise ValueError(
            f"regression_map and valid_mask shape mismatch: {grid.shape!r} vs {mask.shape!r}"
        )

    lvl = float(level)
    if not np.isfinite(lvl):
        raise ValueError(f"level must be finite, got {level!r}")

    return np.ascontiguousarray(grid), np.ascontiguousarray(mask, dtype=bool), lvl


def marching_area_pixels(regression_map, valid_mask, level, step_size=1):
    """
    Return iso-surface area in pixel units for *level* on *regression_map*.

    Uses the native Cython + OpenMP implementation by default when available.
    Falls back to skimage.measure.marching_cubes if native is unavailable,
    disabled, or raises.

    Parameters
    ----------
    regression_map : 3-D float64 array  (nz, ny, nx)
    valid_mask     : 3-D bool array     (nz, ny, nx)  True = include voxel
    level          : float
    step_size      : int, optional (default 1).  Passed to the MC kernel;
                     higher values trade accuracy for speed, identical to
                     skimage.measure.marching_cubes(step_size=...).

    Returns
    -------
    float or None (None when the iso-level produces no surface)
    """
    grid, mask, lvl = _validate_inputs(regression_map, valid_mask, level)
    step = max(1, int(step_size))

    # Default-on for 3DFMM; set OPENMOTOR_EXPERIMENTAL_CY_MARCH=0/false/no to opt out.
    use_cy = _env_enabled_default_true("OPENMOTOR_EXPERIMENTAL_CY_MARCH")
    if _CY_AVAILABLE and use_cy:
        vmask = np.ascontiguousarray(mask, dtype=np.uint8)
        try:
            area = _mc_surface_area_cy(grid, vmask, lvl, step)
            return area if area > 0.0 else None
        except Exception:
            pass  # fall through to skimage fallback

    # --- skimage fallback ---
    from skimage import measure
    try:
        verts, faces, _, _ = measure.marching_cubes(grid, level=lvl, mask=mask, step_size=step)
        return measure.mesh_surface_area(verts, faces)
    except (RuntimeError, ValueError):
        return None
