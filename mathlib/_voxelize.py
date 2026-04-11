"""Python wrapper for _voxelize_cy – parallel mesh voxelizer."""

import numpy as np
import os

try:
    from ._voxelize_cy import voxelize_mesh as _voxelize_cy  # type: ignore[import-not-found]
    _CY_AVAILABLE = True
except ImportError:
    _CY_AVAILABLE = False


_MAX_AXIS_POINTS = 4096
_MAX_TOTAL_POINTS = 512_000_000


def _env_enabled_default_true(name):
    """Return True unless env var explicitly disables the feature."""
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _validate_and_prepare_mesh(vertices, faces, voxel_density):
    """Validate mesh inputs and return sanitized contiguous arrays.

    This keeps the native kernel robust against malformed/partial mesh payloads
    coming from imported geometry.
    """
    try:
        density = float(voxel_density)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"voxel_density must be numeric, got {voxel_density!r}") from exc

    if not np.isfinite(density) or density <= 0:
        raise ValueError(f"voxel_density must be finite and > 0, got {density!r}")
    if density < 1e-9:
        raise ValueError(f"voxel_density is unreasonably small: {density!r}")

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape!r}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {tris.shape!r}")
    if verts.shape[0] == 0 or tris.shape[0] == 0:
        raise ValueError("mesh has no vertices/faces")
    if not np.isfinite(verts).all():
        raise ValueError("vertices contain NaN/Inf")

    vcount = int(verts.shape[0])
    if int(np.min(tris)) < 0 or int(np.max(tris)) >= vcount:
        raise ValueError("face indices are out of bounds")

    # Drop degenerate triangles to avoid unstable ray intersections.
    tri_verts = verts[tris]
    area2 = np.linalg.norm(
        np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0]),
        axis=1,
    )
    keep = np.isfinite(area2) & (area2 > 1e-18)
    if not np.any(keep):
        raise ValueError("all faces are degenerate")
    if np.count_nonzero(~keep) > 0:
        tris = tris[keep]

    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int32)
    return verts, tris, density


def voxelize_mesh(vertices, faces, voxel_density):
    """
    Voxelize a closed triangle mesh at *voxel_density* spacing.

    Replicates PyVista's ``pv.voxelize_volume(mesh, density=voxel_density)``
    + ``cell_data_to_point_data()`` pipeline but in pure C / OpenMP.

    Grid construction matches PyVista exactly:
        coords = np.arange(bounds_min, bounds_max, voxel_density)

    Parameters
    ----------
    vertices      : (N, 3) float64 – vertex positions in metres.
    faces         : (M, 3) int32   – triangle vertex indices (0-based).
    voxel_density : float           – grid spacing in metres.

    Returns
    -------
    coreArray : (nx, ny, nz) bool ndarray
        True where the grid point is **outside** (propellant), False where
        inside the mesh bore.  Axes correspond to mesh X, Y, Z after the
        same ``rot90(axes=(1,0))`` + ``logical_not`` that the original
        PyVista path applied.
    bounds : (3,) float64
        Mesh extent [dx, dy, dz] in metres (same as ``mesh.bounds`` extents).

    Raises
    ------
    ImportError
        If the Cython extension has not been built.  Build with:
        ``python setup.py build_ext --inplace``
    """
    if not _CY_AVAILABLE:
        raise ImportError(
            "mathlib._voxelize_cy is unavailable. "
            "Build extensions with: python setup.py build_ext --inplace"
        )

    if not _env_enabled_default_true("OPENMOTOR_EXPERIMENTAL_CY_VOXEL"):
        raise RuntimeError("native voxelizer disabled via OPENMOTOR_EXPERIMENTAL_CY_VOXEL")

    verts, tris, density = _validate_and_prepare_mesh(vertices, faces, voxel_density)

    # Mesh bounds.
    xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()

    if not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax, zmin, zmax)):
        raise ValueError("mesh bounds contain NaN/Inf")

    bounds = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    # Build grid coordinates matching PyVista: arange(min, max, density).
    x_coords = np.arange(xmin, xmax, density)
    y_coords = np.arange(ymin, ymax, density)
    z_coords = np.arange(zmin, zmax, density)

    # Ensure at least one point per axis.
    if len(x_coords) == 0:
        x_coords = np.array([xmin])
    if len(y_coords) == 0:
        y_coords = np.array([ymin])
    if len(z_coords) == 0:
        z_coords = np.array([zmin])

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    if nx > _MAX_AXIS_POINTS or ny > _MAX_AXIS_POINTS or nz > _MAX_AXIS_POINTS:
        raise ValueError(
            f"voxel grid axis too large ({nx}, {ny}, {nz}); "
            f"check mesh units or map settings"
        )
    total_points = nx * ny * nz
    if total_points > _MAX_TOTAL_POINTS:
        raise ValueError(
            f"voxel grid too large ({total_points} points); "
            f"check mesh units or map settings"
        )

    x_coords = np.ascontiguousarray(x_coords, dtype=np.float64)
    y_coords = np.ascontiguousarray(y_coords, dtype=np.float64)
    z_coords = np.ascontiguousarray(z_coords, dtype=np.float64)

    # Result: (nx, ny, nz) uint8 – 1 = inside mesh bore.
    inside = _voxelize_cy(verts, tris, x_coords, y_coords, z_coords)

    # Post-processing to match the PyVista path (rot90 + logical_not), keeping
    # peak allocation at 2× grid size instead of 3×:
    #  1. Flip 0↔1 in-place  (bore=1 → propellant=True, propellant=0 → bore=False)
    #     — avoids a separate logical_not copy.
    #  2. View as bool (same bytes, no copy).
    #  3. rot90 returns a non-contiguous view; ascontiguousarray makes the final copy.
    # Total peak = inside(uint8) + result(bool) = 2 × grid bytes.
    np.bitwise_xor(inside, np.uint8(1), out=inside)
    coreArray = np.ascontiguousarray(np.rot90(inside.view(np.bool_), axes=(1, 0)))

    return coreArray, bounds
