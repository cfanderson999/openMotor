"""Orientation of the camera of the 3-D view, as three unit vectors in (slice, y, x) order that point right and up on the
screen and at the camera."""

import numpy as np

SCREEN_RIGHT, SCREEN_UP, TOWARD_CAMERA = 0, 1, 2


def defaultView() -> np.ndarray:
    """Looking down the +y axis with the long axis across the screen, the head end on the left and the nozzle on the right."""
    return np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def startingView() -> np.ndarray:
    """The default view turned a little, so that a bore cut open along the axis is seen at an angle."""
    return turn(defaultView(), 0.35, 0.3)


def _rotate(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """`vectors` (rows) turned by `angle` about the unit `axis` (Rodrigues' formula)."""
    cos, sin = np.cos(angle), np.sin(angle)
    return vectors * cos + np.cross(axis, vectors) * sin + np.outer(vectors @ axis, axis) * (1.0 - cos)


def turn(view: np.ndarray, across: float, down: float) -> np.ndarray:
    """The view after dragging the scene by `across` radians toward the right and `down` radians toward the bottom."""
    right, up = view[SCREEN_RIGHT], view[SCREEN_UP]
    turned = _rotate(view, up, across)
    turned = _rotate(turned, right, down)
    return orthonormal(turned)


def orthonormal(view: np.ndarray) -> np.ndarray:
    """The view with its rows brought back to unit length and at right angles, so drift does not build up."""
    right = view[SCREEN_RIGHT] / np.linalg.norm(view[SCREEN_RIGHT])
    up = view[SCREEN_UP] - right * (view[SCREEN_UP] @ right)
    up /= np.linalg.norm(up)
    return np.array([right, up, np.cross(up, right)])
