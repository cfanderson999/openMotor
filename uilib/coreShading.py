"""Colours for shading the core of a grain in the slice views by a simulated quantity.

`CoreShading` follows one simulation result. It reads the core profiles of recorded steps on demand, keeps them, and fixes
the colour range of every metric from steps spread over the whole burn, so a colour means the same value at every instant.
The functions below turn per-slice colours into the images the slice views paint.
"""

from typing import Dict, List, Optional

import numpy as np
from matplotlib import colormaps

from motorlib.coreprofiles import CoreMetric, CoreProfile, availableMetrics, getCoreProfiles

RANGE_SAMPLES = 24
NO_VALUE = (128, 128, 128, 255)


def buildLut(name: str = "plasma") -> np.ndarray:
    """(256, 4) uint8 RGBA table of a matplotlib colormap."""
    return (colormaps[name](np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)


CORE_LUT = buildLut()


def sliceColors(values: np.ndarray, low: float, high: float, lut: np.ndarray = CORE_LUT) -> np.ndarray:
    """(slices, 4) uint8 colours for per-slice `values` spread over `low` to `high`; NaN is grey."""
    span = high - low if high > low else 1.0
    index = np.nan_to_num((np.asarray(values, dtype=float) - low) / span, nan=0.0)
    colors = lut[np.clip(np.rint(index * (len(lut) - 1)), 0, len(lut) - 1).astype(int)].copy()
    colors[~np.isfinite(values)] = NO_VALUE
    return colors


def axialImage(colors: np.ndarray, z: int, shape) -> np.ndarray:
    """The axial plane at slice `z` (from the aft end) in the colour of that slice, as (rows, cols, 4)."""
    return np.broadcast_to(colors[z], (*shape, 4))


def sideImage(colors: np.ndarray, rows: int) -> np.ndarray:
    """Coronal or sagittal plane, as (rows, slices, 4) with the head end in the leftmost column as the views draw it."""
    return np.broadcast_to(colors[None, ::-1, :], (rows, len(colors), 4))


class CoreShading:
    """Core profiles of one simulation result, with a fixed colour range for each metric."""

    def __init__(self, simResult):
        self.simResult = simResult
        self.steps = len(simResult.channels["time"].getData())
        self.metrics: List[CoreMetric] = availableMetrics(simResult)
        self._profiles: Dict[int, Dict[int, CoreProfile]] = {}
        self._ranges: Optional[Dict[str, tuple]] = None
        self.reported = any(profiles is not None for profiles in simResult.coreProfiles)

    def profiles(self, step: int) -> Dict[int, CoreProfile]:
        if step not in self._profiles:
            self._profiles[step] = getCoreProfiles(self.simResult, step)
        return self._profiles[step]

    def profile(self, step: int, gid: int) -> Optional[CoreProfile]:
        return self.profiles(step).get(gid)

    def depths(self, step: int, gid: int) -> Optional[np.ndarray]:
        """How deep the core of the grain is on each slice at the step, when the model reported it slice by slice."""
        profile = self.profile(step, gid) if self.reported else None
        return profile.regression if profile is not None else None

    def range(self, key: str) -> tuple:
        """Lowest and highest value of the metric over steps spread across the burn, in every grain."""
        if self._ranges is None:
            sampled = np.unique(np.linspace(0, self.steps - 1, min(self.steps, RANGE_SAMPLES)).astype(int))
            self._ranges = {}
            for metric in self.metrics:
                values = [p.metrics[metric.key] for step in sampled for p in self.profiles(int(step)).values()]
                joined = np.concatenate(values) if values else np.array([])
                finite = joined[np.isfinite(joined)]
                self._ranges[metric.key] = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        return self._ranges[key]

    def colors(self, key: str, step: int, gid: int) -> Optional[np.ndarray]:
        """Per-slice colours of the metric for the grain at the step, or None when the grain has no core there."""
        profile = self.profile(step, gid)
        if profile is None:
            return None
        return sliceColors(profile.metrics[key], *self.range(key))


BORE_PLAIN = (96, 112, 176, 255)
CUT_FACE = (214, 214, 208, 255)
PROPELLANT = (168, 168, 164, 255)
END_FACE = (190, 190, 186, 255)
BACKGROUND = (0, 0, 0, 0)
SURFACE_COLORS = [BACKGROUND, BORE_PLAIN, CUT_FACE, PROPELLANT, END_FACE]  # by kind of surface: nothing, bore, cut, wall, end
BORE_KIND = 1


def composeScene(kind: np.ndarray, hitSlice: np.ndarray, shade: np.ndarray, boreColors: Optional[np.ndarray]) -> np.ndarray:
    """(rows, columns, 4) uint8 picture of the surfaces `castCore` found, each lit by its shade. The bore takes the colour of
    the slice it is on from `boreColors` (one per slice), or a plain one when there are none."""
    colors = np.array(SURFACE_COLORS, dtype=np.float32)[kind]
    if boreColors is not None:
        bore = kind == BORE_KIND
        colors[bore] = boreColors[hitSlice[bore]].astype(np.float32)
    colors[..., :3] *= shade[..., None]
    return colors.clip(0, 255).astype(np.uint8)
