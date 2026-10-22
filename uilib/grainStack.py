"""Grains laid end to end as one stack of slices, which is what the 3-D viewer draws.

The slices run from the aft end to the head end, as the maps of a grain do. A motor lists its grains from the head end, so
the stack of the whole motor is the grains in the reverse of that order.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np


class GrainStack:
    """The regression maps and masks of `members`, a list of (grain index, grain) from the aft end, joined into one."""

    def __init__(self, members: List[Tuple[int, object]]):
        self.members = members
        self.counts = [grain.regressionMap.shape[0] for _, grain in members]
        self.regressionMap = np.ascontiguousarray(np.concatenate([grain.regressionMap for _, grain in members]),
                                                  dtype=np.float32)
        self.mask = np.concatenate([np.asarray(grain.mask, dtype=bool) for _, grain in members])

    @classmethod
    def single(cls, gid: int, grain) -> "GrainStack":
        return cls([(gid, grain)])

    @classmethod
    def fullMotor(cls, grains: List[Tuple[int, object]]) -> "GrainStack":
        """Every grain of `grains` (head end first) that has the cross section of the first, as one stack."""
        shape = grains[0][1].regressionMap.shape[1:]
        return cls([(gid, grain) for gid, grain in reversed(grains) if grain.regressionMap.shape[1:] == shape])

    @property
    def gids(self) -> List[int]:
        return [gid for gid, _ in self.members]

    def join(self, perGrain: Callable[[int, object], Optional[np.ndarray]], fill) -> np.ndarray:
        """One value per slice of the stack from `perGrain(gid, grain)`, which gives an array with a value (or a row of
        values) for each slice of that grain, or None for `fill` on all of them."""
        parts = []
        for (gid, grain), count in zip(self.members, self.counts):
            part = perGrain(gid, grain)
            parts.append(np.broadcast_to(np.asarray(fill, dtype=float), (count,) + np.shape(fill)) if part is None else part)
        return np.concatenate(parts)
