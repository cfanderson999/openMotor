import numpy as np

from .burnback import BurnbackGrid


def _headToNozzleSlices(grain) -> np.ndarray:
    """Propellant mask of a set-up Fmm3DGrain with open ends padded by a void slice. Axial order runs from the head end
    to the nozzle, the reverse of the grain's own aft-first axis."""
    propellant = np.asarray(grain.coreMap, dtype=bool)
    ends = grain.props['inhibitedEnds'].getValue()
    voidSlice = np.zeros((1,) + propellant.shape[1:], dtype=bool)
    if ends in ('Top', 'Neither'):
        propellant = np.concatenate([voidSlice, propellant], axis=0)
    if ends in ('Bottom', 'Neither'):
        propellant = np.concatenate([propellant, voidSlice], axis=0)
    return propellant[::-1]


def burnbackFromFmm3dGrains(grains, advanceMethod: str = 'narrow-band', wallRing: bool = True) -> BurnbackGrid:
    """Burnback grid for set-up Fmm3DGrains stacked head end first, sharing one casting tube."""
    voxel = grains[0].props['diameter'].getValue() / grains[0].mapDim
    valid = ~np.asarray(grains[0].mask[0], dtype=bool)
    for grain in grains[1:]:
        sameTube = np.array_equal(~np.asarray(grain.mask[0], dtype=bool), valid)
        if not sameTube or not np.isclose(grain.props['diameter'].getValue() / grain.mapDim, voxel):
            raise ValueError("stacked grains must share diameter, map dimension and casting mask")
    stacks = [_headToNozzleSlices(g) for g in grains]
    grid = BurnbackGrid.fromMask(np.concatenate(stacks, axis=0), valid, voxel, advanceMethod, wallRing)
    edges = np.concatenate([[0], np.cumsum([len(stack) for stack in stacks])])
    grid.grainSlices = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    return grid


def burnbackFromFmm3dGrain(grain, advanceMethod: str = 'narrow-band', wallRing: bool = True) -> BurnbackGrid:
    """Burnback grid for a single set-up Fmm3DGrain. The nozzle (aft, index 0) ends up last."""
    return burnbackFromFmm3dGrains([grain], advanceMethod, wallRing)
