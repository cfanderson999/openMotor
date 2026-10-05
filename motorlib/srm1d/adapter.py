import numpy as np

from .burnback import BurnbackGrid


def burnbackFromFmm3dGrain(grain) -> BurnbackGrid:
    """Burnback grid for a set-up Fmm3DGrain. The axial axis is flipped so the nozzle (aft, index 0) ends up last."""
    propellant = np.asarray(grain.coreMap, dtype=bool)
    ends = grain.props['inhibitedEnds'].getValue()
    voidSlice = np.zeros((1,) + propellant.shape[1:], dtype=bool)
    if ends in ('Top', 'Neither'):
        propellant = np.concatenate([voidSlice, propellant], axis=0)
    if ends in ('Bottom', 'Neither'):
        propellant = np.concatenate([propellant, voidSlice], axis=0)

    ny, nx = propellant.shape[1:]
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing='ij')
    valid = xx ** 2 + yy ** 2 <= 1.0
    voxel = grain.props['diameter'].getValue() / grain.mapDim
    return BurnbackGrid.fromMask(propellant[::-1], valid, voxel)
