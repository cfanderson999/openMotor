"""Check volume curve vs expected for bore grain."""
import sys, math, numpy as np
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'test'))

from unit.grains.custom3d import _make_bore_cylinder, _MockConfig
from motorlib.grains.custom3d import Custom3DGrain

faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
grain = Custom3DGrain()
grain.setProperties({
    'diameter': 0.05, 'length': 0.0, 'stlUnit': 'm',
    'coreAxis': '-Y', 'inhibitedEnds': 'Both',
})
grain.props['mesh'].setValue((faces, verts, ''))
grain.simulationSetup(_MockConfig(map_dim_3d=48))

R = 0.025  # outer radius
r0 = 0.01  # bore radius
L = 0.04   # length

print("  reg(mm)   actual_vol    expected_vol   ratio")
for reg_mm in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]:
    reg = reg_mm / 1000.0
    actual = grain.getVolumeAtRegression(reg)
    # For inhibited ends, only radial burning:
    r = r0 + reg
    if r >= R:
        expected = 0.0
    else:
        expected = math.pi * (R**2 - r**2) * L
    ratio = actual / expected if expected > 0 else float('inf')
    print(f"    {reg_mm:4d}   {actual:.8f}   {expected:.8f}   {ratio:.4f}")

# Check: maybe the regression map's *2 factor means mapDist advances 
# at 2x the rate, so volume drops twice as fast?
print(f"\nnormalize(0.001): {grain.normalize(0.001):.6f}")
print(f"Expected: {0.001 / (0.5 * 0.05):.6f}  = {0.04}")
# The *2 on FMM distances: does it mean that mapDist=0.04 corresponds to 
# physical regression of 1mm or 0.5mm?

# Let's manually count voxels above different thresholds
regMap = grain.regressionMap
mask = grain.mask
valid = ~mask

# For physical regression of 1mm, mapDist = normalize(0.001) = 0.04
md_1mm = grain.normalize(0.001)
count_1mm = np.sum(regMap[valid] > md_1mm)
vol_1mm = grain.mapToVolume(count_1mm)
print(f"\nAt reg=1mm: mapDist={md_1mm:.4f}")
print(f"  Voxels above threshold: {count_1mm}")
print(f"  Volume: {vol_1mm:.8f}")
print(f"  Expected: {math.pi * (R**2 - 0.011**2) * L:.8f}")

# What thresholds in the regression map correspond to the bore surface?
# Min non-zero value in regMap
valid_vals = regMap[valid]
nonzero = valid_vals[valid_vals > 0]
print(f"\nMin non-zero regMap value: {nonzero.min():.6f}")
print(f"Max regMap value: {nonzero.max():.6f}")
print(f"Mean: {nonzero.mean():.6f}")

# The FMM distance was computed with dx=1/mapDim, then *2
# For bore radius 0.01m: pixel radius = 0.01 / (0.05/48) = 9.6 pixels
# FMM distance at surface = 0 (it's the zero-crossing of the binary map)
# Distance 1 voxel into propellant: dx = 1/48 ~ 0.0208 map units
# After *2: 0.0417 map units
# unNormalize(0.0417) = 0.0417/2 * 0.05 = 0.00104 m = 1.04 mm

# So 1 pixel of regression = 1.04 mm physically
# But if we use normalize(1mm) = 2*0.001/0.05 = 0.04
# and the *2 factor in regMap means the values at 1 pixel distance are 0.0417
# These should MATCH because normalize produces the same mapping
print(f"\nnormalize(1mm) = {grain.normalize(0.001):.6f}")
print(f"1 pixel FMM dist after *2 = {2.0 / 48:.6f}")
