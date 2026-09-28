"""Directly compare MC SA on pre-removal vs post-removal regression map."""
import sys; sys.path.insert(0, '.')
import numpy as np
from skimage import measure
from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor

data = loadFile('3dfmm_testing/long_fin_bates.ric', fileTypes.MOTOR)
motor = Motor(data)
grain = motor.grains[0]
grain.mapDim = motor.config.getProperty('3DmapDim')

# Run generateCoreMap manually
grain.simulationSetup(motor.config)

# Access the stored (post-removal) regression map
postMap = grain.regressionMap
postValid = np.logical_not(grain.mask)
diameter = grain.props['diameter'].getValue()
mapDim = grain.mapDim

# Rebuild the pre-removal map: re-insert uninhibited disk + re-run FMM (or just match)
# Actually, let's compute MC SA on BOTH maps at the same levels
print("=== Direct MC SA comparison: post-removal vs faceAreaFunc ===")
print(f"postMap shape: {postMap.shape}")

validPost = np.logical_not(grain.mask)
maxDistPost = float(np.max(postMap[validPost]))
print(f"maxDist in post-removal map: {maxDistPost:.6f}")
print(f"maxPolledDist from setup:    {grain._maxPolledDist:.6f}")

print(f"\n{'reg(mm)':>8} {'rawMC_post':>12} {'smoothSA':>12} {'ratio':>8}")
for reg_mm in [0, 1, 2, 5, 10, 15, 20, 25, 28, 30, 32, 34]:
    reg = reg_mm / 1000.0
    mapDist = grain.normalize(reg)
    if mapDist > maxDistPost:
        break
    
    # Raw MC on post-removal map
    try:
        verts, faces, _, _ = measure.marching_cubes(postMap, level=mapDist, mask=validPost)
        rawMC = grain.mapToArea(measure.mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        rawMC = 0
    
    # Smoothed SA from faceAreaFunc
    if mapDist < grain._maxPolledDist:
        smthSA = float(grain.faceAreaFunc(mapDist))
    else:
        smthSA = 0
    
    ratio = smthSA / rawMC if rawMC > 0 else float('inf')
    print(f"{reg_mm:8.1f} {rawMC:12.6f} {smthSA:12.6f} {ratio:8.4f}")

# Now compute integrated SA from raw MC on post-removal map
print("\n=== Integrated raw MC SA on post-removal map ===")
levels = np.linspace(0, grain.wallWeb * 0.95, 100)
raw_integ = 0
smooth_integ = 0
dr = levels[1] - levels[0]

from mathlib._fmm3d import get_volume_count_gt_threshold
for reg in levels:
    mapDist = grain.normalize(reg)
    if mapDist > maxDistPost:
        break
    try:
        verts, faces, _, _ = measure.marching_cubes(postMap, level=mapDist, mask=validPost)
        rawMC = grain.mapToArea(measure.mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        rawMC = 0
    raw_integ += rawMC * dr
    if mapDist < grain._maxPolledDist:
        smooth_integ += float(grain.faceAreaFunc(mapDist)) * dr

density = motor.propellant.getProperty('density')
actVol = grain.getVolumeAtRegression(0)
print(f"Integrated raw MC:    {raw_integ:.6f} m³ = {raw_integ*density:.4f} kg")
print(f"Integrated smooth SA: {smooth_integ:.6f} m³ = {smooth_integ*density:.4f} kg")
print(f"Actual volume:        {actVol:.6f} m³ = {actVol*density:.4f} kg")
print(f"raw MC / actual vol:    {raw_integ/actVol:.4f}")
print(f"smooth SA / actual vol: {smooth_integ/actVol:.4f}")
