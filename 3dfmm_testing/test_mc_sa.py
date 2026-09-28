"""Validate MC surface area on known geometry, then check finocyl cross-section perimeter."""
import sys; sys.path.insert(0, '.')
import numpy as np
from skimage import measure

# ---- TEST 1: Sphere ----
print("=== TEST 1: Sphere SA via marching cubes ===")
dim = 100
center = dim / 2
radius = 30  # pixels
x, y, z = np.mgrid[:dim, :dim, :dim]
dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2) - radius
# MC at level 0 -> sphere surface
verts, faces, _, _ = measure.marching_cubes(dist, level=0)
mc_area = measure.mesh_surface_area(verts, faces)
analytical_area = 4 * np.pi * radius**2
print(f"Analytical area: {analytical_area:.2f} pixel²")
print(f"MC area:         {mc_area:.2f} pixel²")
print(f"MC / analytical: {mc_area/analytical_area:.6f}")
print()

# ---- TEST 2: Cylinder bore along Z ----
print("=== TEST 2: Cylinder bore SA via marching cubes ===")
dimxy = 100
dimz = 500
center_xy = dimxy / 2
r_bore = 20  # pixels
# Distance field: positive outside bore, negative inside
y2d, x2d = np.mgrid[:dimxy, :dimxy]
dist2d = np.sqrt((x2d - center_xy)**2 + (y2d - center_xy)**2) - r_bore
# Stack along z
dist3d = np.broadcast_to(dist2d[np.newaxis, :, :], (dimz, dimxy, dimxy)).copy()

verts, faces, _, _ = measure.marching_cubes(dist3d, level=0)
mc_area = measure.mesh_surface_area(verts, faces)
analytical_area = 2 * np.pi * r_bore * dimz  # cylinder lateral surface
print(f"Analytical cylinder area: {analytical_area:.2f} pixel²")
print(f"MC cylinder area:         {mc_area:.2f} pixel²")
print(f"MC / analytical: {mc_area/analytical_area:.6f}")
print()

# ---- TEST 3: Check multiple iso-levels of the cylinder ----
print("=== TEST 3: Multiple iso-levels of cylinder bore ===")
print(f"{'level':>6} {'MC_area':>12} {'analytical':>12} {'ratio':>8}")
for level in [0, 2, 5, 10, 15, 20]:
    if r_bore + level >= center_xy:
        break
    verts, faces, _, _ = measure.marching_cubes(dist3d, level=level)
    mc_a = measure.mesh_surface_area(verts, faces)
    analytical_a = 2 * np.pi * (r_bore + level) * dimz
    print(f"{level:6d} {mc_a:12.2f} {analytical_a:12.2f} {mc_a/analytical_a:8.6f}")
print()

# ---- TEST 4: Check the actual finocyl grain SA vs 2D perimeter ----
print("=== TEST 4: 3D Finocyl grain - MC SA vs 2D perimeter * length ===")
from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import mathlib

data = loadFile('3dfmm_testing/long_fin_bates.ric', fileTypes.MOTOR)
motor = Motor(data)
grain = motor.grains[0]
grain.mapDim = motor.config.getProperty('3DmapDim')
grain.simulationSetup(motor.config)

# Get the stored regression map (post disk-removal)
regMap = grain.regressionMap
mask = grain.mask
print(f"regMap shape: {regMap.shape}")
print(f"mapDim={grain.mapDim}, mapLength={grain.mapLength}")
print(f"diameter={grain.props['diameter'].getValue():.6f} m")
diameter = grain.props['diameter'].getValue()
hxy = diameter / grain.mapDim

# Take a 2D slice from the middle of the grain
mid_z = regMap.shape[0] // 2
slice_2d = regMap[mid_z]
mask_2d = mask[mid_z] if mask.ndim == 3 else mask

# Compute 2D perimeter at various regression levels
print(f"\n{'reg(mm)':>8} {'2D_perim*L':>12} {'MC_3D_SA':>12} {'ratio':>8} {'dV/dr(raw)':>12}")
from mathlib._fmm3d import get_volume_count_gt_threshold
regMapF64 = grain._regressionMapF64
maskU8 = grain._maskU8

for reg_mm in [0, 2, 5, 10, 15, 20, 25]:
    reg = reg_mm / 1000.0
    mapDist = grain.normalize(reg)
    if mapDist >= grain._maxPolledDist:
        break
    
    # MC 3D SA (from the smoothed curve)
    mc_sa = grain.faceAreaFunc(mapDist)
    
    # 2D perimeter at this regression level
    # Create a boolean map: True where coremap threshold is exceeded
    core_at_reg = slice_2d < mapDist  # pixels that have been consumed
    core_at_reg = core_at_reg | mask_2d  # add mask
    # The perimeter of this 2D shape
    perim_pixels = mathlib.find_perimeter(core_at_reg.astype(np.uint8))
    perim_m = grain.mapToLength(perim_pixels)
    
    # SA from 2D perimeter * grain length (bore surface only, no end face)
    bore_sa = perim_m * grain.totalLength.getValue()
    
    # dV/dr from raw volume
    rawCount = get_volume_count_gt_threshold(regMapF64, maskU8, mapDist)
    rawVol = grain.mapToVolume(rawCount)
    rawCount2 = get_volume_count_gt_threshold(regMapF64, maskU8, mapDist + grain.normalize(0.0005))
    rawVol2 = grain.mapToVolume(rawCount2)
    dvdr = -(rawVol2 - rawVol) / 0.0005
    
    print(f"{reg_mm:8.1f} {bore_sa:12.6f} {mc_sa:12.6f} {mc_sa/bore_sa if bore_sa > 0 else 0:8.4f} {dvdr:12.6f}")
