"""Compare 3D MC surface area with 2D perimeter * length for the finocyl grain."""
import sys; sys.path.insert(0, '.')
import numpy as np
from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor
import mathlib
from mathlib._fmm3d import get_volume_count_gt_threshold

data = loadFile('3dfmm_testing/long_fin_bates.ric', fileTypes.MOTOR)
motor = Motor(data)
grain = motor.grains[0]
grain.mapDim = motor.config.getProperty('3DmapDim')
grain.simulationSetup(motor.config)

regMap = grain.regressionMap
mask = grain.mask
regMapF64 = grain._regressionMapF64
maskU8 = grain._maskU8
diameter = grain.props['diameter'].getValue()
totalLen = grain.totalLength.getValue()
mapDim = grain.mapDim
mapLength = grain.mapLength

print(f"regMap shape: {regMap.shape}")
print(f"mapDim={mapDim}, mapLength={mapLength}")
print(f"diameter={diameter:.6f}, totalLength={totalLen:.6f}")

# Take a 2D slice from the middle of regressionMap (post disk-removal)
mid_z = regMap.shape[0] // 2
slice_2d = regMap[mid_z].astype(np.float64)
mask_2d = mask[mid_z] if mask.ndim == 3 else mask

print(f"\n{'reg(mm)':>8} {'2D_perim*L':>12} {'MC_3D_SA':>12} {'ratio3d/2d':>10} {'dV/dr(raw)':>12} {'SA/dVdr':>8}")

for reg_mm in [0, 1, 2, 5, 10, 15, 20, 25, 28]:
    reg = reg_mm / 1000.0
    mapDist = grain.normalize(reg)
    if mapDist >= grain._maxPolledDist:
        break
    
    # MC 3D SA from the pre-built curve
    mc_sa = float(grain.faceAreaFunc(mapDist))
    
    # 2D perimeter at this regression level using marching squares
    perim_result = mathlib.find_perimeter(slice_2d, mapDist)
    perim_pixels = perim_result[0] if isinstance(perim_result, tuple) else perim_result
    perim_m = grain.mapToLength(perim_pixels)
    
    # 2D perimeter * total length (bore surface contribution only)
    bore_sa = perim_m * totalLen
    
    # dV/dr from raw volume counts
    d = grain.normalize(0.0005)
    rawCount0 = get_volume_count_gt_threshold(regMapF64, maskU8, mapDist)
    rawCount1 = get_volume_count_gt_threshold(regMapF64, maskU8, mapDist + d)
    rawVol0 = grain.mapToVolume(rawCount0)
    rawVol1 = grain.mapToVolume(rawCount1)
    dvdr = -(rawVol1 - rawVol0) / 0.0005

    sa_dvdr = mc_sa / dvdr if dvdr > 0 else float('inf')
    ratio = mc_sa / bore_sa if bore_sa > 0 else 0
    
    print(f"{reg_mm:8.1f} {bore_sa:12.6f} {mc_sa:12.6f} {ratio:10.4f} {dvdr:12.6f} {sa_dvdr:8.4f}")

# Also compute total integral of 2D perimeter * L vs MC SA
print("\n=== Integrated comparison ===")
levels = np.linspace(0, grain.wallWeb, 200)
mc_integ = 0
perim_integ = 0
dr = levels[1] - levels[0]
for reg in levels:
    mapDist = grain.normalize(reg)
    if mapDist >= grain._maxPolledDist:
        break
    mc_integ += float(grain.faceAreaFunc(mapDist)) * dr
    perim_result = mathlib.find_perimeter(slice_2d, mapDist)
    perim = perim_result[0] if isinstance(perim_result, tuple) else perim_result
    perim_integ += grain.mapToLength(perim) * totalLen * dr

density = motor.propellant.getProperty('density')
print(f"Integrated MC SA * dr:      {mc_integ:.6f} m³ = {mc_integ*density:.4f} kg")
print(f"Integrated perim*L * dr:    {perim_integ:.6f} m³ = {perim_integ*density:.4f} kg")
print(f"Actual volume consumed:     {grain.getVolumeAtRegression(0):.6f} m³ = {grain.getVolumeAtRegression(0)*density:.4f} kg")
print(f"MC / perim ratio: {mc_integ/perim_integ:.4f}")
print(f"MC / actual_vol ratio: {mc_integ/grain.getVolumeAtRegression(0):.4f}")
print(f"perim / actual_vol ratio: {perim_integ/grain.getVolumeAtRegression(0):.4f}")
