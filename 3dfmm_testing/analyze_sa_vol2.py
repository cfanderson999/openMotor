"""Check whether smoothing or the volume/SA mismatch causes the divergence."""
import sys; sys.path.insert(0, '.')
import numpy as np
from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor

data = loadFile('3dfmm_testing/long_fin_bates.ric', fileTypes.MOTOR)
motor = Motor(data)
grain = motor.grains[0]
grain.mapDim = motor.config.getProperty('3DmapDim')
grain.simulationSetup(motor.config)

# Get the regression map and mask
regMap = grain._regressionMapF64
mask = grain.mask
maskU8 = grain._maskU8
valid = np.logical_not(mask)

diameter = grain.props['diameter'].getValue()
mapDim = grain.mapDim

# Compute RAW voxel counts (no smoothing) at fine intervals
print("Comparing RAW volume (no smoothing) vs smoothed volumeFunc vs raw MC vs smoothed SA")
print(f"{'reg(mm)':>8} {'rawVol':>12} {'smthVol':>12} {'rawDvdr':>10} {'smthDvdr':>10} {'smthSA':>10} {'SA/rawDv':>8} {'SA/smDv':>8}")

step_mm = 0.5
n_steps = int(grain.wallWeb * 1000 / step_mm) + 1
prev_raw_vol = None

for i in range(n_steps):
    reg = i * step_mm / 1000.0
    mapDist = grain.normalize(reg)
    
    if mapDist >= grain._maxPolledDist:
        break
    
    # Raw voxel count
    from mathlib._fmm3d import get_volume_count_gt_threshold
    rawCount = get_volume_count_gt_threshold(regMap, maskU8, mapDist)
    rawVol = grain.mapToVolume(rawCount)
    
    # Smoothed volume
    smthVol = float(grain.volumeFunc(mapDist)) if grain.volumeFunc is not None else rawVol
    
    # Smoothed SA
    smthSA = float(grain.faceAreaFunc(mapDist))
    
    # dV/dr from raw
    if prev_raw_vol is not None:
        rawDvdr = -(rawVol - prev_raw_vol) / (step_mm / 1000.0)
        smthDvdr = -(smthVol - prev_smth_vol) / (step_mm / 1000.0)
        if rawDvdr > 0:
            ratio_raw = smthSA / rawDvdr
        else:
            ratio_raw = float('inf')
        if smthDvdr > 0:
            ratio_smth = smthSA / smthDvdr
        else:
            ratio_smth = float('inf')
        if i % 5 == 0 or i > n_steps - 10:
            print(f"{reg*1000:8.2f} {rawVol:12.8f} {smthVol:12.8f} {rawDvdr:10.6f} {smthDvdr:10.6f} {smthSA:10.6f} {ratio_raw:8.4f} {ratio_smth:8.4f}")
    
    prev_raw_vol = rawVol
    prev_smth_vol = smthVol

# Total integrals
regs2 = np.arange(0, grain.wallWeb, 0.0001)
mapDists = [grain.normalize(r) for r in regs2]
rawVols = [grain.mapToVolume(get_volume_count_gt_threshold(regMap, maskU8, md)) for md in mapDists]
smthVols = [float(grain.volumeFunc(md)) for md in mapDists]
smthSAs = [float(grain.faceAreaFunc(md)) if md < grain._maxPolledDist else 0 for md in mapDists]

mass_from_sa = np.trapz(smthSAs, regs2) * motor.propellant.getProperty('density')
mass_from_rawvol = (rawVols[0] - rawVols[-1]) * motor.propellant.getProperty('density')
mass_from_smthvol = (smthVols[0] - smthVols[-1]) * motor.propellant.getProperty('density')

print(f"\n=== Mass budget ===")
print(f"From integrated smooth SA:  {mass_from_sa:.4f} kg")
print(f"From raw volume change:     {mass_from_rawvol:.4f} kg")
print(f"From smooth volume change:  {mass_from_smthvol:.4f} kg")
print(f"SA/rawVol ratio: {mass_from_sa/mass_from_rawvol:.4f}")
print(f"SA/smthVol ratio: {mass_from_sa/mass_from_smthvol:.4f}")
print(f"rawVol/smthVol ratio: {mass_from_rawvol/mass_from_smthvol:.4f}")
