"""Compare marching-cubes SA with dV/dr to find where they diverge."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor

data = loadFile('3dfmm_testing/long_fin_bates.ric', fileTypes.MOTOR)
motor = Motor(data)
grain = motor.grains[0]
grain.mapDim = motor.config.getProperty('3DmapDim')
grain.simulationSetup(motor.config)

diameter = grain.props['diameter'].getValue()

# Fine sweep: compute SA and volume at 0.1mm increments
step = 0.0001  # 0.1mm
n_steps = int(grain.wallWeb / step) + 1
regs = [i * step for i in range(n_steps)]

sas = [grain.getSurfaceAreaAtRegression(r) for r in regs]
vols = [grain.getVolumeAtRegression(r) for r in regs]

# dV/dr via finite differences
dvdr = [-(vols[i+1] - vols[i]) / step for i in range(len(regs)-1)]

print(f"wallWeb = {grain.wallWeb*1000:.2f} mm")
print(f"{'reg(mm)':>8} {'SA(m²)':>10} {'dV/dr':>10} {'SA/dVdr':>8} {'Vol(m³)':>12}")
for i in range(0, len(regs)-1, max(1, len(regs)//30)):
    if sas[i] > 0 and dvdr[i] > 0:
        ratio = sas[i] / dvdr[i]
    else:
        ratio = float('inf')
    print(f"{regs[i]*1000:8.2f} {sas[i]:10.6f} {dvdr[i]:10.6f} {ratio:8.4f} {vols[i]:12.8f}")

# Integrated check: use the SA to predict mass consumed, compare with actual
# mass_consumed = integral(SA * dr) * density
sa_arr = np.array(sas[:-1])
r_arr = np.array(regs[:-1])
predicted_mass = np.trapz(sa_arr, r_arr) * motor.propellant.getProperty('density')
actual_mass = (vols[0] - vols[-1]) * motor.propellant.getProperty('density')
print(f"\nIntegrated SA*dr (predicted mass consumed): {predicted_mass:.4f} kg = {predicted_mass*2.20462:.4f} lb")
print(f"Actual mass consumed (volume):              {actual_mass:.4f} kg = {actual_mass*2.20462:.4f} lb")
print(f"Ratio SA-predicted/actual: {predicted_mass/actual_mass:.4f}")
print(f"This means SA overestimates burning by {(predicted_mass/actual_mass - 1)*100:.1f}%")
