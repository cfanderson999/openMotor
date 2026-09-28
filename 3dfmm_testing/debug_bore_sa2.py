"""Debug: dump the raw polled/volume data for the bore grain."""
import sys, math, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'test'))

from unit.grains.custom3d import _make_bore_cylinder, _MockConfig
from motorlib.grains.custom3d import Custom3DGrain
from motorlib.grain import _smoothSeries

faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
grain = Custom3DGrain()
grain.setProperties({
    'diameter':      0.05,
    'length':        0.0,
    'stlUnit':       'm',
    'coreAxis':      '-Y',
    'inhibitedEnds': 'Both',
})
grain.props['mesh'].setValue((faces, verts, ''))
grain.simulationSetup(_MockConfig(map_dim_3d=48))

# Access the internal data
# The volumeFunc and faceAreaFunc are built during generateRegressionMap
# Let me re-examine the volume derivative computation

# Manually replicate the volume derivative calculation
diameter = grain.props['diameter'].getValue()
print(f"diameter: {diameter}")
print(f"mapDim: {grain.mapDim}")
print(f"wallWeb: {grain.wallWeb:.6f} m")
print(f"maxDist: {grain._maxPolledDist:.6f}")

# Get volume at many small steps
regs = np.linspace(0, grain.wallWeb*0.5, 200)
vols = np.array([grain.getVolumeAtRegression(r) for r in regs])
dr = regs[1] - regs[0]

# SA from finite diff of volume(regDist)
dvdr = -np.gradient(vols, regs)
print(f"\nSA from finite diff of volume(regDist) at reg=0: {dvdr[0]:.6f}")
print(f"SA from finite diff at reg=1mm: {dvdr[5]:.6f}")
print(f"Expected SA at reg=0: {2*math.pi*0.01*0.04:.6f}")
print(f"Ratio: {dvdr[0]/(2*math.pi*0.01*0.04):.4f}")

# Now check: is the volume function itself correct?
# Expected V(0) = pi * (R^2 - r^2) * L = pi * (0.025^2 - 0.01^2) * 0.04
expected_v0 = math.pi * (0.025**2 - 0.01**2) * 0.04
print(f"\nExpected V(0): {expected_v0:.8f}")
print(f"Actual V(0):   {vols[0]:.8f}")

# Expected V(0.002) = pi * (R^2 - (r+0.002)^2) * L
expected_v2 = math.pi * (0.025**2 - 0.012**2) * 0.04
print(f"\nExpected V(0.002): {expected_v2:.8f}")
print(f"Actual V(0.002):   {grain.getVolumeAtRegression(0.002):.8f}")

# Check faceAreaFunc directly with getSurfaceAreaAtRegression
sa0 = grain.getSurfaceAreaAtRegression(0.0)
sa1 = grain.getSurfaceAreaAtRegression(0.001)
print(f"\ngetSurfaceAreaAtRegression(0): {sa0:.6f}")
print(f"getSurfaceAreaAtRegression(0.001): {sa1:.6f}")
print(f"Expected: {2*math.pi*0.01*0.04:.6f}")
print(f"Ratio at 0: {sa0/(2*math.pi*0.01*0.04):.4f}")

# Check if mapToVolume gives correct volume per voxel
voxel_vol = diameter**3 / grain.mapDim**3
print(f"\nvoxel_vol: {voxel_vol:.10f}")
n_voxels = grain.regressionMap.shape[0] * grain.regressionMap.shape[1] * grain.regressionMap.shape[2]
print(f"Total voxels: {n_voxels}")
print(f"regMap shape: {grain.regressionMap.shape}")
print(f"zdim: {grain.regressionMap.shape[0]}")

# The key check: is the volume too large because the z voxels are counted 
# but the physical z dimension doesn't match?
zdim = grain.regressionMap.shape[0]
phys_z_from_voxels = zdim * (diameter / grain.mapDim)
print(f"\nPhysical z from voxels: {phys_z_from_voxels:.6f} m")
print(f"Actual total length: {grain.totalLength.getValue():.6f} m") 
print(f"Ratio: {phys_z_from_voxels / grain.totalLength.getValue():.4f}")

# Count propellant voxels
prop_voxels = np.sum(grain.regressionMap[~grain.mask] > 0)
print(f"\nPropellant voxels (regMap > 0, not masked): {prop_voxels}")
vol_from_voxels = grain.mapToVolume(prop_voxels)
print(f"Volume from voxels: {vol_from_voxels:.8f}")
print(f"Expected: {expected_v0:.8f}")
