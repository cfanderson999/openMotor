"""Quick debug: check SA vs volume derivative for the bore test grain."""
import sys, math, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'test'))

from unit.grains.custom3d import _make_bore_cylinder, _MockConfig
from motorlib.grains.custom3d import Custom3DGrain

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

# Expected SA = 2*pi*r*L
expected_sa = 2 * math.pi * 0.01 * 0.04
actual_sa = grain.getSurfaceAreaAtRegression(0.0)
vol_at_0 = grain.getVolumeAtRegression(0.0)
print(f"Expected SA: {expected_sa:.6f} m²")
print(f"Actual SA:   {actual_sa:.6f} m²")
print(f"Ratio:       {actual_sa/expected_sa:.4f}")
print(f"Volume at 0: {vol_at_0:.8f} m³")

# Check the volume function derivative directly
print(f"\nDiameter: {grain.props['diameter'].getValue()}")
print(f"mapDim:   {grain.mapDim}")
print(f"wallWeb:  {grain.wallWeb:.6f} m")
print(f"maxDist:  {grain._maxPolledDist:.6f}")

# Examine faceArea and volumeFunc data
if hasattr(grain, 'faceArea') and grain.faceArea is not None:
    fa = np.asarray(grain.faceArea)
    print(f"\nfaceArea first 5: {fa[:5]}")
    print(f"faceArea shape: {fa.shape}")

# Manual SA from volume finite difference
regs = np.linspace(0, grain.wallWeb * 0.8, 50)
vols = [grain.getVolumeAtRegression(r) for r in regs]
dr = regs[1] - regs[0]
manual_sa = []
for i in range(len(regs) - 1):
    dv = vols[i] - vols[i+1]
    manual_sa.append(dv / dr)

print(f"\nManual SA from vol at reg=0: {manual_sa[0]:.6f} m²")
print(f"Manual SA from vol at reg=1mm: {manual_sa[1]:.6f} m²")
print(f"Vol[0]: {vols[0]:.8f}, Vol[1]: {vols[1]:.8f}")
print(f"dr: {dr:.6f} m")

# Compare with the MC-based SA that was overridden
# Check the regMap shape
print(f"\nregMap shape: {grain.regressionMap.shape}")
print(f"totalLength:  {grain.totalLength.getValue():.4f}")

# Check whether the *2 factor and normalize are consistent
cell_h = grain.props['diameter'].getValue() / grain.mapDim
print(f"\nVoxel size: {cell_h:.6f} m")
# Sample a regression map value near bore
mid_z = grain.regressionMap.shape[0] // 2
mid_x = grain.regressionMap.shape[2] // 2
mid_y = grain.regressionMap.shape[1] // 2
# Find a propellant voxel near the bore
print(f"regMap[mid_z, mid_y, mid_x]: {grain.regressionMap[mid_z, mid_y, mid_x]:.6f}")
print(f"normalize(wallWeb): {grain.normalize(grain.wallWeb):.6f}")
print(f"maxDist: {grain._maxPolledDist:.6f}")
