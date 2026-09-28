import sys, math, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'test')
from unit.grains.custom3d import _make_bore_cylinder, _MockConfig
from motorlib.grains.custom3d import Custom3DGrain

faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
g = Custom3DGrain()
g.setProperties({'diameter':0.05,'length':0.0,'stlUnit':'m','coreAxis':'-Y','inhibitedEnds':'Both'})
g.props['mesh'].setValue((faces, verts, ''))
g.simulationSetup(_MockConfig(map_dim_3d=48))

rm = g.regressionMap  # post disk-removal
mask = g.mask
valid = ~mask
D = 0.05

print(f"regMap shape: {rm.shape}")
print(f"mask shape: {mask.shape}")
print(f"mapDim: {g.mapDim}")

# For the bore grain with inhibitedEnds='Both':
# No disks added/removed, so regMap has same zdim as coreMap.

# Manual voxel counting at different PHYSICAL regression distances
print(f"\n{'reg_mm':>8} {'mapDist':>8} {'voxels':>8} {'vol_m3':>14} {'expected_m3':>14} {'ratio':>8}")
for reg_mm in np.arange(0, 15.5, 0.5):
    reg = reg_mm / 1000.0
    md = g.normalize(reg)  # = 2*reg/D
    count = int(np.sum(rm[valid] > md))
    vol = g.mapToVolume(count)
    r = 0.01 + reg
    R = 0.025
    if r < R:
        expected = math.pi * (R**2 - r**2) * 0.04
    else:
        expected = 0
    ratio = vol / expected if expected > 0 else float('inf')
    print(f"{reg_mm:8.1f} {md:8.4f} {count:8d} {vol:14.8f} {expected:14.8f} {ratio:8.4f}")

# Now compute dV/dr directly from raw voxel counts
print(f"\n--- dV/dr from raw voxel counts ---")
regs = np.arange(0, 0.015, 0.0001)
counts = []
vols = []
for r in regs:
    md = g.normalize(r)
    c = int(np.sum(rm[valid] > md))
    counts.append(c)
    vols.append(g.mapToVolume(c))

vols = np.array(vols)
dvdr = -np.gradient(vols, regs)
expected_sa = 2 * math.pi * 0.01 * 0.04
print(f"dV/dr at reg=0: {dvdr[0]:.6f}")
print(f"dV/dr at reg=1mm: {dvdr[10]:.6f}")
print(f"Expected SA: {expected_sa:.6f}")
print(f"Ratio at 0: {dvdr[0]/expected_sa:.4f}")

# Cross-check: what does the actual MC give for SA at reg=0?
from motorlib.grain import _marchingAreaPixels
# Need the pre-disk-removal map for MC
# Actually for inhibitedEnds='Both', they should be the same
sa_mc_pixels = _marchingAreaPixels(rm, valid, 0.0)
if sa_mc_pixels is not None:
    sa_mc = g.mapToArea(sa_mc_pixels)
    print(f"\nMC SA at reg=0: {sa_mc:.6f} m²")
    print(f"MC SA / expected: {sa_mc/expected_sa:.4f}")
