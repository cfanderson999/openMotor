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

rm = g.regressionMap
mask = g.mask
valid = ~mask

# Check the volumeFunc interpolation points
xdata = g.volumeFunc.x
ydata = g.volumeFunc.y
print("volumeFunc interpolation points:")
for i in range(min(10, len(xdata))):
    print(f"  x={xdata[i]:.6f}  y={ydata[i]:.10f}")
print(f"  ... ({len(xdata)} points total)")

# The key: what's the ratio volume to raw voxel count?
count0 = int(np.sum(rm[valid] > 0))
vol0_map = g.mapToVolume(count0)
print(f"\nVoxels > 0: {count0}")
print(f"mapToVolume(count0): {vol0_map:.10f}")
print(f"volumeFunc(0): {float(g.volumeFunc(0)):.10f}")

# Check faceArea (SA from volume derivative)
fa = g.faceArea
print(f"\nfaceArea (SA from vol-deriv): {fa[:5]}")
expected_sa = 2*math.pi*0.01*0.04
print(f"Expected SA: {expected_sa:.6f}")
print(f"Ratio: {fa[0]/expected_sa:.4f}")

# Check if the issue is the *2 in the regression map
# The FMM gives distances in map-coordinate units (dx=1/mapDim).
# After *2, values are in "normalized" units matching normalize().
# Volume counting: sum(regMap > threshold). 
# Since regMap is *2, threshold must be in the *2 units too.
# normalize(r) = 2*r/diameter gives the right threshold.
# So volume counting is correct.

# The derivative: _smoothVol is volume vs polled (which is in *2 units).
# np.gradient(_smoothVol, polled) gives dV/d(polled).
# To get dV/dr, multiply by d(polled)/dr = 2/diameter.
# But wait — is this double-counting the *2?

# The *2 factor means: regMap = 2 * fmm_dist
# normalize(r) = 2*r/diameter
# The FMM distance for physical distance r is: fmm_dist = r/diameter * mapDim * dx = r/diameter
# So regMap = 2 * r/diameter = normalize(r). Correct.
# polled = mapDist = normalize(r) = 2*r/diameter
# dr/d(polled) = diameter/2
# dV/dr = dV/d(polled) * d(polled)/dr = dV/d(polled) * 2/diameter

# But I see 2x the expected. Let me check if getVolumeAtRegression itself is right:
for r_mm in [0, 0.5, 1.0]:
    r = r_mm / 1000
    md = g.normalize(r)
    vf = float(g.volumeFunc(md))
    # Also raw count
    count = int(np.sum(rm[valid] > md))
    vraw = g.mapToVolume(count)
    R = 0.025
    expected = math.pi*(R**2 - (0.01+r)**2)*0.04
    print(f"reg={r_mm}mm  mapDist={md:.4f}  volFunc={vf:.8f}  rawCount={vraw:.8f}  expected={expected:.8f}")
