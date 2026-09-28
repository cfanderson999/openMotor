import sys, math, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'test')
from unit.grains.custom3d import _make_bore_cylinder, _MockConfig
from motorlib.grains.custom3d import Custom3DGrain
from motorlib.grain import _smoothSeries
from scipy import interpolate

faces, verts = _make_bore_cylinder(radius=0.01, length=0.04, segments=16)
g = Custom3DGrain()
g.setProperties({'diameter':0.05,'length':0.0,'stlUnit':'m','coreAxis':'-Y','inhibitedEnds':'Both'})
g.props['mesh'].setValue((faces, verts, ''))
g.simulationSetup(_MockConfig(map_dim_3d=48))

rm = g._regressionMapF64
mask = g.mask
valid = ~mask
D = 0.05

# Replicate the volumeFunc construction from generateRegressionMap
polled = g.volumeFunc.x.tolist()
print(f"polled ({len(polled)} points):")
for p in polled:
    print(f"  {p:.6f}")

# Reproduce _volCounts
_mask2d = mask[0] if mask.ndim == 3 else mask
_validFlat = np.logical_not(_mask2d).ravel()
_zdimRm = rm.shape[0]
_rmFlat = rm.reshape(_zdimRm, -1)
_rmSorted = np.sort(_rmFlat[:, _validFlat].ravel())
_nVox = len(_rmSorted)
_polledArr = np.asarray(polled)
_idxs = np.searchsorted(_rmSorted, _polledArr, side='right')
_volCounts = g.mapToVolume(_nVox - _idxs.astype(float))
_smoothVol = _smoothSeries(_volCounts)

print(f"\n{'polled':>10} {'rawVol':>14} {'smoothVol':>14}")
for i in range(len(polled)):
    print(f"{polled[i]:10.6f} {_volCounts[i]:14.10f} {_smoothVol[i]:14.10f}")

# Now compute derivative
_dVdPoll = np.gradient(_smoothVol, _polledArr)
_saFromVol = np.maximum(0.0, -_dVdPoll) * (2.0 / D)
_saSmooth = _smoothSeries(_saFromVol)
expected_sa = 2 * math.pi * 0.01 * 0.04

print(f"\n{'polled':>10} {'dVdPoll':>14} {'SA_raw':>14} {'SA_smooth':>14}")
for i in range(len(polled)):
    print(f"{polled[i]:10.6f} {_dVdPoll[i]:14.10f} {_saFromVol[i]:14.10f} {_saSmooth[i]:14.10f}")

print(f"\nExpected SA: {expected_sa:.6f}")
print(f"SA_raw[0]: {_saFromVol[0]:.6f}")
print(f"SA_smooth[0]: {_saSmooth[0]:.6f}")
print(f"Ratio raw: {_saFromVol[0]/expected_sa:.4f}")
print(f"Ratio smooth: {_saSmooth[0]/expected_sa:.4f}")

# What about using raw (unsmoothed) volume for derivative?
_dVdPoll_raw = np.gradient(_volCounts, _polledArr)
_saFromRawVol = np.maximum(0.0, -_dVdPoll_raw) * (2.0 / D)
print(f"\nSA from raw vol derivative[0]: {_saFromRawVol[0]:.6f}")
print(f"Ratio: {_saFromRawVol[0]/expected_sa:.4f}")
