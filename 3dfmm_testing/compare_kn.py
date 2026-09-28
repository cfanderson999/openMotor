"""Compare KN and SA curves between the analytical and 3DFMM long_fin_bates motors."""
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor


def load_and_run(path):
    data = loadFile(str(path), fileTypes.MOTOR)
    motor = Motor(data)
    res = motor.runSimulation()
    return motor, res


def summary(label, motor, res):
    t = np.array(res.channels['time'].getData())
    kn = np.array(res.channels['kn'].getData())
    p = np.array(res.channels['pressure'].getData())
    f = np.array(res.channels['force'].getData())
    masses = np.array(res.channels['mass'].getData())
    if masses.ndim > 1:
        masses = masses.sum(axis=1)
    prop = masses[0] - masses[-1]
    impulse = float(np.trapz(f, t))
    isp = impulse / (prop * 9.80665)

    n = len(t)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Grains: {len(motor.grains)}  ({', '.join(type(g).__name__ for g in motor.grains)})")
    for i, g in enumerate(motor.grains):
        d = g.props['diameter'].getValue() * 1000
        inh = g.props.get('inhibitedEnds')
        inh_val = inh.getValue() if inh else 'N/A'
        print(f"    [{i}] {type(g).__name__} D={d:.1f}mm  inhibited={inh_val}")
    print(f"  Burn time:     {t[-1]:.2f} s")
    print(f"  Prop mass:     {prop:.4f} kg")
    print(f"  Total impulse: {impulse:.0f} Ns")
    print(f"  Delivered Isp: {isp:.1f} s")
    print(f"  Peak pressure: {p.max()/1e6:.2f} MPa")
    print(f"  Peak KN:       {kn.max():.1f} at t={t[np.argmax(kn)]:.3f}s")

    # KN at quartiles
    for frac, lbl in [(0, 'start'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%')]:
        idx = min(int(frac * n), n - 1)
        print(f"  KN at {lbl:>5s} (t={t[idx]:.2f}s): {kn[idx]:.1f}")

    return t, kn, p


print("Loading and running 3DFMM motor...")
m3d, r3d = load_and_run('3dfmm_testing/long_fin_bates.ric')
t3, kn3, p3 = summary("3DFMM (finocyl)", m3d, r3d)

print("\nLoading and running analytical motor...")
mA, rA = load_and_run('3dfmm_testing/long_fin_bates_analytical.ric')
tA, knA, pA = summary("Analytical", mA, rA)

# SA vs regression comparison
print(f"\n{'='*60}")
print("  SA vs regression depth")
print(f"{'='*60}")
print(f"  {'reg_mm':>6s}  {'Analytical':>12s}  {'3DFMM':>12s}  {'ratio':>8s}")

# Set up 3DFMM grain for SA query
g3 = m3d.grains[0]
g3.simulationSetup(m3d.config)

for reg_mm in range(0, 26, 1):
    reg = reg_mm / 1000.0
    sa_a = sum(g.getSurfaceAreaAtRegression(reg) for g in mA.grains)
    sa_3 = g3.getSurfaceAreaAtRegression(reg)
    ratio = sa_3 / sa_a if sa_a > 0 else float('inf')
    marker = " <--" if abs(ratio - 1.0) > 0.15 else ""
    print(f"  {reg_mm:6d}  {sa_a:12.6f}  {sa_3:12.6f}  {ratio:8.3f}{marker}")

# Volume vs regression comparison
print(f"\n{'='*60}")
print("  Volume vs regression depth")
print(f"{'='*60}")
print(f"  {'reg_mm':>6s}  {'Analytical':>12s}  {'3DFMM':>12s}  {'ratio':>8s}")

for reg_mm in range(0, 26, 1):
    reg = reg_mm / 1000.0
    vol_a = sum(g.getVolumeAtRegression(reg) for g in mA.grains)
    vol_3 = g3.getVolumeAtRegression(reg)
    ratio = vol_3 / vol_a if vol_a > 0 else float('inf')
    marker = " <--" if abs(ratio - 1.0) > 0.15 else ""
    print(f"  {reg_mm:6d}  {vol_a:12.6e}  {vol_3:12.6e}  {ratio:8.3f}{marker}")

# KN curves side by side at common time points
print(f"\n{'='*60}")
print("  KN vs time (side-by-side)")
print(f"{'='*60}")
print(f"  {'t(s)':>6s}  {'KN_Analytical':>14s}  {'KN_3DFMM':>10s}  {'ratio':>8s}")

# Interpolate to common time base
t_max = min(t3[-1], tA[-1])
t_common = np.arange(0, t_max, 0.1)
kn3_interp = np.interp(t_common, t3, kn3)
knA_interp = np.interp(t_common, tA, knA)

for i in range(len(t_common)):
    ratio = kn3_interp[i] / knA_interp[i] if knA_interp[i] > 0 else float('inf')
    print(f"  {t_common[i]:6.1f}  {knA_interp[i]:14.1f}  {kn3_interp[i]:10.1f}  {ratio:8.3f}")
