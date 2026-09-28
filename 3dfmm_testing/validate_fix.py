"""Validate the SA-from-volume fix by running both motors and comparing Isp."""
import sys, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uilib.fileIO import loadFile, fileTypes
from motorlib.motor import Motor

def load_motor(path):
    data = loadFile(str(path), fileTypes.MOTOR)
    return Motor(data)

def run_motor(m):
    res = m.runSimulation()
    return res

def summarize(label, res):
    t = np.array(res.channels['time'].getData())
    thrust = np.array(res.channels['force'].getData())
    mass0 = res.channels['mass'].getData()[0]
    if isinstance(mass0, list):
        masses = np.array([sum(row) for row in res.channels['mass'].getData()])
    else:
        masses = np.array(res.channels['mass'].getData())
    dt_arr = np.diff(t)
    total_impulse = float(np.trapz(thrust, t))
    prop_mass = masses[0] - masses[-1]  # kg
    if prop_mass > 0:
        isp = total_impulse / (prop_mass * 9.80665)
    else:
        isp = float('inf')
    peak_p = max(res.channels['pressure'].getData())
    print(f'\n--- {label} ---')
    print(f'  Propellant mass: {prop_mass:.4f} kg')
    print(f'  Total impulse:   {total_impulse:.1f} Ns')
    print(f'  Delivered Isp:   {isp:.1f} s')
    print(f'  Peak pressure:   {peak_p/1e6:.2f} MPa')
    print(f'  Burn time:       {t[-1]:.2f} s')

    # Instantaneous Isp samples
    print(f'  Instantaneous Isp at selected timesteps:')
    for i in [1, len(t)//4, len(t)//2, 3*len(t)//4, len(t)-2]:
        if i >= len(t) or i < 1:
            continue
        dm = masses[i-1] - masses[i]
        dt_step = t[i] - t[i-1]
        if dm > 1e-10 and dt_step > 0:
            inst_isp = thrust[i] / (dm / dt_step * 9.80665)
        else:
            inst_isp = float('inf')
        print(f'    t={t[i]:.3f}s: thrust={thrust[i]:.1f}N, dm={dm:.6f}kg, Isp={inst_isp:.1f}s')

print('Loading 3DFMM motor...')
m3d = load_motor('3dfmm_testing/long_fin_bates.ric')
print('Running 3DFMM simulation...')
r3d = run_motor(m3d)
summarize('3DFMM (finocyl)', r3d)

print('\nLoading analytical motor...')
mA = load_motor('3dfmm_testing/long_fin_bates_analytical.ric')
print('Running analytical simulation...')
rA = run_motor(mA)
summarize('Analytical', rA)
