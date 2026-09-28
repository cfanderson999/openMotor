import csv, numpy as np, sys
sys.path.insert(0, '.')

with open('3dfmm_testing/out/long_fin_bates_3dfmm.csv') as f:
    d3 = list(csv.DictReader(f))

t = np.array([float(r['Time(s)']) for r in d3])
thrust = np.array([float(r['Thrust(N)']) for r in d3])
mass = np.array([float(r['Propellant Mass(G1;lb)']) for r in d3])
kn = np.array([float(r['Kn']) for r in d3])
web = np.array([float(r['Web(G1;in)']) for r in d3])
dt = 0.01

# Instantaneous Isp at selected timesteps
print('t       thrust   mass(lb)  dm(lb)    Isp     KN      web(in)')
for i in [1,50,100,150,200,250,300,350,400,450,500,510,520,525,529]:
    if i >= len(t): break
    dm = mass[max(0,i-1)] - mass[i]
    dm_kg = dm * 0.453592
    isp = thrust[i]/(dm_kg/dt*9.80665) if dm_kg>1e-10 else float('inf')
    print(f'{t[i]:6.2f}  {thrust[i]:8.1f} {mass[i]:9.5f} {dm:8.5f} {isp:7.1f}  {kn[i]:7.2f}  {web[i]:7.5f}')

# Impulse budget
imp_mass = sum(thrust[i]*dt for i in range(1,len(t)) if mass[i-1]-mass[i]>1e-8)
imp_free = sum(thrust[i]*dt for i in range(1,len(t)) if mass[i-1]-mass[i]<=1e-8)
total = float(np.trapz(thrust, t))
print(f'\nTotal impulse:      {total:.1f} Ns')
print(f'With mass loss:     {imp_mass:.1f} Ns')
print(f'Without mass loss:  {imp_free:.1f} Ns ({imp_free/total*100:.1f}%)')

last_mass_idx = np.where(mass>0.001)[0]
last_thrust_idx = np.where(thrust>1)[0]
if len(last_mass_idx) and len(last_thrust_idx):
    print(f'Mass ~0 at t={t[last_mass_idx[-1]]:.2f}, thrust ends t={t[last_thrust_idx[-1]]:.2f}')
    print(f'Extra burn duration: {t[last_thrust_idx[-1]]-t[last_mass_idx[-1]]:.2f}s')
