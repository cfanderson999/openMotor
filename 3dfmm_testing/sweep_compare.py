#!/usr/bin/env python3
"""Compares two runs of performance_quality_sweep.py, judging burn time, impulse, average pressure and the pressure curve
only while chamber pressure is above a threshold (200 psi by default), since the low pressure taildown near burnout is
neither important nor well characterized.

  python 3dfmm_testing/sweep_compare.py --old 3dfmm_testing/out/sweep --new 3dfmm_testing/out/sweep_v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_quality_sweep import GRAINS, PROPELLANTS  # noqa: E402

PSI = 6894.757
THRESHOLD_PA = 200 * PSI
WINDOWED = ['burnTime', 'impulse', 'avgPressure', 'curve']
WHOLE_RUN = ['peakPressure', 'peakKn', 'peakFlux', 'maxMach']
METRICS = WINDOWED + WHOLE_RUN
LABELS = {'burnTime': 'burn time', 'impulse': 'impulse', 'avgPressure': 'average pressure', 'curve': 'pressure curve RMS',
          'peakPressure': 'peak pressure', 'peakKn': 'peak Kn', 'peakFlux': 'peak mass flux', 'maxMach': 'max Mach'}


def load(path):
    rows = [json.loads(line) for line in (Path(path) / 'results.jsonl').read_text().splitlines()]
    return {(r['grain'], r['propellant'], r['mode'], r['mapDim']): r for r in rows if 'failed' not in r}


def window(series, threshold=THRESHOLD_PA):
    """Index range of the first stretch of the burn where pressure is at or above the threshold."""
    pressure = np.array(series['pressure'])
    above = np.nonzero(pressure >= threshold)[0]
    if not len(above):
        return 0, 0
    start = int(above[0])
    below = np.nonzero(pressure[start:] < threshold)[0]
    return start, int(start + (below[0] - 1 if len(below) else len(pressure) - start - 1))


def windowedValues(run, threshold=THRESHOLD_PA):
    s = {k: np.array(v) for k, v in run['series'].items()}
    a, b = window(s, threshold)
    t, p, f = s['time'][a:b + 1], s['pressure'][a:b + 1], s['force'][a:b + 1]
    span = max(t[-1] - t[0], 1e-9)
    return {'burnTime': float(t[-1]), 'impulse': float(np.trapz(f, t)), 'avgPressure': float(np.trapz(p, t) / span)}


def curveRms(run, base, threshold=THRESHOLD_PA):
    b = {k: np.array(v) for k, v in base['series'].items()}
    r = {k: np.array(v) for k, v in run['series'].items()}
    a, e = window(b, threshold)
    grid = np.linspace(b['time'][a], b['time'][e], 120)
    pb, pr = np.interp(grid, b['time'], b['pressure']), np.interp(grid, r['time'], r['pressure'])
    return float(np.sqrt(np.mean(((pr - pb) / pb.max()) ** 2)) * 100)


def errors(run, base):
    wr, wb = windowedValues(run), windowedValues(base)
    out = {m: abs(wr[m] / wb[m] - 1) * 100 for m in ('burnTime', 'impulse', 'avgPressure')}
    out['curve'] = curveRms(run, base)
    for m in WHOLE_RUN:
        out[m] = abs(run[m] / base[m] - 1) * 100
    return out


def table(rows, mode='3d-area'):
    """(grain, mapDim) -> metric -> median over propellants of the error against the baseline."""
    result = {}
    for (grain, propellant, runMode, dim), run in rows.items():
        if runMode != mode:
            continue
        base = rows.get((grain, propellant, 'baseline', 750))
        if base:
            result.setdefault((grain, dim), []).append(errors(run, base))
    return {k: {m: float(np.median([e[m] for e in v])) for m in METRICS} for k, v in result.items()}


def slope(dims, values):
    """Log-log slope of error against map dimension; negative means the error falls as the map gets finer."""
    values = np.maximum(np.array(values), 0.05)
    return float(np.polyfit(np.log(dims), np.log(values), 1)[0]) if len(dims) >= 3 else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', default=str(Path(__file__).resolve().parent / 'out' / 'sweep'))
    parser.add_argument('--new', default=str(Path(__file__).resolve().parent / 'out' / 'sweep_v2'))
    parser.add_argument('--threshold-psi', type=float, default=200)
    args = parser.parse_args()
    global THRESHOLD_PA
    THRESHOLD_PA = args.threshold_psi * PSI
    old = table(load(args.old))
    new = table(load(args.new)) if (Path(args.new) / 'results.jsonl').exists() else {}
    grains = [g for g in GRAINS if any(k[0] == g for k in old)]
    dims = sorted({k[1] for k in old})
    print('Median error against the normal simulation [%], pressure above {:.0f} psi for the windowed measures'.format(args.threshold_psi))
    for label, data in (('OLD', old), ('NEW', new)):
        if not data:
            continue
        print('\n' + label)
        print('{:>18s} '.format('measure') + ' '.join('{:>6d}'.format(d) for d in dims) + '   slope')
        for m in METRICS:
            medians = [float(np.median([data[(g, d)][m] for g in grains if (g, d) in data])) for d in dims if any((g, d) in data for g in grains)]
            have = [d for d in dims if any((g, d) in data for g in grains)]
            print('{:>18s} '.format(LABELS[m]) + ' '.join('{:6.2f}'.format(v) for v in medians) + '   {:5.2f}'.format(slope(have, medians)))
    if new:
        print('\nPer grain at map 128: old -> new')
        for g in grains:
            if (g, 128) in old and (g, 128) in new:
                print('{:>20s}  '.format(g) + '  '.join('{} {:4.1f}->{:4.1f}'.format(m[:5], old[(g, 128)][m], new[(g, 128)][m]) for m in METRICS))
        print('\nGrain stagnation (error slope vs map dim, impulse/avg P/curve/peak P), old -> new')
        for g in grains:
            have = [d for d in dims if (g, d) in old and (g, d) in new]
            if len(have) >= 3:
                print('{:>20s}  '.format(g) + '  '.join('{} {:5.2f}->{:5.2f}'.format(m[:5], slope(have, [old[(g, d)][m] for d in have]), slope(have, [new[(g, d)][m] for d in have])) for m in ('impulse', 'avgPressure', 'curve', 'peakPressure')))


if __name__ == '__main__':
    main()
