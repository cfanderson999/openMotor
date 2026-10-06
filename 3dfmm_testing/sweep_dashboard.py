#!/usr/bin/env python3
"""Builds a single-file interactive dashboard from the output of performance_quality_sweep.py.

  python 3dfmm_testing/sweep_dashboard.py --out 3dfmm_testing/out/sweep

Reads <out>/results.jsonl and sweep_dashboard.html.in, writes <out>/dashboard.html. All runs of a grain and propellant
are resampled onto one time grid so the page can compare them at the same instant.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_quality_sweep import GRAINS, PROPELLANTS  # noqa: E402

GRID_POINTS = 160
FLUX_MODES = ['3d-2dflux', '3d-area', '3d-legacy-Exact']
COLUMNS = ['grain', 'prop', 'mode', 'dim', 'burn', 'peakP', 'avgP', 'peakKn', 'impulse', 'peakFlux', 'maxMach',
           'setup', 'loop', 'flux', 'calls', 'rss', 'voxels', 'steps']


def clean(values, digits):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


def build(out):
    rows = [json.loads(line) for line in (Path(out) / 'results.jsonl').read_text().splitlines()]
    rows = [r for r in rows if 'failed' not in r]
    index = {(r['grain'], r['propellant'], r['mode'], r['mapDim']): r for r in rows}
    grains = [g for g in GRAINS if any(r['grain'] == g for r in rows)]

    summary = []
    for r in rows:
        baseline = r['mode'] == 'baseline'
        summary.append([
            r['grain'], r['propellant'], r['mode'], r['mapDim'], round(r['burnTime'], 3), round(r['peakPressure'] / 1e6, 4),
            round(r['avgPressure'] / 1e6, 4), round(r['peakKn'], 2), round(r['impulse'], 2), round(r['peakFlux'], 2),
            round(r['maxMach'], 4), round(r['setupSeconds'] if baseline else r['gridSetupSeconds'], 3), round(r['loopSeconds'], 3),
            round(r['fluxSeconds'], 4), r['fluxCalls'], 0 if baseline else round(r['peakRssMb'], 1), 0 if baseline else r['voxels'],
            r['steps']])

    grid = {}
    for grain in grains:
        dims = sorted({r['mapDim'] for r in rows if r['grain'] == grain and r['mode'] != 'baseline'})
        grid[grain] = {}
        for propellant in PROPELLANTS:
            base = index.get((grain, propellant, 'baseline', 750))
            if not base:
                continue
            runs = {d: index.get((grain, propellant, '3d-area', d)) for d in dims}
            ends = [base['series']['time'][-1]] + [r['series']['time'][-1] for r in runs.values() if r]
            t = np.linspace(0, max(ends), GRID_POINTS)

            def resample(run, channel, scale, digits):
                time, values = np.array(run['series']['time']), np.array(run['series'][channel]) * scale
                resampled = np.interp(t, time, values, right=np.nan)
                resampled[t > time[-1] + 1e-9] = np.nan
                return clean(resampled, digits)

            entry = {'t': clean(t, 4),
                     'base': {'p': resample(base, 'pressure', 1e-6, 3), 'f': resample(base, 'force', 1.0, 1),
                              'q': resample(base, 'massFlux', 1.0, 1), 'm': resample(base, 'mach', 1.0, 4)},
                     'runs': {}}
            for d in dims:
                run = runs[d]
                if run:
                    entry['runs'][str(d)] = {'p': resample(run, 'pressure', 1e-6, 3), 'f': resample(run, 'force', 1.0, 1)}
            entry['flux'] = {}
            for d in dims:
                modes = {}
                for mode in FLUX_MODES:
                    run = index.get((grain, propellant, mode, d))
                    if run:
                        modes[mode] = {'q': resample(run, 'massFlux', 1.0, 1), 'm': resample(run, 'mach', 1.0, 4)}
                entry['flux'][str(d)] = modes
            grid[grain][propellant] = entry

    meta = {
        'grains': [{'id': g, 'type': GRAINS[g][0], 'length': GRAINS[g][1], 'ends': GRAINS[g][2].get('inhibitedEnds', ''),
                    'dims': sorted({r['mapDim'] for r in rows if r['grain'] == g and r['mode'] != 'baseline'})} for g in grains],
        'propellants': [{'id': p, 'n': PROPELLANTS[p]['tabs'][0]['n'], 'name': PROPELLANTS[p]['name']} for p in PROPELLANTS],
        'modes': FLUX_MODES, 'columns': COLUMNS, 'runs': len(rows),
    }
    return {'meta': meta, 'summary': summary, 'grid': grid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'out' / 'sweep'))
    args = parser.parse_args()
    data = build(args.out)
    template = (Path(__file__).resolve().parent / 'sweep_dashboard.html.in').read_text()
    html = template.replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    body = Path(args.out) / 'dashboard_body.html'
    body.write_text(html)  # The part an artifact page needs; the publisher adds the document wrapper
    page = Path(args.out) / 'dashboard.html'
    page.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                    '</head><body style="margin:0">' + html + '</body></html>')
    for path in (body, page):
        print('wrote', path, '{:.1f} MB'.format(path.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
