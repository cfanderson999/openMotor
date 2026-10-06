#!/usr/bin/env python3
"""Performance vs quality sweep for simulating standard grains with 3D FMM.

Every grain is simulated with several propellants in these modes:
  baseline         the normal analytic or 2D FMM simulation (config mapDim)
  3d-2dflux        converted to 3D, mass flux from the total burning area and the aft port (massFlux3D off)
  3d-area          converted to 3D, mass flux from the per-layer burning area distribution
  3d-legacy-<Q>    converted to 3D, volume fraction mass flux with meshedMassFluxQuality Q (Exact, Fast, Faster)
at several 3D map dimensions. Each (grain, map dimension) runs in its own process with cold caches so setup time and
peak memory are clean. Results are appended to a JSON lines file and a resumable run skips finished units.

  python 3dfmm_testing/performance_quality_sweep.py run --out 3dfmm_testing/out/sweep
  python 3dfmm_testing/performance_quality_sweep.py report --out 3dfmm_testing/out/sweep
"""

import argparse
import functools
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIAMETER = 0.08
SHORT, LONG = 0.15, 0.48
TARGET_KN = 250
BASELINE_MAP_DIM = 750
CURVE_POINTS = 100

# name: (grain type, length, properties, 3D map dimensions)
STANDARD_DIMS = [64, 96, 128, 192, 256]
LONG_DIMS = [64, 96, 128]
GRAINS = {
    'bates-both': ('BATES', SHORT, {'coreDiameter': 0.03, 'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'bates-neither': ('BATES', SHORT, {'coreDiameter': 0.03, 'inhibitedEnds': 'Neither'}, STANDARD_DIMS),
    'bates-top': ('BATES', SHORT, {'coreDiameter': 0.03, 'inhibitedEnds': 'Top'}, STANDARD_DIMS),
    'conical-narrow-aft': ('Conical', SHORT, {'forwardCoreDiameter': 0.045, 'aftCoreDiameter': 0.02,
                                              'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'conical-wide-aft': ('Conical', SHORT, {'forwardCoreDiameter': 0.02, 'aftCoreDiameter': 0.045,
                                            'inhibitedEnds': 'Neither'}, STANDARD_DIMS),
    'finocyl': ('Finocyl', SHORT, {'coreDiameter': 0.02, 'numFins': 4, 'finWidth': 0.004, 'finLength': 0.015,
                                   'invertedFins': False, 'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'moon-burner': ('Moon Burner', SHORT, {'coreDiameter': 0.025, 'coreOffset': 0.012, 'inhibitedEnds': 'Both'},
                    STANDARD_DIMS),
    'star': ('Star Grain', SHORT, {'numPoints': 5, 'pointLength': 0.025, 'pointWidth': 0.006,
                                   'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'x-core': ('X Core', SHORT, {'slotWidth': 0.006, 'slotLength': 0.03, 'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'c-grain': ('C Grain', SHORT, {'slotWidth': 0.01, 'slotOffset': 0.01, 'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'd-grain': ('D Grain', SHORT, {'slotOffset': 0.01, 'inhibitedEnds': 'Both'}, STANDARD_DIMS),
    'rod-tube': ('Rod and Tube', SHORT, {'coreDiameter': 0.05, 'rodDiameter': 0.03, 'supportDiameter': 0.01,
                                         'inhibitedEnds': 'Neither'}, STANDARD_DIMS),
    'bates-long': ('BATES', LONG, {'coreDiameter': 0.03, 'inhibitedEnds': 'Both'}, LONG_DIMS),
    'finocyl-long': ('Finocyl', LONG, {'coreDiameter': 0.02, 'numFins': 4, 'finWidth': 0.004, 'finLength': 0.015,
                                       'invertedFins': False, 'inhibitedEnds': 'Both'}, LONG_DIMS),
}

# The first three come from the test motors in this folder; the last has a high pressure exponent.
PROPELLANTS = {
    'blue-thunder': {'name': 'Blue Thunder', 'density': 1625, 'tabs': [
        {'minPressure': 0, 'maxPressure': 1.0342e7, 'a': 6.99e-05, 'n': 0.321, 'k': 1.235, 't': 2616.5, 'm': 22.959}]},
    'cherry-limeade': {'name': 'Cherry Limeade', 'density': 1670, 'tabs': [
        {'minPressure': 0, 'maxPressure': 1.0342e7, 'a': 3.52e-05, 'n': 0.3273, 'k': 1.21, 't': 2800.0, 'm': 23.67}]},
    'wr1': {'name': 'WR1', 'density': 1681, 'tabs': [
        {'minPressure': 0, 'maxPressure': 1.0342e7, 'a': 5.44e-06, 'n': 0.4564, 'k': 1.2031, 't': 2882.0, 'm': 24.478}]},
    'high-n': {'name': 'High n', 'density': 1700, 'tabs': [
        {'minPressure': 0, 'maxPressure': 1.0342e7, 'a': 5.6e-07, 'n': 0.62, 'k': 1.22, 't': 2900.0, 'm': 24.0}]},
}

FLUX_MODES = ['3d-2dflux', '3d-area', '3d-legacy-Exact', '3d-legacy-Fast', '3d-legacy-Faster']
SECTION = '=' * 100


def buildMotor(grains, config, propellant, throat):
    import motorlib.motor
    import motorlib.propellant
    motor = motorlib.motor.Motor()
    motor.config = config
    motor.nozzle.setProperties({'throat': throat, 'exit': throat * 2.2, 'efficiency': 0.95, 'divAngle': 15, 'convAngle': 45,
                                'throatLength': 0.005, 'slagCoeff': 0, 'erosionCoeff': 0})
    motor.propellant = motorlib.propellant.Propellant(PROPELLANTS[propellant])
    motor.grains = grains
    return motor


def makeConfig(mapDim3D):
    import motorlib.motor
    config = motorlib.motor.MotorConfig()
    config.setProperties({'maxPressure': 7e7, 'maxMassFlux': 1e4, 'maxMachNumber': 1, 'minPortThroat': 1,
                          'flowSeparationWarnPercent': 0.05, 'burnoutWebThres': 2.54e-5, 'burnoutThrustThres': 0.1,
                          'timestep': 0.03, 'ambPressure': 101325, 'mapDim': BASELINE_MAP_DIM, '3DmapDim': mapDim3D,
                          'sepPressureRatio': 0.4, 'convertTo3D': False})
    return config


def makeGrain(name):
    import motorlib.grains
    geomName, length, props, _ = GRAINS[name]
    grain = motorlib.grains.grainTypes[geomName]()
    grain.setProperties({'diameter': DIAMETER, 'length': length, **props})
    return grain


@functools.lru_cache(maxsize=None)
def sizeThroat(name):
    """Throat diameter that gives the grain an initial Kn of TARGET_KN."""
    grain = makeGrain(name)
    grain.simulationSetup(makeConfig(64))
    return math.sqrt(4 * grain.getSurfaceAreaAtRegression(0) / (math.pi * TARGET_KN))


class Timer:
    """Accumulates the wall time of the calls to a method it wraps."""

    def __init__(self, owner, methodName):
        self.owner, self.methodName, self.seconds, self.calls = owner, methodName, 0.0, 0
        self.original = getattr(owner, methodName)

    def __enter__(self):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return self.original(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - start
                self.calls += 1
        setattr(self.owner, self.methodName, timed)
        return self

    def __exit__(self, *exc):
        delattr(self.owner, self.methodName)


def runOnce(motor, timedGrain):
    """Runs a simulation and returns its metrics and timings, with grain setup time separated out."""
    with Timer(timedGrain, 'simulationSetup') as setup, Timer(timedGrain, 'getPeakMassFlux') as flux:
        start = time.perf_counter()
        result = motor.runSimulation()
        total = time.perf_counter() - start
    if not result.success:
        return {'failed': [a.description for a in result.alerts][:3]}
    time_ = np.array(result.channels['time'].getData())
    pressure = np.array(result.channels['pressure'].getData())
    burn = result.getBurnTime()
    curve = np.interp(np.linspace(0, 1, CURVE_POINTS) * burn, time_, pressure)
    return {
        'burnTime': burn, 'peakPressure': result.getMaxPressure(), 'avgPressure': result.getAveragePressure(),
        'peakKn': result.getPeakKN(), 'impulse': result.getImpulse(), 'peakFlux': result.getPeakMassFlux(),
        'maxMach': float(np.max(result.channels['machNumber'].getData())), 'curve': curve.tolist(),
        'series': {'time': time_, 'pressure': pressure, 'force': np.array(result.channels['force'].getData()),
                   'kn': np.array(result.channels['kn'].getData()),
                   'massFlux': np.array([row[0] for row in result.channels['massFlux'].getData()]),
                   'mach': np.array([row[0] for row in result.channels['machNumber'].getData()])},
        'steps': len(time_), 'setupSeconds': setup.seconds, 'loopSeconds': total - setup.seconds,
        'fluxSeconds': flux.seconds, 'fluxCalls': flux.calls,
    }


def peakRssMb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def disableDistanceCache():
    from motorlib.grain import Fmm3DGrain
    Fmm3DGrain._loadFmmCache = lambda self, key: None
    Fmm3DGrain._saveFmmCache = lambda self, key, data: None


def worker(args):
    disableDistanceCache()
    throat = args.throat
    rows = []
    if args.mode == 'baseline':
        config = makeConfig(64)
        grain = makeGrain(args.grain)
        for propellant in PROPELLANTS:
            motor = buildMotor([grain], config, propellant, throat)
            rows.append({'propellant': propellant, 'mode': 'baseline', 'mapDim': BASELINE_MAP_DIM,
                         **runOnce(motor, grain)})
    else:
        config = makeConfig(args.map_dim)
        source = makeGrain(args.grain)
        converted = source.convertTo3D()
        start = time.perf_counter()
        converted.simulationSetup(config)
        setupSeconds = time.perf_counter() - start
        rss = peakRssMb()
        issue = converted.getGeometryIssue(config)
        savedLayers = (converted._layerLevels, converted._layerAreas)
        for mode in FLUX_MODES:
            converted.props['massFlux3D'].setValue(mode != '3d-2dflux')
            converted._layerLevels, converted._layerAreas = savedLayers if mode == '3d-area' else (None, None)
            if mode.startswith('3d-legacy-'):
                converted.props['meshedMassFluxQuality'].setValue(mode.split('-')[-1])
            for propellant in PROPELLANTS:
                motor = buildMotor([converted], config, propellant, throat)
                rows.append({'propellant': propellant, 'mode': mode, 'mapDim': args.map_dim,
                             'gridSetupSeconds': setupSeconds, 'peakRssMb': rss, 'gateIssue': issue,
                             'voxels': int(np.prod(converted.coreMap.shape)), **runOnce(motor, converted)})
    with open(args.result, 'a') as out:
        for row in rows:
            if 'series' in row:
                row['series'] = {k: [float('{:.6g}'.format(v)) for v in values] for k, values in row['series'].items()}
            out.write(json.dumps({'grain': args.grain, **row}) + '\n')


def finishedUnits(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add((row['grain'], row['mapDim'], row['mode'] == 'baseline'))
    return done


def runSweep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = out / 'results.jsonl'
    done = finishedUnits(result)
    units = []
    for name, (_, _, _, dims) in GRAINS.items():
        if args.grains and name not in args.grains:
            continue
        units.append((name, 'baseline', 0))
        units += [(name, '3d', d) for d in dims if d <= args.max_map_dim]
    started = time.time()
    for index, (name, mode, mapDim) in enumerate(units):
        if (name, BASELINE_MAP_DIM if mode == 'baseline' else mapDim, mode == 'baseline') in done:
            continue
        throat = sizeThroat(name)
        command = [sys.executable, __file__, 'worker', '--grain', name, '--mode', mode, '--map-dim', str(mapDim),
                   '--throat', repr(throat), '--result', str(result)]
        unitStart = time.time()
        try:
            subprocess.run(command, check=True, timeout=args.timeout, env={**os.environ, 'PYTHONPATH': str(REPO_ROOT)})
            status = 'ok'
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
        except subprocess.CalledProcessError as error:
            status = 'FAILED rc={}'.format(error.returncode)
        print('[{:>5.0f}s] {}/{} {} {} {} ({:.0f}s)'.format(time.time() - started, index + 1, len(units), name, mode,
                                                               mapDim, time.time() - unitStart), status, flush=True)


def loadRows(out):
    rows = [json.loads(line) for line in (Path(out) / 'results.jsonl').read_text().splitlines()]
    return [r for r in rows if 'failed' not in r]


METRICS = ['burnTime', 'peakPressure', 'peakKn', 'impulse', 'peakFlux', 'maxMach']


def relativeError(value, reference):
    return abs(value / reference - 1) if reference else float('nan')


def curveError(curve, reference):
    curve, reference = np.array(curve), np.array(reference)
    return float(np.sqrt(np.mean(((curve - reference) / reference.max()) ** 2)))


def report(args):
    rows = loadRows(args.out)
    baseline = {(r['grain'], r['propellant']): r for r in rows if r['mode'] == 'baseline'}
    lines = []

    # Error against the normal simulation, median over propellants
    table = {}
    for r in rows:
        if r['mode'] == 'baseline' or (r['grain'], r['propellant']) not in baseline:
            continue
        ref = baseline[(r['grain'], r['propellant'])]
        errors = {m: relativeError(r[m], ref[m]) for m in METRICS}
        errors['curve'] = curveError(r['curve'], ref['curve'])
        table.setdefault((r['grain'], r['mode'], r['mapDim']), []).append((errors, r))

    def median(key, field):
        values = [e[field] for e, _ in table[key]]
        return float(np.median(values))

    def seconds(key, field):
        return float(np.median([r[field] for _, r in table[key]]))

    grains = [g for g in GRAINS if any(k[0] == g for k in table)]
    lines += ['# Performance vs quality sweep', '',
              'Errors are relative to the normal analytic or 2D FMM simulation (median over {} propellants). '
              'Times are medians per simulation.'.format(len(PROPELLANTS)), '']
    for grain in grains:
        dims = sorted({k[2] for k in table if k[0] == grain})
        lines += ['## ' + grain, '']
        base = [baseline[(grain, p)] for p in PROPELLANTS if (grain, p) in baseline]
        lines += ['baseline setup+loop {:.2f}s, {} steps'.format(np.median([b['setupSeconds'] + b['loopSeconds'] for b in base]),
                                                                  int(np.median([b['steps'] for b in base]))), '']
        for mode in FLUX_MODES:
            if not any((grain, mode, d) in table for d in dims):
                continue
            lines += ['**{}**'.format(mode), '',
                      '| map dim | setup s | loop s | flux s | RSS MB | burn | pressure | Kn | impulse | flux | Mach | curve |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for d in dims:
                key = (grain, mode, d)
                if key not in table:
                    continue
                r0 = table[key][0][1]
                errors = ['{:.1%}'.format(median(key, m)) for m in METRICS + ['curve']]
                lines.append('| {} | {:.1f} | {:.2f} | {:.2f} | {:.0f} | {} |'.format(
                    d, r0['gridSetupSeconds'], seconds(key, 'loopSeconds'), seconds(key, 'fluxSeconds'), r0['peakRssMb'],
                    ' | '.join(errors)))
            lines.append('')

    # Stagnation: error that stops falling as the map gets finer
    lines += ['## Quality stagnation', '',
              'A metric stagnates when its error at the finest map dimension is above 1% and at least 70% of the error at '
              'the map dimension one step coarser.', '',
              '| grain | mode | metric | errors by map dim | finest / previous |', '|---|---|---|---|---|']
    flagged = 0
    for grain in grains:
        for mode in FLUX_MODES:
            dims = sorted(d for (g, m, d) in table if g == grain and m == mode)
            if len(dims) < 3:
                continue
            for metric in ['peakPressure', 'peakKn', 'impulse', 'peakFlux', 'curve']:
                errors = [median((grain, mode, d), metric) for d in dims]
                if errors[-1] > 0.01 and errors[-1] >= 0.7 * errors[-2]:
                    flagged += 1
                    lines.append('| {} | {} | {} | {} | {:.2f} |'.format(
                        grain, mode, metric, ', '.join('{}:{:.1%}'.format(d, e) for d, e in zip(dims, errors)),
                        errors[-1] / errors[-2] if errors[-2] else float('nan')))
    lines += ['', '{} stagnating series.'.format(flagged), '']

    # Cost scaling
    lines += ['## Cost scaling', '',
              'Power law fits, time ~ mapDim^p, over the standard grains (median).', '',
              '| quantity | exponent |', '|---|---|']
    for label, getter in (('grid setup', lambda r: r['gridSetupSeconds']), ('loop (3d-2dflux)', None),
                          ('loop (3d-area)', None), ('loop (3d-legacy-Exact)', None), ('peak RSS', lambda r: r['peakRssMb'])):
        points = {}
        for r in rows:
            if r['mode'] == 'baseline' or GRAINS[r['grain']][1] != SHORT:
                continue
            if label.startswith('loop'):
                if r['mode'] != label[6:-1]:
                    continue
                value = r['loopSeconds']
            else:
                value = getter(r)
            points.setdefault(r['mapDim'], []).append(value)
        if len(points) >= 3:
            dims = sorted(points)
            slope = np.polyfit(np.log(dims), np.log([np.median(points[d]) for d in dims]), 1)[0]
            lines.append('| {} | {:.2f} |'.format(label, slope))
    lines.append('')

    # Quality modes in the new flux path
    lines += ['## Does meshedMassFluxQuality still matter?', '',
              '| grain | map dim | 3d-area peak flux | legacy Exact | legacy Fast | legacy Faster |', '|---|---|---|---|---|---|']
    for grain in grains:
        for d in sorted({k[2] for k in table if k[0] == grain})[-2:]:
            cells = []
            for mode in ('3d-area', '3d-legacy-Exact', '3d-legacy-Fast', '3d-legacy-Faster'):
                key = (grain, mode, d)
                cells.append('{:.1%}'.format(median(key, 'peakFlux')) if key in table else '-')
            lines.append('| {} | {} | {} |'.format(grain, d, ' | '.join(cells)))
    lines.append('')
    rejected = sorted({(r['grain'], r['mapDim'], r['gateIssue']) for r in rows if r.get('gateIssue')})
    if rejected:
        lines += ['## Conversions the safety net would reject', '']
        lines += ['- {} at {}: {}'.format(*r) for r in rejected]
    path = Path(args.out) / 'report.md'
    path.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print('\nwritten to', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--out', default=str(REPO_ROOT / '3dfmm_testing' / 'out' / 'sweep'))
    run.add_argument('--grains', nargs='*')
    run.add_argument('--max-map-dim', type=int, default=256)
    run.add_argument('--timeout', type=int, default=3600)
    rep = sub.add_parser('report')
    rep.add_argument('--out', default=str(REPO_ROOT / '3dfmm_testing' / 'out' / 'sweep'))
    work = sub.add_parser('worker')
    work.add_argument('--grain')
    work.add_argument('--mode')
    work.add_argument('--map-dim', type=int)
    work.add_argument('--throat', type=float)
    work.add_argument('--result')
    args = parser.parse_args()
    {'run': runSweep, 'report': report, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    main()
