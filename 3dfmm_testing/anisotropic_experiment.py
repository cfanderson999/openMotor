#!/usr/bin/env python3
"""Does a 3D map that is fine across the grain and coarse along it beat a cubic one at the same cost?

For each grain this builds the signed distance map with voxels dxy across and dz = aspect * dxy along the axis, measures
the burning area and propellant volume against regression from marching cubes areas, and compares them with the
reference (analytic, or the 2D FMM at the configured map dimension). The curves then run through a quasi-steady burn
with each propellant, so the errors show up as burn time, impulse, average pressure and pressure curve differences.

  python 3dfmm_testing/anisotropic_experiment.py run --out 3dfmm_testing/out/aniso
  python 3dfmm_testing/anisotropic_experiment.py report --out 3dfmm_testing/out/aniso
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import skfmm
from scipy import interpolate, ndimage
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import performance_quality_sweep as sweep  # noqa: E402
from mathlib._march import marching_area_pixels_by_layer  # noqa: E402
from motorlib.grains.conical import ConicalGrain  # noqa: E402

PSI = 6894.757
THRESHOLD = 200 * PSI
LEVELS = 56
TARGET_KN = 250
DEFAULT_GRAINS = ['bates-both', 'bates-neither', 'conical-narrow-aft', 'conical-wide-aft', 'finocyl', 'moon-burner', 'star', 'd-grain']
# (aspect, cross-section dimensions)
CONFIGS = [(1, [64, 96, 128, 192]), (2, [128, 192, 256]), (4, [128, 192, 256, 384]), (8, [192, 256, 384])]
SDF_CONFIGS = [(1, [64, 96, 128, 192]), (4, [128, 192])]


def referenceCurves(name):
    source = sweep.makeGrain(name)
    source.simulationSetup(sweep.makeConfig(64))
    web = source.getWebLeft(0)
    return source, web, (lambda r: source.getSurfaceAreaAtRegression(r)), (lambda r: source.getVolumeAtRegression(r))


HIGH_RES = 1024
_sdfCache = {}


def highResolutionSdf(source):
    """Signed distance of the grain cross section [m], positive in propellant, on a 1024 pixel grid."""
    key = id(source)
    if key not in _sdfCache:
        propellant, _ = source._getResampledFaceImage(HIGH_RES)
        diameter = source.props['diameter'].getValue()
        _sdfCache[key] = skfmm.distance(np.where(propellant, 1.0, -1.0), dx=diameter / HIGH_RES)
    return _sdfCache[key]


def subvoxelField(source, nxy, nz, dxy, dz, padLo, padHi):
    """Initial level set sampled from the exact (analytic or high resolution) geometry instead of a binary voxel mask.
    Positive in propellant, zero on the true surface, so the distance solve starts from a surface that is not
    quantized to the voxel size."""
    diameter = source.props['diameter'].getValue()
    length = source.props['length'].getValue()
    centers = ((2 * np.arange(nxy) + 1) / nxy - 1) * 0.5 * diameter
    yy, xx = np.meshgrid(centers, centers, indexing='ij')
    z = (np.arange(nz + padLo + padHi) - padLo + 0.5) * dz
    if isinstance(source, ConicalGrain):
        aft, fore = source.props['aftCoreDiameter'].getValue(), source.props['forwardCoreDiameter'].getValue()
        radius = 0.5 * (aft + (fore - aft) * np.clip(z / length, 0.0, 1.0))
        field = np.hypot(xx, yy)[None, :, :] - radius[:, None, None]
    else:
        sdf = highResolutionSdf(source)
        index = (centers / (0.5 * diameter) + 1.0) * 0.5 * HIGH_RES - 0.5
        plane = ndimage.map_coordinates(sdf, np.meshgrid(index, index, indexing='ij'), order=3, mode='nearest')
        field = np.broadcast_to(plane, (z.size, nxy, nxy)).copy()
    ends = source.props['inhibitedEnds'].getValue()
    if ends in ('Top', 'Neither'):
        field = np.minimum(field, z[:, None, None])
    if ends in ('Bottom', 'Neither'):
        field = np.minimum(field, (length - z)[:, None, None])
    return field


def buildCurves(source, nxy, aspect, init='binary'):
    """Area and volume against regression from a map with dz = aspect * dxy. Returns (curves, cost)."""
    diameter = source.props['diameter'].getValue()
    length = source.props['length'].getValue()
    ends = source.props['inhibitedEnds'].getValue()
    dxy = diameter / nxy
    nz = max(6, int(round(length / (aspect * dxy))))
    dz = length / nz
    outside = source.getExtrudedMask(nxy)
    padLo = 1 if ends in ('Top', 'Neither') else 0
    padHi = 1 if ends in ('Bottom', 'Neither') else 0
    if init == 'sdf':
        level = subvoxelField(source, nxy, nz, dxy, dz, padLo, padHi)
        core = level > 0
    else:
        core = source.getExtrudedCoreMap(nz, nxy)
        voidSlice = np.zeros((1, nxy, nxy), dtype=bool)
        if padLo:
            core = np.concatenate([voidSlice, core], axis=0)
        if padHi:
            core = np.concatenate([core, voidSlice], axis=0)
    valid2 = ~outside
    valid = np.broadcast_to(valid2, core.shape)
    voxels = int(core.size)

    start = time.perf_counter()
    seed = level if init == 'sdf' else np.where(core, 1.0, -1.0)
    phi = np.ma.MaskedArray(seed, np.broadcast_to(outside, core.shape))
    distance = skfmm.distance(phi, dx=[dz, dxy, dxy]).filled(1e9)
    fmmSeconds = time.perf_counter() - start

    start = time.perf_counter()
    _, (rows, cols) = ndimage.distance_transform_edt(outside, return_indices=True)
    field = np.ascontiguousarray(distance[:, rows, cols])
    maxDist = float(distance[valid].max())
    levels = np.linspace(0.0, maxDist, LEVELS)
    areas = np.array([marching_area_pixels_by_layer(field, valid, level, dz / dxy).sum() * dxy ** 2 for level in levels])
    mcSeconds = time.perf_counter() - start
    del phi, distance, field

    if init == 'sdf':  # Partial volume from the level set, so the wall and bore edges are not counted in whole voxels
        propellantVolume = float(np.sum(np.clip(level / min(dxy, dz) + 0.5, 0.0, 1.0)[valid])) * dxy ** 2 * dz
    else:
        propellantVolume = float(np.count_nonzero(core & valid)) * dxy ** 2 * dz
    grid = np.linspace(0.0, maxDist, max(int(math.ceil(maxDist / (0.5 * dxy))) + 1, 50))
    area = np.nan_to_num(interpolate.PchipInterpolator(levels, areas, extrapolate=False)(grid), nan=0.0)
    area = np.maximum(savgol_filter(area, 5, 3), 0.0)
    shortfall = (propellantVolume - np.trapz(area, grid)) / propellantVolume
    if shortfall > 0:
        lastArea = float(area[np.nonzero(area > 0.02 * area.max())[0][-1]])
        extra = (propellantVolume - np.trapz(area, grid)) / lastArea
        more = np.arange(1, int(math.ceil(extra / (0.5 * dxy))) + 1) * 0.5 * dxy
        grid = np.concatenate([grid, grid[-1] + more])
        area = np.concatenate([area, np.full(more.size, lastArea)])
        area[-1] = 0.0
    consumed = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(grid))])
    volume = np.maximum(propellantVolume - consumed, 0.0)
    return (grid, area, volume, shortfall), {'voxels': voxels, 'fmmSeconds': fmmSeconds, 'mcSeconds': mcSeconds,
                                              'seconds': fmmSeconds + mcSeconds, 'dxy': dxy, 'dz': dz, 'nz': nz,
                                              'estimatedMb': voxels * 8 * 3 / 1e6}


def burn(grid, area, throatArea, propellant, step=0.004):
    tab = sweep.PROPELLANTS[propellant]['tabs'][0]
    a, n, rho, k, T, M = tab['a'], tab['n'], sweep.PROPELLANTS[propellant]['density'], tab['k'], tab['t'], tab['m']
    gas = 8.314462 / (M / 1000.0)
    gamma = k
    cstar = math.sqrt(gas * T) / (math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
    areaAt = interpolate.interp1d(grid, area, bounds_error=False, fill_value=0.0)
    t, r, times, pressures = 0.0, 0.0, [], []
    while r < grid[-1] and len(times) < 40000:
        kn = float(areaAt(r)) / throatArea
        if kn <= 0.5:
            break
        pressure = (rho * a * cstar * kn) ** (1.0 / (1.0 - n))
        times.append(t)
        pressures.append(pressure)
        r += a * pressure ** n * step
        t += step
    times.append(t)
    pressures.append(0.0)
    return np.array(times), np.array(pressures)


def windowed(times, pressures):
    above = np.nonzero(pressures >= THRESHOLD)[0]
    if not len(above):
        return None
    start = int(above[0])
    below = np.nonzero(pressures[start:] < THRESHOLD)[0]
    end = int(start + (below[0] - 1 if len(below) else len(pressures) - start - 1))
    t, p = times[start:end + 1], pressures[start:end + 1]
    return {'burn': float(t[-1]), 'impulse': float(np.trapz(p, t)), 'avgP': float(np.trapz(p, t) / max(t[-1] - t[0], 1e-9)),
            'peakP': float(pressures.max()), 'window': (start, end)}


def metrics(curves, reference, web, throatArea):
    grid, area, volume, shortfall = curves
    rr = np.linspace(0.02 * web, 0.9 * web, 80)
    refArea = np.array([reference['area'](r) for r in rr])
    got = np.interp(rr, grid, area)
    out = {'areaRms': float(np.sqrt(np.mean(((got - refArea) / refArea.max()) ** 2)) * 100),
           'volumeErr': float(max(abs(np.interp(r, grid, volume) - reference['volume'](r)) / reference['volume'](0.0) for r in rr) * 100),
           'webErr': float((grid[-1] / web - 1) * 100), 'shortfall': float(shortfall * 100)}
    perPropellant = []
    for propellant in sweep.PROPELLANTS:
        refGrid = np.linspace(0, web, 400)
        tr, pr = burn(refGrid, np.array([reference['area'](r) for r in refGrid]), throatArea, propellant)
        tm, pm = burn(grid, area, throatArea, propellant)
        wr, wm = windowed(tr, pr), windowed(tm, pm)
        if not wr or not wm:
            continue
        # Shape only: each curve on its own burn time, so a small burn time difference does not misalign the burnout
        normalized = np.linspace(0.0, 0.97, 100)
        curve = np.sqrt(np.mean(((np.interp(normalized * wm['burn'], tm, pm) - np.interp(normalized * wr['burn'], tr, pr)) / pr.max()) ** 2)) * 100
        perPropellant.append({'burn': abs(wm['burn'] / wr['burn'] - 1) * 100, 'impulse': abs(wm['impulse'] / wr['impulse'] - 1) * 100,
                              'avgP': abs(wm['avgP'] / wr['avgP'] - 1) * 100, 'curve': float(curve),
                              'peakP': abs(wm['peakP'] / wr['peakP'] - 1) * 100})
    for key in ('burn', 'impulse', 'avgP', 'curve', 'peakP'):
        out[key] = float(np.median([p[key] for p in perPropellant])) if perPropellant else float('nan')
    return out


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.jsonl'
    done = {(r['grain'], r['aspect'], r['nxy'], r.get('init', 'binary')) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    for name in args.grains:
        source, web, area, volume = referenceCurves(name)
        reference = {'area': area, 'volume': volume}
        throatArea = area(0.0) / TARGET_KN
        plan = [(init, aspect, nxy) for init, configs in (('binary', CONFIGS), ('sdf', SDF_CONFIGS)) if init in args.inits
                for aspect, dims in configs for nxy in dims]
        for init, aspect, nxy in plan:
            if (name, aspect, nxy, init) in done or nxy > args.max_nxy:
                continue
            try:
                curves, cost = buildCurves(source, nxy, aspect, init)
                row = {'grain': name, 'aspect': aspect, 'nxy': nxy, 'init': init, **cost, **metrics(curves, reference, web, throatArea)}
            except MemoryError:
                row = {'grain': name, 'aspect': aspect, 'nxy': nxy, 'init': init, 'failed': 'memory'}
            with open(path, 'a') as handle:
                handle.write(json.dumps(row) + '\n')
            print('{:>20s} {} aspect {} xy {:>3d}: {}'.format(name, init, aspect, nxy, 'failed' if 'failed' in row else
                  '{:5.1f}M voxels {:5.1f}s | area {:4.2f}% web {:+5.2f}% impulse {:4.2f}% curve {:4.2f}%'.format(
                      row['voxels'] / 1e6, row['seconds'], row['areaRms'], row['webErr'], row['impulse'], row['curve'])), flush=True)


def report(args):
    allRows = [r for r in map(json.loads, (Path(args.out) / 'results.jsonl').read_text().splitlines()) if 'failed' not in r]
    rows = [r for r in allRows if r.get('init', 'binary') == 'binary']
    grains = [g for g in DEFAULT_GRAINS if any(r['grain'] == g for r in rows)] or sorted({r['grain'] for r in rows})
    targets = {'areaRms': 1.0, 'webErr_abs': 1.0, 'impulse': 1.0, 'curve': 1.0}
    lines = ['# Anisotropic voxel experiment', '',
             'Voxels are dxy across and dz = aspect x dxy along the grain. Errors are against the reference (analytic or 2D FMM) '
             'for the burning area curve (RMS, % of peak), the total burn distance (web) and a quasi-steady burn judged above 200 psi '
             '(median over 4 propellants; the curve error compares shape on each run\'s own normalized burn time).', '']
    for metric, target in targets.items():
        lines += ['## Cheapest map reaching {} error under {}%'.format(metric, target), '',
                  '| grain | cubic: setup s (voxels, config) | anisotropic: setup s (voxels, config) | time saved | voxels saved |', '|---|---|---|---|---|']
        timeRatios, voxelRatios = [], []
        for g in grains:
            value = (lambda r: abs(r['webErr'])) if metric == 'webErr_abs' else (lambda r: r[metric])
            sub = [r for r in rows if r['grain'] == g and value(r) < target]
            iso = min((r for r in sub if r['aspect'] == 1), key=lambda r: r['seconds'], default=None)
            aniso = min((r for r in sub if r['aspect'] > 1), key=lambda r: r['seconds'], default=None)
            fmt = lambda r: '{:.1f} ({:.1f}M, {})'.format(r['seconds'], r['voxels'] / 1e6, r['nxy']) if r else 'not reached'
            if iso and aniso:
                timeRatios.append(iso['seconds'] / aniso['seconds'])
                voxelRatios.append(iso['voxels'] / aniso['voxels'])
            lines.append('| {} | {} | {} (aspect {}, xy {}) | {} | {} |'.format(
                g, fmt(iso), '{:.1f} ({:.1f}M)'.format(aniso['seconds'], aniso['voxels'] / 1e6) if aniso else 'not reached',
                aniso['aspect'] if aniso else '-', aniso['nxy'] if aniso else '-',
                '{:.1f}x'.format(iso['seconds'] / aniso['seconds']) if iso and aniso else '-',
                '{:.1f}x'.format(iso['voxels'] / aniso['voxels']) if iso and aniso else '-'))
        if timeRatios:
            lines.append('\nmedian time saved {:.1f}x, voxels saved {:.1f}x over {} grains'.format(np.median(timeRatios), np.median(voxelRatios), len(timeRatios)))
        lines.append('')
    lines += ['## Error at fixed voxel budget', '', '| grain | ~voxels | cubic area/impulse/curve | best anisotropic area/impulse/curve |', '|---|---|---|---|']
    for g in grains:
        gr = [r for r in rows if r['grain'] == g]
        for budget in (4e6, 15e6):
            near = lambda aspects: min((r for r in gr if (r['aspect'] in aspects)), key=lambda r: abs(math.log(r['voxels'] / budget)), default=None)
            iso, aniso = near([1]), near([2, 4, 8])
            if iso and aniso:
                f = lambda r: '{:.1f}M {:.2f}/{:.2f}/{:.2f}% (a{} xy{})'.format(r['voxels'] / 1e6, r['areaRms'], r['impulse'], r['curve'], r['aspect'], r['nxy'])
                lines.append('| {} | {:.0f}M | {} | {} |'.format(g, budget / 1e6, f(iso), f(aniso)))
    sdf = [r for r in allRows if r.get('init') == 'sdf']
    if sdf:
        lines += ['', '## Subvoxel initial surface (sdf) against a binary mask, same map', '',
                  '| grain | map | voxels | binary: area / web / impulse / curve | sdf: area / web / impulse / curve |', '|---|---|---|---|---|']
        for g in grains:
            for r in sorted((x for x in sdf if x['grain'] == g), key=lambda x: (x['aspect'], x['nxy'])):
                b = next((x for x in rows if x['grain'] == g and x['aspect'] == r['aspect'] and x['nxy'] == r['nxy']), None)
                if b:
                    f = lambda x: '{:.2f}% / {:+.2f}% / {:.2f}% / {:.2f}%'.format(x['areaRms'], x['webErr'], x['impulse'], x['curve'])
                    lines.append('| {} | a{} xy{} | {:.1f}M | {} | {} |'.format(g, r['aspect'], r['nxy'], r['voxels'] / 1e6, f(b), f(r)))
        lines += ['', 'Median over grains of the error at each cubic map dimension (binary -> sdf):', '']
        for nxy in (64, 96, 128, 192):
            pairs = [(next((x for x in rows if x['grain'] == r['grain'] and x['aspect'] == 1 and x['nxy'] == nxy), None), r)
                     for r in sdf if r['aspect'] == 1 and r['nxy'] == nxy]
            if pairs:
                med = lambda key, i: np.median([abs(p[i][key]) for p in pairs if p[0]])
                lines.append('- map {}: area {:.2f}% -> {:.2f}%, web {:.2f}% -> {:.2f}%, impulse {:.2f}% -> {:.2f}%, curve {:.2f}% -> {:.2f}%'.format(
                    nxy, med('areaRms', 0), med('areaRms', 1), med('webErr', 0), med('webErr', 1), med('impulse', 0), med('impulse', 1), med('curve', 0), med('curve', 1)))
    text = '\n'.join(lines)
    (Path(args.out) / 'report.md').write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run_ = sub.add_parser('run')
    run_.add_argument('--out', default=str(REPO_ROOT / '3dfmm_testing' / 'out' / 'aniso'))
    run_.add_argument('--grains', nargs='*', default=DEFAULT_GRAINS)
    run_.add_argument('--max-nxy', type=int, default=512)
    run_.add_argument('--inits', nargs='*', default=['binary', 'sdf'])
    rep = sub.add_parser('report')
    rep.add_argument('--out', default=str(REPO_ROOT / '3dfmm_testing' / 'out' / 'aniso'))
    args = parser.parse_args()
    {'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
