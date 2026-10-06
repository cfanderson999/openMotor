#!/usr/bin/env python3
"""Plots and an HTML report for the output of performance_quality_sweep.py.

  python 3dfmm_testing/sweep_plots.py --out 3dfmm_testing/out/sweep

Writes PNG files to <out>/plots and a self-contained <out>/report.html.
"""

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_quality_sweep import GRAINS, PROPELLANTS, FLUX_MODES  # noqa: E402

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MAP_DIM_COLORS = {64: '#86b6ef', 96: '#5598e7', 128: '#2a78d6', 192: '#1c5cab', 256: '#104281'}
MODE_COLORS = {'3d-2dflux': '#eb6834', '3d-area': '#2a78d6', '3d-legacy-Exact': '#1baf7a',
               '3d-legacy-Fast': '#eda100', '3d-legacy-Faster': '#e87ba4'}
PROPELLANT_COLORS = dict(zip(PROPELLANTS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
PROPELLANT_N = {name: p['tabs'][0]['n'] for name, p in PROPELLANTS.items()}
ERROR_METRICS = ['burnTime', 'peakPressure', 'avgPressure', 'peakKn', 'impulse', 'peakFlux', 'maxMach']
METRIC_LABELS = {'burnTime': 'burn time', 'peakPressure': 'peak pressure', 'avgPressure': 'average pressure',
                 'peakKn': 'peak Kn', 'impulse': 'total impulse', 'peakFlux': 'peak mass flux', 'maxMach': 'max core Mach'}


def styleAxes(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def plainLogTicks(ax, axis='y'):
    formatter = matplotlib.ticker.FuncFormatter(lambda v, _: '{:g}'.format(v))
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(formatter)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_minor_formatter(matplotlib.ticker.NullFormatter())


def figureLegend(fig, ax, y=1.0, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, y), frameon=False, fontsize=8,
               labelcolor=INK2, **kwargs)


def newFigure(rows, cols, width, height, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, facecolor=SURFACE, **kwargs)
    for ax in axes.ravel():
        styleAxes(ax)
    return fig, axes


def save(fig, directory, name):
    path = directory / name
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return path


def legendFor(ax, **kwargs):
    kwargs.setdefault('fontsize', 7.5)
    return ax.legend(frameon=False, labelcolor=INK2, **kwargs)


class Sweep:
    def __init__(self, out):
        rows = [json.loads(line) for line in (Path(out) / 'results.jsonl').read_text().splitlines()]
        self.rows = [r for r in rows if 'failed' not in r]
        self.index = {}
        for r in self.rows:
            self.index[(r['grain'], r['mode'], r['mapDim'], r['propellant'])] = r
        self.grains = [g for g in GRAINS if any(r['grain'] == g for r in self.rows)]

    def dims(self, grain):
        return sorted({r['mapDim'] for r in self.rows if r['grain'] == grain and r['mode'] != 'baseline'})

    def baseline(self, grain, propellant):
        return self.index.get((grain, 'baseline', 750, propellant))

    def run(self, grain, mode, dim, propellant):
        return self.index.get((grain, mode, dim, propellant))

    def error(self, grain, mode, dim, metric, reference='baseline'):
        """Median over propellants of the relative error of a metric against the baseline or the finest 3D run."""
        values = []
        for p in PROPELLANTS:
            r = self.run(grain, mode, dim, p)
            if reference == 'baseline':
                ref = self.baseline(grain, p)
            else:
                ref = self.run(grain, mode, self.dims(grain)[-1], p)
            if r and ref and ref[metric]:
                values.append(abs(r[metric] / ref[metric] - 1))
        return float(np.median(values)) if values else float('nan')

    def median(self, grain, mode, dim, field):
        values = [self.run(grain, mode, dim, p)[field] for p in PROPELLANTS if self.run(grain, mode, dim, p)]
        return float(np.median(values)) if values else float('nan')


# ---------------------------------------------------------------------------------------------------------------------
# Per grain figures
# ---------------------------------------------------------------------------------------------------------------------

def plotPressureThrust(sweep, grain, directory):
    dims = sweep.dims(grain)
    fig, axes = newFigure(2, len(PROPELLANTS), 3.1 * len(PROPELLANTS), 5.6, sharex='col')
    for col, propellant in enumerate(PROPELLANTS):
        base = sweep.baseline(grain, propellant)
        for row, (channel, scale, unit) in enumerate((('pressure', 1e-6, 'MPa'), ('force', 1.0, 'N'))):
            ax = axes[row][col]
            for dim in dims:
                r = sweep.run(grain, '3d-area', dim, propellant)
                if r:
                    ax.plot(r['series']['time'], np.array(r['series'][channel]) * scale, color=MAP_DIM_COLORS[dim],
                            linewidth=1.1, label='3D {}'.format(dim))
            if base:
                ax.plot(base['series']['time'], np.array(base['series'][channel]) * scale, color=INK, linewidth=1.6,
                        linestyle=(0, (4, 2)), label='baseline')
            ax.set_ylabel('chamber pressure [{}]'.format(unit) if row == 0 else 'thrust [{}]'.format(unit))
            if row == 1:
                ax.set_xlabel('time [s]')
            if row == 0:
                ax.set_title('{} (n {})'.format(propellant, PROPELLANT_N[propellant]), fontsize=9)
    fig.suptitle('{}: chamber pressure and thrust, 3D map dimension overlaid on the normal simulation'.format(grain),
                 color=INK, fontsize=11, y=1.03)
    fig.tight_layout()
    figureLegend(fig, axes[0][0], y=1.0)
    return save(fig, directory, 'pc_thrust_{}.png'.format(grain))


def plotResiduals(sweep, grain, directory):
    dims = sweep.dims(grain)
    fig, axes = newFigure(2, len(PROPELLANTS), 3.1 * len(PROPELLANTS), 5.0, sharex='col')
    for col, propellant in enumerate(PROPELLANTS):
        base = sweep.baseline(grain, propellant)
        if not base:
            continue
        bt, bp, bf = (np.array(base['series'][k]) for k in ('time', 'pressure', 'force'))
        for row, (baseValues, channel, label) in enumerate(((bp, 'pressure', 'pressure error [% of baseline peak]'),
                                                            (bf, 'force', 'thrust error [% of baseline peak]'))):
            ax = axes[row][col]
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.axhspan(-1, 1, color=GRID, alpha=0.5, linewidth=0)
            for dim in dims:
                r = sweep.run(grain, '3d-area', dim, propellant)
                if not r:
                    continue
                t = np.array(r['series']['time'])
                values = np.array(r['series'][channel])
                diff = 100 * (values - np.interp(t, bt, baseValues)) / baseValues.max()
                ax.plot(t, diff, color=MAP_DIM_COLORS[dim], linewidth=1.0, label='3D {}'.format(dim))
            ax.set_ylim(-8, 8)
            ax.set_ylabel(label, fontsize=8)
            if row == 1:
                ax.set_xlabel('time [s]')
            if row == 0:
                ax.set_title(propellant, fontsize=9)
    axes[1][0].text(0.02, 0.04, 'the burnout step can be off scale', transform=axes[1][0].transAxes, fontsize=7, color=INK2)
    fig.suptitle('{}: error against the normal simulation over time (band is +/-1%)'.format(grain), color=INK, fontsize=11, y=1.03)
    fig.tight_layout()
    figureLegend(fig, axes[0][0], y=1.0)
    return save(fig, directory, 'residuals_{}.png'.format(grain))


def curveRms(run, reference):
    """RMS difference of the two pressure curves on normalized burn time, as a percent of the reference peak."""
    a, b = np.array(run['curve']), np.array(reference['curve'])
    return 100 * float(np.sqrt(np.mean(((a - b) / b.max()) ** 2)))


def plotConvergence(sweep, grain, directory):
    dims = sweep.dims(grain)
    metrics = ['burnTime', 'peakPressure', 'avgPressure', 'peakKn', 'impulse', 'curve']
    labels = dict(METRIC_LABELS, curve='pressure curve (RMS, normalized time)')
    fig, axes = newFigure(2, 3, 11.0, 6.4)
    for ax, metric in zip(axes.ravel(), metrics):
        for propellant in PROPELLANTS:
            vs = [sweep.run(grain, '3d-area', d, propellant) for d in dims]
            base = sweep.baseline(grain, propellant)
            if not base or any(v is None for v in vs):
                continue
            if metric == 'curve':
                vsBase = [curveRms(v, base) for v in vs]
                vsFinest = [curveRms(v, vs[-1]) for v in vs]
            else:
                vsBase = [100 * abs(v[metric] / base[metric] - 1) for v in vs]
                vsFinest = [100 * abs(v[metric] / vs[-1][metric] - 1) for v in vs]
            ax.plot(dims, np.maximum(vsBase, 0.01), color=PROPELLANT_COLORS[propellant], marker='o', markersize=3.5, linewidth=1.2,
                    label=propellant)
            ax.plot(dims[:-1], np.maximum(vsFinest[:-1], 0.01), color=PROPELLANT_COLORS[propellant], linestyle=':', linewidth=1.0)
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle='--')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_ylim(0.008, None)
        ax.set_xticks(dims)
        ax.set_xticklabels(dims)
        plainLogTicks(ax)
        ax.set_xlabel('3D map dimension')
        ax.set_ylabel('error [%]')
        ax.set_title(labels[metric], fontsize=9)
    fig.suptitle('{}: error vs 3D map dimension. Solid: against the normal simulation. Dotted: against the finest 3D run. '
                 'Dashed: 1%. Values at 0.01% mean "no difference".'.format(grain), color=INK, fontsize=10, y=1.03)
    fig.tight_layout()
    figureLegend(fig, axes[0][0], y=1.0)
    return save(fig, directory, 'convergence_{}.png'.format(grain))


def plotFluxMach(sweep, grain, directory, propellant='blue-thunder'):
    dims = sweep.dims(grain)
    shown = [dims[0], dims[len(dims) // 2], dims[-1]] if len(dims) >= 3 else dims
    fig, axes = newFigure(2, len(shown), 3.7 * len(shown), 5.4, sharex='col')
    base = sweep.baseline(grain, propellant)
    for col, dim in enumerate(shown):
        for row, (channel, label, scale) in enumerate((('massFlux', 'mass flux [kg/(m^2 s)]', 1.0), ('mach', 'core Mach number', 1.0))):
            ax = axes[row][col]
            for mode in FLUX_MODES:
                r = sweep.run(grain, mode, dim, propellant)
                if not r:
                    continue
                ax.plot(r['series']['time'], np.array(r['series'][channel]) * scale, color=MODE_COLORS[mode], linewidth=1.2 if mode == '3d-area' else 0.9,
                        alpha=1.0 if mode in ('3d-area', '3d-2dflux') else 0.8, label=mode.replace('3d-', ''))
            if base:
                ax.plot(base['series']['time'], base['series'][channel], color=INK, linewidth=1.5, linestyle=(0, (4, 2)), label='baseline')
            ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel('time [s]')
            if row == 0:
                ax.set_title('3D map dimension {}'.format(dim), fontsize=9)
    fig.suptitle('{}: mass flux and Mach number by flux method ({})'.format(grain, propellant), color=INK, fontsize=11, y=1.03)
    fig.tight_layout()
    figureLegend(fig, axes[0][0], y=1.0)
    return save(fig, directory, 'flux_{}.png'.format(grain))


def plotCost(sweep, grain, directory):
    dims = sweep.dims(grain)
    fig, axes = newFigure(1, 4, 14.0, 3.5)
    ax = axes[0][0]
    ax.plot(dims, [sweep.median(grain, '3d-area', d, 'gridSetupSeconds') for d in dims], color=MAP_DIM_COLORS[192], marker='o', markersize=3.5)
    ax.set_title('grid setup', fontsize=9)
    ax.set_ylabel('seconds')
    ax = axes[0][1]
    for mode in FLUX_MODES:
        ax.plot(dims, [sweep.median(grain, mode, d, 'loopSeconds') for d in dims], color=MODE_COLORS[mode], marker='o', markersize=3, linewidth=1.1,
                label=mode.replace('3d-', ''))
    base = [sweep.baseline(grain, p) for p in PROPELLANTS if sweep.baseline(grain, p)]
    if base:
        ax.axhline(np.median([b['loopSeconds'] + b['setupSeconds'] for b in base]), color=INK, linestyle=(0, (4, 2)), linewidth=1.3, label='baseline total')
    ax.set_title('simulation loop (setup excluded)', fontsize=9)
    legendFor(ax, loc='upper left')
    ax = axes[0][2]
    for mode in ('3d-area', '3d-legacy-Exact', '3d-legacy-Faster'):
        ax.plot(dims, [sweep.median(grain, mode, d, 'fluxSeconds') for d in dims], color=MODE_COLORS[mode], marker='o', markersize=3, linewidth=1.1,
                label=mode.replace('3d-', ''))
    ax.set_title('time inside getPeakMassFlux', fontsize=9)
    legendFor(ax, loc='upper left')
    ax = axes[0][3]
    ax.plot(dims, [sweep.median(grain, '3d-area', d, 'peakRssMb') for d in dims], color=MAP_DIM_COLORS[256], marker='o', markersize=3.5)
    ax.set_title('peak memory', fontsize=9)
    ax.set_ylabel('MB')
    for ax in axes.ravel():
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(dims)
        ax.set_xticklabels(dims)
        ax.set_xlabel('3D map dimension')
        plainLogTicks(ax)
    fig.suptitle('{}: cost vs 3D map dimension'.format(grain), color=INK, fontsize=11, y=1.03)
    fig.tight_layout()
    return save(fig, directory, 'cost_{}.png'.format(grain))


# ---------------------------------------------------------------------------------------------------------------------
# Overview figures
# ---------------------------------------------------------------------------------------------------------------------

def heatmap(ax, grid, rowLabels, colLabels, fmt, cmap, title, vmin=None, vmax=None, log=False):
    data = np.array(grid, dtype=float)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax) if log else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(colLabels)))
    ax.set_xticklabels(colLabels, fontsize=8)
    ax.set_yticks(range(len(rowLabels)))
    ax.set_yticklabels(rowLabels, fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK)
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            light = norm(value) > 0.55
            ax.text(j, i, fmt(value), ha='center', va='center', fontsize=7, color='#ffffff' if light else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    return image


def plotErrorHeatmaps(sweep, directory):
    dims = sorted({r['mapDim'] for r in sweep.rows if r['mode'] != 'baseline'})
    metrics = ['peakPressure', 'avgPressure', 'peakKn', 'impulse', 'burnTime']
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.36 * len(sweep.grains) + 1.8), facecolor=SURFACE, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        grid = [[100 * sweep.error(g, '3d-area', d, metric) if d in sweep.dims(g) else np.nan for d in dims] for g in sweep.grains]
        heatmap(ax, grid, sweep.grains if ax is axes[0][0] else [''] * len(sweep.grains), dims, lambda v: '{:.1f}'.format(v), 'Blues',
                '{} error [%]'.format(METRIC_LABELS[metric]), vmin=0.3, vmax=15, log=True)
        ax.set_xlabel('3D map dimension', color=INK2, fontsize=8)
    fig.suptitle('Error against the normal simulation (median over propellants)', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_errors.png')


def plotFluxErrorHeatmaps(sweep, directory):
    dims = sorted({r['mapDim'] for r in sweep.rows if r['mode'] != 'baseline'})
    modes = ['3d-2dflux', '3d-area', '3d-legacy-Exact']
    fig, axes = plt.subplots(2, len(modes), figsize=(4.0 * len(modes), 0.7 * len(sweep.grains) + 3.4), facecolor=SURFACE, squeeze=False)
    for col, mode in enumerate(modes):
        for row, metric in enumerate(('peakFlux', 'maxMach')):
            grid = [[100 * sweep.error(g, mode, d, metric) if d in sweep.dims(g) else np.nan for d in dims] for g in sweep.grains]
            heatmap(axes[row][col], grid, sweep.grains if col == 0 else [''] * len(sweep.grains), dims, lambda v: '{:.0f}'.format(v), 'Oranges',
                    '{}: {} error [%]'.format(mode.replace('3d-', ''), METRIC_LABELS[metric]), vmin=1, vmax=60, log=True)
            axes[row][col].set_xlabel('3D map dimension', color=INK2, fontsize=8)
    fig.suptitle('Mass flux and Mach error against the normal simulation, by flux method', color=INK, fontsize=11, y=1.01)
    fig.tight_layout()
    return save(fig, directory, 'overview_flux.png')


def plotCostHeatmaps(sweep, directory):
    dims = sorted({r['mapDim'] for r in sweep.rows if r['mode'] != 'baseline'})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 0.36 * len(sweep.grains) + 1.8), facecolor=SURFACE, squeeze=False)
    setup = [[sweep.median(g, '3d-area', d, 'gridSetupSeconds') if d in sweep.dims(g) else np.nan for d in dims] for g in sweep.grains]
    rss = [[sweep.median(g, '3d-area', d, 'peakRssMb') if d in sweep.dims(g) else np.nan for d in dims] for g in sweep.grains]
    ratio = []
    for g in sweep.grains:
        base = [sweep.baseline(g, p) for p in PROPELLANTS if sweep.baseline(g, p)]
        baseTotal = np.median([b['setupSeconds'] + b['loopSeconds'] for b in base]) if base else np.nan
        ratio.append([(sweep.median(g, '3d-area', d, 'gridSetupSeconds') + sweep.median(g, '3d-area', d, 'loopSeconds')) / baseTotal
                      if d in sweep.dims(g) else np.nan for d in dims])
    heatmap(axes[0][0], setup, sweep.grains, dims, lambda v: '{:.1f}'.format(v), 'Purples', 'grid setup [s]', vmin=0.1, vmax=150, log=True)
    heatmap(axes[0][1], rss, [''] * len(sweep.grains), dims, lambda v: '{:.0f}'.format(v), 'Greens', 'peak memory [MB]', vmin=100, vmax=3000, log=True)
    heatmap(axes[0][2], ratio, [''] * len(sweep.grains), dims, lambda v: '{:.0f}x'.format(v), 'Reds', '3D time / normal simulation time',
            vmin=1, vmax=3000, log=True)
    for ax in axes[0]:
        ax.set_xlabel('3D map dimension', color=INK2, fontsize=8)
    fig.suptitle('Cost of the 3D simulation', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_cost.png')


def plotScaling(sweep, directory):
    fig, axes = newFigure(1, 2, 11.0, 4.2)
    ax = axes[0][0]
    cmap = plt.get_cmap('tab20')
    points = []
    for index, grain in enumerate(sweep.grains):
        xs = [sweep.median(grain, '3d-area', d, 'voxels') for d in sweep.dims(grain)]
        ys = [sweep.median(grain, '3d-area', d, 'gridSetupSeconds') for d in sweep.dims(grain)]
        ax.plot(xs, ys, color=cmap(index % 20), linewidth=0.9, marker='o', markersize=2.5, label=grain)
        points += list(zip(xs, ys))
    xs, ys = np.array(points).T
    mask = (xs > 3e6) & (ys > 0)
    slope, intercept = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    grid = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    ax.plot(grid, np.exp(intercept) * grid ** slope, color=INK, linestyle='--', linewidth=1.2, label='fit, exponent {:.2f} above 3M voxels'.format(slope))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('voxels in the 3D map')
    ax.set_ylabel('grid setup [s]')
    ax.set_title('setup time follows the voxel count, not the grain', fontsize=9.5)
    legendFor(ax, loc='upper left', ncol=2, fontsize=6.5)
    ax = axes[0][1]
    for index, grain in enumerate(sweep.grains):
        xs = [sweep.median(grain, '3d-area', d, 'voxels') for d in sweep.dims(grain)]
        ys = [sweep.median(grain, '3d-area', d, 'peakRssMb') for d in sweep.dims(grain)]
        ax.plot(xs, ys, color=cmap(index % 20), linewidth=0.9, marker='o', markersize=2.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('voxels in the 3D map')
    ax.set_ylabel('peak memory [MB]')
    ax.set_title('memory grows linearly with voxels', fontsize=9.5)
    fig.tight_layout()
    return save(fig, directory, 'overview_scaling.png'), slope


def plotPareto(sweep, directory):
    fig, axes = newFigure(1, 3, 14.5, 4.6)
    for ax, metric in zip(axes[0], ('impulse', 'peakPressure', 'peakKn')):
        for grain in sweep.grains:
            dims = sweep.dims(grain)
            xs = [sweep.median(grain, '3d-area', d, 'gridSetupSeconds') + sweep.median(grain, '3d-area', d, 'loopSeconds') for d in dims]
            ys = [100 * sweep.error(grain, '3d-area', d, metric) for d in dims]
            ax.plot(xs, ys, color=GRID, linewidth=0.8, zorder=1)
            for x, y, d in zip(xs, ys, dims):
                ax.scatter([x], [y], s=16, color=MAP_DIM_COLORS[d], zorder=2, linewidths=0)
        ax.axhline(1, color=INK2, linestyle='--', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('setup + one simulation [s]')
        ax.set_ylabel('{} error [%]'.format(METRIC_LABELS[metric]))
        ax.set_title(METRIC_LABELS[metric], fontsize=9.5)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label='map {}'.format(d)) for d, c in MAP_DIM_COLORS.items()]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK2, loc='lower left')
    fig.suptitle('Quality per unit cost: every grain and map dimension (3D area flux)', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_pareto.png')


def plotStagnation(sweep, directory):
    metrics = ['peakPressure', 'avgPressure', 'peakKn', 'impulse', 'burnTime']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.38 * len(sweep.grains) + 1.8), facecolor=SURFACE, squeeze=False)
    finest = [[100 * sweep.error(g, '3d-area', sweep.dims(g)[-1], m) for m in metrics] for g in sweep.grains]
    gain = []
    for g in sweep.grains:
        dims = sweep.dims(g)
        row = []
        for m in metrics:
            a, b = sweep.error(g, '3d-area', dims[-2], m), sweep.error(g, '3d-area', dims[-1], m)
            row.append(b / a if a else np.nan)
        gain.append(row)
    heatmap(axes[0][0], finest, sweep.grains, [METRIC_LABELS[m] for m in metrics], lambda v: '{:.1f}'.format(v), 'Blues',
            'error at the finest map dimension [%]', vmin=0.3, vmax=15, log=True)
    heatmap(axes[0][1], gain, [''] * len(sweep.grains), [METRIC_LABELS[m] for m in metrics], lambda v: '{:.2f}'.format(v), 'Oranges',
            'error ratio, finest / previous map dimension\n(near 1 or above means stagnation)', vmin=0.2, vmax=2, log=True)
    fig.suptitle('Quality stagnation', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_stagnation.png')


def plotPropellantSensitivity(sweep, directory):
    fig, axes = newFigure(1, 2, 11.0, 4.0)
    dims = sorted({r['mapDim'] for r in sweep.rows if r['mode'] != 'baseline'})
    for ax, metric in zip(axes[0], ('peakPressure', 'impulse')):
        for d in dims:
            values = []
            for propellant in PROPELLANTS:
                errors = []
                for grain in sweep.grains:
                    r, base = sweep.run(grain, '3d-area', d, propellant), sweep.baseline(grain, propellant)
                    if r and base:
                        errors.append(100 * abs(r[metric] / base[metric] - 1))
                values.append(np.median(errors) if errors else np.nan)
            ax.plot([PROPELLANT_N[p] for p in PROPELLANTS], values, color=MAP_DIM_COLORS[d], marker='o', markersize=4, linewidth=1.2, label='map {}'.format(d))
        ax.set_xlabel('propellant burn rate exponent n')
        ax.set_ylabel('median error over grains [%]')
        ax.set_title(METRIC_LABELS[metric], fontsize=9.5)
    legendFor(axes[0][0], loc='upper left')
    fig.suptitle('Propellants with a higher exponent amplify area errors into pressure errors', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_propellants.png')


def plotQualityModes(sweep, directory):
    modes = ['3d-2dflux', '3d-area', '3d-legacy-Exact', '3d-legacy-Fast', '3d-legacy-Faster']
    fig, axes = newFigure(1, 2, 12.0, 0.34 * len(sweep.grains) + 2.2)
    for ax, metric in zip(axes[0], ('peakFlux', 'maxMach')):
        ypos = np.arange(len(sweep.grains))
        for k, mode in enumerate(modes):
            values = [100 * sweep.error(g, mode, sweep.dims(g)[-1], metric) for g in sweep.grains]
            ax.scatter(values, ypos + (k - 2) * 0.14, s=14, color=MODE_COLORS[mode], label=mode.replace('3d-', ''), linewidths=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sweep.grains if ax is axes[0][0] else [''] * len(sweep.grains), fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlabel('error against the normal simulation [%] at the finest map dimension')
        ax.set_title(METRIC_LABELS[metric], fontsize=9.5)
    legendFor(axes[0][1], loc='lower right')
    fig.suptitle('Mass flux methods compared; the legacy quality modes only matter where their dots separate', color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return save(fig, directory, 'overview_quality_modes.png')


# ---------------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------------

def embed(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def findings(sweep, scalingExponent):
    items = []
    dims = sorted({r['mapDim'] for r in sweep.rows if r['mode'] != 'baseline'})
    flagged = []
    for g in sweep.grains:
        gd = sweep.dims(g)
        for m in ('peakPressure', 'impulse', 'peakKn'):
            errors = [sweep.error(g, '3d-area', d, m) for d in gd]
            if errors[-1] > 0.01 and errors[-1] >= 0.7 * errors[-2]:
                flagged.append('{} {} ({:.1%} at {})'.format(g, METRIC_LABELS[m], errors[-1], gd[-1]))
    items.append(('Stagnation', 'Error stops falling at the finest map dimension for: ' + ('; '.join(flagged) if flagged else 'nothing') + '.'))
    inert = []
    for g in sweep.grains:
        d = sweep.dims(g)[-1]
        values = [sweep.median(g, m, d, 'peakFlux') for m in ('3d-legacy-Exact', '3d-legacy-Fast', '3d-legacy-Faster')]
        if np.nanmax(values) / max(np.nanmin(values), 1e-12) - 1 < 0.005:
            inert.append(g)
    items.append(('Quality modes', 'The legacy Exact, Fast and Faster modes give the same peak flux (within 0.5%) for {} of {} grains. '
                  'The per-layer flux ignores the setting.'.format(len(inert), len(sweep.grains))))
    items.append(('Setup cost', 'Grid setup time scales with the voxel count to the power {:.2f} above 3M voxels.'.format(scalingExponent)))
    return items


def buildReport(sweep, out, figures, scalingExponent):
    parts = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D FMM Sweep Report</title><style>
:root{--bg:#f4f4f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--accent:#2a78d6}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121211;--card:#1c1c1a;--ink:#f2f2ee;--ink2:#b9b8b0;--rule:#33332f;--accent:#6da7ec;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:16px;border-top:1px solid var(--rule)}
p.sub{color:var(--ink2);margin:0 0 20px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:12px 0 8px}nav a{color:var(--accent);text-decoration:none;font-size:14px}
figure{margin:14px 0;background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:8px}
figure img{width:100%;height:auto;display:block;border-radius:3px}figcaption{color:var(--ink2);font-size:13px;margin:6px 4px 2px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:14px 0}dt{color:var(--ink2)}dd{margin:0}
</style></head><body><main>""")
    parts.append('<h1>3D FMM performance and quality sweep</h1>')
    parts.append('<p class="sub">{} grains, {} propellants, {} runs. Error is measured against the normal analytic or 2D FMM simulation.</p>'.format(
        len(sweep.grains), len(PROPELLANTS), len(sweep.rows)))
    parts.append('<dl>' + ''.join('<dt>{}</dt><dd>{}</dd>'.format(k, v) for k, v in findings(sweep, scalingExponent)) + '</dl>')
    parts.append('<nav>' + ''.join('<a href="#{0}">{0}</a>'.format(g) for g in sweep.grains) + '</nav>')
    captions = {
        'overview_errors.png': 'Error of each grain and map dimension. Rows that stay dark to the right are not improving with resolution.',
        'overview_stagnation.png': 'Left: error at the finest map dimension. Right: how much the last refinement helped; values near 1 mean it did not.',
        'overview_flux.png': 'Mass flux and core Mach error for the flux methods. The per-layer method is "area".',
        'overview_quality_modes.png': 'Peak flux and Mach error by method at the finest map dimension. Dots that overlap mean the setting makes no difference.',
        'overview_cost.png': 'Setup time, memory and total time relative to the normal simulation.',
        'overview_scaling.png': 'Setup time and memory against voxel count; all grains fall on one curve.',
        'overview_pareto.png': 'Error against cost. Lines connect map dimensions of one grain; the flat tails are quality stagnation.',
        'overview_propellants.png': 'Pressure scales as Kn to the power 1/(1-n), so a larger exponent magnifies the same area error.',
    }
    for name in ('overview_errors.png', 'overview_stagnation.png', 'overview_pareto.png', 'overview_cost.png', 'overview_scaling.png',
                 'overview_flux.png', 'overview_quality_modes.png', 'overview_propellants.png'):
        parts.append('<figure><img alt="{0}" src="{1}"><figcaption>{2}</figcaption></figure>'.format(name, embed(figures[name]), captions[name]))
    perGrain = [('pc_thrust_{}', 'Chamber pressure and thrust with every 3D map dimension overlaid on the normal simulation.'),
                ('residuals_{}', 'Error over time against the normal simulation, as a percent of its peak.'),
                ('convergence_{}', 'Error against map dimension, one line per propellant.'),
                ('flux_{}', 'Mass flux and Mach number by flux method at three map dimensions.'),
                ('cost_{}', 'Setup, loop, flux and memory cost against map dimension.')]
    for grain in sweep.grains:
        parts.append('<h2 id="{0}">{0}</h2>'.format(grain))
        for key, caption in perGrain:
            file = figures.get(key.format(grain) + '.png')
            if file:
                parts.append('<figure><img alt="{0}" loading="lazy" src="{1}"><figcaption>{2}</figcaption></figure>'.format(key.format(grain), embed(file), caption))
    parts.append('</main></body></html>')
    path = Path(out) / 'report.html'
    path.write_text(''.join(parts))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'out' / 'sweep'))
    parser.add_argument('--grains', nargs='*')
    args = parser.parse_args()
    sweep = Sweep(args.out)
    if args.grains:
        sweep.grains = [g for g in sweep.grains if g in args.grains]
    directory = Path(args.out) / 'plots'
    directory.mkdir(exist_ok=True)
    figures = {}
    for grain in sweep.grains:
        for function in (plotPressureThrust, plotResiduals, plotConvergence, plotFluxMach, plotCost):
            path = function(sweep, grain, directory)
            figures[path.name] = path
        print('plotted', grain, flush=True)
    for function in (plotErrorHeatmaps, plotStagnation, plotPareto, plotCostHeatmaps, plotFluxErrorHeatmaps, plotQualityModes, plotPropellantSensitivity):
        path = function(sweep, directory)
        figures[path.name] = path
    path, exponent = plotScaling(sweep, directory)
    figures[path.name] = path
    figures['overview_flux.png'] = directory / 'overview_flux.png'
    print('report:', buildReport(sweep, args.out, figures, exponent))


if __name__ == '__main__':
    main()
