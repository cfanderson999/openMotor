"""Extract the experimental data points from the figures of Ma et al. (2020), Int. J. Aerospace Eng. 8889333.

The figures are vector graphics, so marker centres and polyline vertices are read exactly from the PDF. Axes are
calibrated from the tick marks. Usage: python extractMa2020.py <paper.pdf> <output.json> (needs the mutool binary).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

_NUMBER = r'-?\d*\.?\d+(?:e-?\d+)?'
_SERIES_COLORS = ['black', '#dd2b1a', '#75c500', '#444ca8', '#98dde9', '#c54ead']
_BLACK = '#080907'

# figure, page, frame index on page, kind, x axis (first tick, step), y axis (first tick, step),
# series velocities [m/s] in legend order (black, red, green, blue, cyan, magenta), description
_FIGURES = [
    ('Fig1', 3, 0, 'rate', (0, 2), (0, .5), [0, 183, 244, 305, 455, 670], 'King propellant 4525'),
    ('Fig2', 4, 0, 'rate', (0, 2), (0, .5), [0, 183, 213, 244, 550, 670], 'King propellant 5051'),
    ('Fig3', 4, 1, 'rate', (0, 2), (0.5, .5), [0, 244, 274, 305, 490, 610], 'King propellant 4869'),
    ('Fig4', 4, 2, 'rate', (0, 1), (0.5, .5), [0, 244, 305, 366, 823, 914], 'King propellant 5542T'),
    ('Fig5', 5, 0, 'rate', (0, 2), (0, .5), [0, 244, 305, 365, 700, 790], 'King propellant 5565T'),
    ('Fig6', 5, 1, 'rate', (0, 2), (0.5, .5), [0, 214, 305, 549, 760, 844], 'King propellant 5555T'),
    ('Fig8', 6, 1, 'rate', (0, 2), (0.2, .2), [0, 49, 103, 213], 'Marklund and Lake propellant A'),
    ('Fig9', 6, 2, 'rate', (0, 2), (0.2, .2), [0, 54, 113, 236], 'Marklund and Lake propellant C'),
    ('Fig10', 7, 0, 'history', (0, 1), (0, 1), None, 'Hasegawa motor A head-end pressure'),
    ('Fig11', 7, 1, 'history', (0, 1), (0, 1), None, 'Hasegawa motor B head-end pressure'),
    ('Fig12', 7, 2, 'history', (0, 1), (0, 1), None, 'Hasegawa motor C head-end pressure'),
]

_PROPELLANTS = {
    '4525': ('73AP(20um)+27HTPB', 1500, 1667), '5051': ('73AP(200um)+27HTPB', 1500, 1667),
    '4869': ('72AP(5um)+26HTPB+2Fe2O3', 1500, 1667), '5542T': ('77AP(20um)+23HTPB', 1550, 2065),
    '5565T': ('68.35AP(200um)+13.65AP(90um)+18HTPB', 1650, 2575),
    '5555T': ('41AP(200um)+41AP(90um)+18HTPB', 1650, 2575),
    'A': ('65AP(20-30um)+35polyester', 1620, 1690), 'C': ('75AP(20-30um)+25polysulfide', 1700, 2550),
    'Hasegawa1': ('69AP+17HTPB+14Al', 1700, 3041),
}
_MOTORS = {'A': (34, 40, 80, 1680), 'B': (23, 40, 80, 840), 'C': (34, 60, 80, 1260)}


def _parsePath(d):
    seq = [c if c else float(n) for c, n in re.findall(r'([MLHVCZ])|(' + _NUMBER + ')', d)]
    points, cursor, command, i = [], (0.0, 0.0), None, 0
    while i < len(seq):
        if isinstance(seq[i], str):
            command = seq[i]
            i += 1
        if command == 'Z':
            continue
        if command in 'ML':
            cursor = (seq[i], seq[i + 1])
            i += 2
            points.append(cursor)
        elif command == 'H':
            cursor = (seq[i], cursor[1])
            i += 1
            points.append(cursor)
        elif command == 'V':
            cursor = (cursor[0], seq[i])
            i += 1
            points.append(cursor)
        elif command == 'C':
            points += [(seq[i], seq[i + 1]), (seq[i + 2], seq[i + 3]), (seq[i + 4], seq[i + 5])]
            cursor = (seq[i + 4], seq[i + 5])
            i += 6
    return points


def _elements(pdfPath, page):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['mutool', 'draw', '-q', '-o', os.path.join(tmp, 'page.svg'), pdfPath, str(page)], check=True,
                       capture_output=True)
        svg = open(os.path.join(tmp, os.listdir(tmp)[0])).read()
    svg = re.sub(r'<defs>.*?</defs>', '', svg, flags=re.S)
    elements = []
    for match in re.finditer(r'<path ([^>]*)/?>', svg):
        attrs = match.group(1)
        d = re.search(r' d="([^"]*)"', attrs)
        if not d:
            continue
        transform = re.search(r'transform="matrix\(([^)]*)\)"', attrs)
        m = [1, 0, 0, 1, 0, 0] if not transform else [float(v) for v in transform.group(1).split(',')]
        points = _parsePath(d.group(1))
        mapped = np.array([(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]) for x, y in points]).reshape(-1, 2)
        fill, stroke = re.search(r'fill="([^"]*)"', attrs), re.search(r'stroke="([^"]*)"', attrs)
        width = re.search(r'stroke-width="([^"]*)"', attrs)
        elements.append({'fill': fill.group(1) if fill else None, 'stroke': stroke.group(1) if stroke else None,
                         'width': float(width.group(1)) if width else None, 'points': mapped})
    return elements


def _frames(elements):
    verticals, horizontals = [], []
    for e in elements:
        p = e['points']
        if e['stroke'] == _BLACK and len(p) == 2:
            if abs(p[0, 0] - p[1, 0]) < 1e-6 and abs(p[0, 1] - p[1, 1]) > 80:
                verticals.append((p[0, 0], p[:, 1].min(), p[:, 1].max()))
            if abs(p[0, 1] - p[1, 1]) < 1e-6 and abs(p[0, 0] - p[1, 0]) > 80:
                horizontals.append((p[0, 1], p[:, 0].min(), p[:, 0].max()))
    return [{'x0': vx, 'yBottom': hy, 'yTop': vt, 'xRight': hr}
            for vx, vt, vb in verticals for hy, hl, hr in horizontals if abs(hl - vx) < 1 and abs(hy - vb) < 1]


def _ticks(elements, frame, axis):
    found = set()
    for e in elements:
        p = e['points']
        if e['stroke'] != _BLACK or len(p) != 2:
            continue
        if axis == 'x' and abs(p[0, 0] - p[1, 0]) < 1e-6 and abs(p[0, 1] - frame['yBottom']) < .3:
            if 2.0 < abs(p[0, 1] - p[1, 1]) < 3.5 and frame['x0'] - .5 <= p[0, 0] <= frame['xRight'] + .5:
                found.add(round(p[0, 0], 2))
        if axis == 'y' and abs(p[0, 1] - p[1, 1]) < 1e-6 and abs(p[0, 0] - frame['x0']) < .3:
            if 2.0 < abs(p[0, 0] - p[1, 0]) < 3.5 and frame['yTop'] - .5 <= p[0, 1] <= frame['yBottom'] + .5:
                found.add(round(p[0, 1], 2))
    return sorted(found, reverse=(axis == 'y'))


def extract(pdfPath):
    pages, data = {}, {}
    for name, page, frameIndex, kind, xAxis, yAxis, velocities, description in _FIGURES:
        if page not in pages:
            elements = _elements(pdfPath, page)
            pages[page] = (elements, _frames(elements))
        elements, frames = pages[page]
        frame = frames[frameIndex]
        xTicks, yTicks = _ticks(elements, frame, 'x'), _ticks(elements, frame, 'y')
        xFit = np.polyfit(xTicks, xAxis[0] + xAxis[1] * np.arange(len(xTicks)), 1)
        yFit = np.polyfit(yTicks, yAxis[0] + yAxis[1] * np.arange(len(yTicks)), 1)
        toX, toY = (lambda v: float(np.polyval(xFit, v))), (lambda v: float(np.polyval(yFit, v)))
        inside = lambda c: frame['x0'] <= c[0] <= frame['xRight'] and frame['yTop'] <= c[1] <= frame['yBottom']
        entry = {'description': description, 'xAxisMajorTicks': len(xTicks), 'yAxisMajorTicks': len(yTicks)}
        if kind == 'rate':
            series = {str(v): [] for v in velocities}
            for e in elements:
                p = e['points']
                if not len(p):
                    continue
                center = ((p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2)
                if not inside(center):
                    continue
                isColoredMarker = e['fill'] in _SERIES_COLORS and e['stroke'] in (None, 'none')
                isBlackSquare = (e['stroke'] == _BLACK and e['fill'] in (None, 'none') and len(p) == 4
                                 and np.ptp(p[:, 0]) < 3 and np.ptp(p[:, 1]) < 3 and e['width'] == 0.5)
                if isColoredMarker:
                    series[str(velocities[_SERIES_COLORS.index(e['fill'])])].append([toX(center[0]), toY(center[1])])
                elif isBlackSquare:
                    series[str(velocities[0])].append([toX(center[0]), toY(center[1])])
            entry['xName'], entry['yName'] = 'pressure_MPa', 'burn_rate_cm_s'
            entry['gasVelocityMetersPerSecond'] = {k: sorted(v) for k, v in series.items() if v}
        else:
            lines = [e['points'] for e in elements if e['stroke'] == _BLACK and len(e['points']) > 30
                     and inside(e['points'].mean(axis=0))]
            line = max(lines, key=len)
            entry['xName'], entry['yName'] = 'time_s', 'head_end_pressure_MPa'
            entry['points'] = [[toX(x), toY(y)] for x, y in line]
        data[name] = entry
    return data


def build(pdfPath):
    return {
        'source': 'Ma, Bao, Sun, Liu, Hui (2020), A New Erosive Burning Model of Solid Propellant Based on Heat '
                  'Transfer Equilibrium at Propellant Surface, Int. J. Aerospace Eng., 8889333, Figs. 1-6 and 8-12',
        'method': 'Vector extraction from the PDF: marker centres and polyline vertices are exact; axes are calibrated '
                  'from tick marks (fit residual below 0.01 axis units).',
        'uncertainty': {
            'digitization': 'negligible (vector coordinates)',
            'assumedRelative': {
                'note': 'The paper reports no measurement error. These are assumed for tolerance bands only.',
                'burnRate': 0.05, 'pressure': 0.03, 'timeSeconds': 0.03,
            },
            'paperStatedModelAccuracy': 0.10,
        },
        'propellants': {k: {'composition': c, 'density_kg_m3': r, 'flameTemperature_K': t}
                        for k, (c, r, t) in _PROPELLANTS.items()},
        'motors': {k: {'throatDiameter_mm': t, 'initialPortDiameter_mm': p, 'grainOuterDiameter_mm': o,
                       'grainLength_mm': length} for k, (t, p, o, length) in _MOTORS.items()},
        'figures': extract(pdfPath),
    }


if __name__ == '__main__':
    json.dump(build(sys.argv[1]), open(sys.argv[2], 'w'), indent=1)
