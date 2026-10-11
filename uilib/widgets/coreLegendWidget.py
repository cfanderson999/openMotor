"""
CoreLegendWidget
================
The key to the colours that shade the core in the 3-D viewer: the metric's name, a colour bar from its lowest to its
highest value over the burn, and its value at the slice the crosshair is on.
"""

import numpy as np

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

from motorlib import units

BAR_HEIGHT = 16


class CoreLegendWidget(QFrame):
    """Colour bar and readout for the metric that shades the core."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._preferences = None
        self._unit = ''
        self._displayUnit = ''

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        self._title = QLabel('')
        self._title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._title)

        self._bar = QLabel()
        self._bar.setFixedHeight(BAR_HEIGHT)
        self._bar.setScaledContents(True)
        layout.addWidget(self._bar)

        ends = QHBoxLayout()
        self._low, self._high = QLabel(''), QLabel('')
        self._high.setAlignment(Qt.AlignmentFlag.AlignRight)
        ends.addWidget(self._low)
        ends.addStretch()
        ends.addWidget(self._high)
        layout.addLayout(ends)

        self._readout = QLabel('')
        layout.addWidget(self._readout)
        layout.addStretch()

        self._hint = QLabel('Choose a core metric to shade the bore')
        self._hint.setStyleSheet('color: gray; font-size: 11px;')
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self.clear()

    def setPreferences(self, preferences) -> None:
        self._preferences = preferences

    def _convert(self, value: float) -> float:
        return units.convert(value, self._unit, self._displayUnit) if self._unit != self._displayUnit else value

    def _format(self, value: float) -> str:
        if value is None or not np.isfinite(value):
            return '—'
        return '{:.4g}'.format(self._convert(value))

    def clear(self) -> None:
        for widget in (self._title, self._bar, self._low, self._high, self._readout):
            widget.hide()
        self._hint.show()

    def setMetric(self, name: str, unit: str, low: float, high: float, lut: np.ndarray) -> None:
        """Show the bar for `name`, with the lookup table `lut` (256 RGBA rows) spanning `low` to `high`."""
        self._unit = unit
        self._displayUnit = self._preferences.getUnit(unit) if self._preferences is not None else unit
        self._title.setText('{}{}'.format(name, ' ({})'.format(self._displayUnit) if self._displayUnit else ''))
        row = np.ascontiguousarray(lut[:, :3])[None, :, :]
        image = np.ascontiguousarray(np.repeat(row, BAR_HEIGHT, axis=0))
        self._barBuffer = image
        self._bar.setPixmap(QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3,
                                                     QImage.Format.Format_RGB888)))
        self._low.setText(self._format(low))
        self._high.setText(self._format(high))
        for widget in (self._title, self._bar, self._low, self._high, self._readout):
            widget.show()
        self._hint.hide()

    def setReadout(self, label: str, value: float) -> None:
        self._readout.setText('{}: {} {}'.format(label, self._format(value), self._displayUnit).rstrip())
