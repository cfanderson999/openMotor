"""
CoreColorBar
============
The key to the colours that shade the core in the 3-D viewer: a colour bar from the lowest to the highest value of the
metric over the burn, with ticks at round values along it.
"""

import numpy as np
from matplotlib.ticker import MaxNLocator

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QImage, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRectF

from motorlib import units

BAR_HEIGHT = 14
TICK_LENGTH = 4
LABEL_HEIGHT = 14
LABEL_WIDTH = 64
END_CLEARANCE = 80
HINT = 'Choose a core metric to shade the bore'


class CoreColorBar(QWidget):
    """Colour bar with ticks. `setMetric` gives the colours and range, `clear` takes them away."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(BAR_HEIGHT + TICK_LENGTH + LABEL_HEIGHT + 2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._preferences = None
        self._unit = ''
        self.displayUnit = ''
        self._low = self._high = 0.0
        self._image = None
        self._buffer = None

    def setPreferences(self, preferences) -> None:
        self._preferences = preferences

    def convert(self, value: float) -> float:
        return units.convert(value, self._unit, self.displayUnit) if self._unit != self.displayUnit else value

    def format(self, value: float) -> str:
        """The value in the display unit, as it reads on the ticks and in the readout."""
        return '—' if value is None or not np.isfinite(value) else '{:.4g}'.format(self.convert(value))

    def clear(self) -> None:
        self._image = None
        self.update()

    def setMetric(self, unit: str, low: float, high: float, lut: np.ndarray) -> None:
        """Show `lut` (256 RGBA rows) as the colours from `low` to `high`, which are in the internal `unit`."""
        self._unit = unit
        self.displayUnit = self._preferences.getUnit(unit) if self._preferences is not None else unit
        self._low, self._high = low, high
        self._buffer = np.ascontiguousarray(lut[None, :, :3])
        self._image = QImage(self._buffer.data, self._buffer.shape[1], 1, self._buffer.shape[1] * 3,
                             QImage.Format.Format_RGB888)
        self.update()

    def _ticks(self, width: float) -> list:
        """(fraction along the bar, value) of the two ends of the range and of the round values between them, leaving out
        any that would overlap the label of an end."""
        low, high = self.convert(self._low), self.convert(self._high)
        span = high - low if high > low else 1.0
        nice = MaxNLocator(nbins=max(2, int(width / (1.5 * LABEL_WIDTH))), steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
        room = END_CLEARANCE / width
        between = [((v - low) / span, v) for v in nice if room < (v - low) / span < 1.0 - room]
        return [(0.0, low)] + between + [(1.0, high)]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        text = self.palette().color(self.foregroundRole())
        if self._image is None:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, HINT)
            return
        width = self.width() - 1
        painter.drawImage(QRectF(0, 0, width, BAR_HEIGHT), self._image)
        painter.setPen(QPen(text, 1))
        painter.drawRect(QRectF(0, 0, width, BAR_HEIGHT))
        for fraction, value in self._ticks(width):
            x = fraction * width
            painter.drawLine(int(x), BAR_HEIGHT, int(x), BAR_HEIGHT + TICK_LENGTH)
            align = Qt.AlignmentFlag.AlignLeft if fraction == 0.0 else Qt.AlignmentFlag.AlignRight if fraction == 1.0 \
                else Qt.AlignmentFlag.AlignHCenter
            left = 0.0 if fraction == 0.0 else self.width() - LABEL_WIDTH if fraction == 1.0 else x - LABEL_WIDTH / 2
            painter.drawText(QRectF(left, BAR_HEIGHT + TICK_LENGTH, LABEL_WIDTH, LABEL_HEIGHT), align, '{:.4g}'.format(value))
