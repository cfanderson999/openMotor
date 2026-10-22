"""
Core3DPane
==========
A 3-D view of a stack of grains and their burned core. Drag to turn it, drag with the right or middle button (or Shift) to
pan, scroll to zoom, double-click to reset. The grain, the core or both are drawn, and the part beyond the slice planes of
the 2-D views can be cut away to look inside.

Pictures are made by `mathlib._raycast` on a worker thread, so that the window stays responsive while the burn-time
slider is dragged. Only the newest request is drawn. While things are changing quickly the pictures are made at a
resolution that follows how long the last one took, and once they stop a sharper one, with more rays than pixels, replaces
the last.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from PyQt6.QtWidgets import QFrame, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QToolButton, QSizePolicy
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal

from uilib import coreShading, orbit
from uilib.logger import logger

try:
    from mathlib import _raycast
except ImportError:  # the compiled extension is needed to draw the pictures
    _raycast = None

NO_LIMIT = 1e9
MIN_RAYS, MAX_SIZE = 32, 900  # the fewest rays across either side, and the most across the longer
SLOW, FAST = 0.05, 0.015  # seconds a picture may take before the resolution drops, and below which it may rise again
QUIET = 0.25  # seconds without a request after which the scene counts as having stopped changing
MOVING_SCALE, SHARP_SCALE = 0.75, 1.5  # rays per pixel across, while things change and once they have stopped
STEP_MOVING, STEP_SHARP = 1.0, 0.6  # voxels a ray advances at a time
SMOOTH_ACROSS = 0.7  # voxels, the blur of the picture's volume across the grain
SMOOTH_ALONG = (2.0, 0.012, 12.0)  # voxels along the grain: at least, as a share of its length, at most
DRAG_RADIANS_PER_PIXEL = 0.01
SHOWN = [('Core + Grain', 2), ('Core', 1), ('Grain', 0)]
CUTS = [('Ax', 'Cut away the grain beyond the axial slice'),
        ('Cor', 'Cut away the grain beyond the coronal slice'),
        ('Sag', 'Cut away the grain beyond the sagittal slice')]


def smoothed(regression: np.ndarray) -> np.ndarray:
    """The regression map blurred for drawing. A bore that tapers gently is built from voxels, so its radius jumps by a whole
    voxel every so often along the grain. The blur is stronger along the grain than across it, which rounds those steps
    off without softening the shape of the section."""
    along = float(np.clip(SMOOTH_ALONG[1] * regression.shape[0], SMOOTH_ALONG[0], SMOOTH_ALONG[2]))
    width = int(np.sqrt(4.0 * along * along + 1.0)) | 1  # three running averages of this width have the variance of a Gaussian
    edge = float(np.exp(-0.5 / SMOOTH_ACROSS ** 2) / (1.0 + 2.0 * np.exp(-0.5 / SMOOTH_ACROSS ** 2)))
    return _raycast.smoothVolume(np.ascontiguousarray(regression, dtype=np.float32), width, edge)


class _Blurred:
    """A regression map blurred for drawing, which is done the first time it is asked for, on the thread that draws."""

    def __init__(self, regression: np.ndarray):
        self._regression = regression
        self._volume = None

    def volume(self) -> np.ndarray:
        if self._volume is None:
            self._volume = np.ascontiguousarray(smoothed(self._regression), dtype=np.float32)
        return self._volume


class _Scene(NamedTuple):
    """Everything a picture is made from, taken at the moment it was asked for."""
    blurred: _Blurred
    valid: np.ndarray
    depths: np.ndarray
    colors: 'np.ndarray | None'
    view: np.ndarray
    limits: np.ndarray
    shown: int
    pan: np.ndarray
    pixelSize: float
    rows: int
    cols: int
    step: float
    moving: bool


class _Canvas(QWidget):
    """Shows the picture, and turns the view when dragged."""

    viewChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(96, 96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view = orbit.startingView()
        self.zoom = 1.0
        self.pan = np.zeros(2)  # how far the scene has been dragged across and down the picture, in voxels
        self.voxelsPerPixel = 1.0  # the scale of the picture on screen, set by the pane when it draws
        self.message = ''
        self._image = None
        self._buffer = None
        self._last = None
        self._panning = False

    def setPicture(self, picture: np.ndarray) -> None:
        self._buffer = np.ascontiguousarray(picture)
        h, w = self._buffer.shape[:2]
        self._image = QImage(self._buffer.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.update()

    def clear(self, message: str = '') -> None:
        self._image, self._buffer, self.message = None, None, message
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(51, 51, 51))
        if self._image is None:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(self.rect()), self._image)

    def mousePressEvent(self, event) -> None:
        self._last = event.position()
        panButtons = Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton
        self._panning = bool(event.button() & panButtons) or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

    def mouseMoveEvent(self, event) -> None:
        if self._last is None or not event.buttons():
            return
        delta = event.position() - self._last
        self._last = event.position()
        if self._panning:
            self.pan = self.pan + np.array([delta.x(), delta.y()]) * self.voxelsPerPixel
        else:
            self.view = orbit.turn(self.view, delta.x() * DRAG_RADIANS_PER_PIXEL, delta.y() * DRAG_RADIANS_PER_PIXEL)
        self.viewChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._last = None

    def mouseDoubleClickEvent(self, event) -> None:
        self.resetView()

    def resetView(self) -> None:
        self.view, self.zoom, self.pan = orbit.startingView(), 1.0, np.zeros(2)
        self.viewChanged.emit()

    def wheelEvent(self, event) -> None:
        """Zoom toward the cursor, so that the point under it stays where it is."""
        zoom = float(np.clip(self.zoom * 1.15 ** (event.angleDelta().y() / 120.0), 0.4, 40.0))
        cursor = np.array([event.position().x() - 0.5 * self.width(), event.position().y() - 0.5 * self.height()])
        scale = self.voxelsPerPixel * self.zoom / zoom
        self.pan = self.pan + cursor * (scale - self.voxelsPerPixel)
        self.zoom, self.voxelsPerPixel = zoom, scale
        self.viewChanged.emit()


class Core3DPane(QFrame):
    """The 3-D picture and the controls for what it shows."""

    _drawn = pyqtSignal(object, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._regression = None
        self._blurred = None
        self._valid = None
        self._depths = None
        self._colors = None
        self._cursor = (0, 0, 0)
        self._busy = False
        self._newest = None
        self._scale = 1.0
        self._lastRequest = 0.0
        self._refine = QTimer(self)
        self._refine.setSingleShot(True)
        self._refine.setInterval(int(QUIET * 1000 * 0.9))
        self._refine.timeout.connect(lambda: self._request(sharp=True))
        self._executor = ThreadPoolExecutor(max_workers=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        controls = QHBoxLayout()
        controls.setSpacing(3)
        self._shownCombo = QComboBox()
        for name, _ in SHOWN:
            self._shownCombo.addItem(name)
        self._shownCombo.setToolTip('What the 3-D view draws')
        controls.addWidget(self._shownCombo, stretch=1)
        controls.addWidget(QLabel('Cut:'))
        self._cutButtons = []
        for name, tip in CUTS:
            button = QToolButton()
            button.setText(name)
            button.setToolTip(tip)
            button.setCheckable(True)
            controls.addWidget(button)
            self._cutButtons.append(button)
        self._cutButtons[1].setChecked(True)
        layout.addLayout(controls)
        self._canvas = _Canvas()
        layout.addWidget(self._canvas, stretch=1)

        self._shownCombo.currentIndexChanged.connect(self._request)
        for button in self._cutButtons:
            button.toggled.connect(self._request)
        self._canvas.viewChanged.connect(self._request)
        self._drawn.connect(self._onDrawn)
        self.clear()

    # ------------------------------------------------------------------
    # What to draw
    # ------------------------------------------------------------------

    def clear(self, message: str = '') -> None:
        self._regression = self._blurred = self._valid = self._depths = self._colors = None
        self._canvas.clear(message or ('The 3-D view needs the compiled extensions (python setup.py build_ext --inplace)' if _raycast is None else ''))

    def setVolume(self, regression: np.ndarray, mask: np.ndarray) -> None:
        """The stack to draw: its regression maps (slice from the aft end, y, x) and the mask of what is outside the casting."""
        self._regression = np.ascontiguousarray(regression, dtype=np.float32)
        self._blurred = _Blurred(self._regression)
        self._valid = ndimage.gaussian_filter((~np.asarray(mask, dtype=bool)).mean(axis=0, dtype=np.float32),
                                              sigma=SMOOTH_ACROSS).astype(np.float32)
        self._depths = np.zeros(self._regression.shape[0])
        self._colors = None
        self._canvas.message = 'Preparing the 3-D view…'
        self._request()

    def setCore(self, depths: np.ndarray, colors: 'np.ndarray | None') -> None:
        """How deep the core is on each slice, in the units of the maps, and the colour of the core on each slice."""
        self._depths = np.asarray(depths, dtype=np.float64)
        self._colors = None if colors is None else np.asarray(colors, dtype=np.uint8)
        self._request()

    def setCursor(self, z: int, y: int, x: int) -> None:
        """The slice of each 2-D view, where the cutting planes lie."""
        cursor = (int(z), int(y), int(x))
        changed = cursor != self._cursor
        self._cursor = cursor
        if changed and any(button.isChecked() for button in self._cutButtons):
            self._request()

    # ------------------------------------------------------------------
    # Making pictures
    # ------------------------------------------------------------------

    def _limits(self) -> np.ndarray:
        return np.array([cut + 1.0 if button.isChecked() else NO_LIMIT
                         for button, cut in zip(self._cutButtons, self._cursor)])

    def resetView(self) -> None:
        """Back to the starting angle, zoom and position."""
        self._canvas.resetView()

    def _request(self, *_, sharp: bool = False) -> None:
        """Ask for a picture of the scene as it is now. While one is being made, the newest request waits for it. Requests
        that come in a quick run are drawn coarsely, and a sharp picture follows when the run is over."""
        if self._regression is None or _raycast is None:
            return
        now = time.monotonic()
        moving = not sharp and now - self._lastRequest < QUIET
        self._lastRequest = now
        ratio = self._canvas.devicePixelRatioF()
        width, height = max(1, self._canvas.width()), max(1, self._canvas.height())
        if moving:
            factor, step = MOVING_SCALE * self._scale, STEP_MOVING
        else:
            factor, step = SHARP_SCALE, STEP_SHARP
        factor = min(factor * ratio, MAX_SIZE / max(width, height))
        rows, cols = max(MIN_RAYS, int(height * factor)), max(MIN_RAYS, int(width * factor))
        shown = SHOWN[self._shownCombo.currentIndex()][1]
        view, zoom = self._canvas.view.copy(), self._canvas.zoom
        half = 0.5 * np.array(self._regression.shape, dtype=np.float64)
        reachAcross = 1.05 * np.abs(view[orbit.SCREEN_RIGHT]) @ half + 2.0
        reachUp = 1.05 * np.abs(view[orbit.SCREEN_UP]) @ half + 2.0
        voxelsPerPixel = max(2.0 * reachAcross / width, 2.0 * reachUp / height) / zoom
        self._canvas.voxelsPerPixel = voxelsPerPixel
        colors = None if self._colors is None else self._colors.copy()
        self._newest = _Scene(self._blurred, self._valid, self._depths.copy(), colors, view, self._limits(), shown,
                              self._canvas.pan.copy(), voxelsPerPixel * width / cols, rows, cols, step, moving)
        self._refine.start()
        if not self._busy:
            self._start()

    def _start(self) -> None:
        scene, self._newest = self._newest, None
        self._busy = True
        self._executor.submit(self._draw, scene)

    def _draw(self, scene: _Scene) -> None:
        """On the worker thread. A picture that cannot be made is reported to the log, and the pane carries on with the next."""
        began = time.perf_counter()
        try:
            view = scene.view
            offset = np.array([-scene.pan[0], scene.pan[1]])
            kind, hitSlice, shade = _raycast.castCore(
                scene.blurred.volume(), scene.valid, scene.depths, view[orbit.SCREEN_RIGHT], view[orbit.SCREEN_UP],
                view[orbit.TOWARD_CAMERA], scene.limits, scene.shown, offset, scene.pixelSize, scene.rows, scene.cols,
                scene.step)
            picture = coreShading.composeScene(kind, hitSlice, shade, scene.colors)
        except Exception:
            logger.error('The 3-D view could not be drawn:\n' + traceback.format_exc())
            picture = None
        self._drawn.emit(picture, time.perf_counter() - began if scene.moving else 0.0)

    def _onDrawn(self, picture: 'np.ndarray | None', seconds: float) -> None:
        """On the interface thread. `seconds` is how long a coarse picture took, or 0 for a sharp one."""
        self._busy = False
        if self._regression is not None and picture is None:
            self._canvas.clear('The 3-D view could not be drawn. The log says why.')
        elif self._regression is not None:
            self._canvas.setPicture(picture)
            if seconds > SLOW:
                self._scale = max(0.35, self._scale * 0.8)
            elif 0.0 < seconds < FAST:
                self._scale = min(1.0, self._scale * 1.15)
        if self._newest is not None:
            self._start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._request()

    def waitUntilDrawn(self, timeout: float = 30.0) -> None:
        """Process events until the newest picture has been drawn (for tests and scripts)."""
        from PyQt6.QtWidgets import QApplication
        end = time.monotonic() + timeout
        while (self._busy or self._newest is not None) and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()
