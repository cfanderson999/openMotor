"""
SliceViewWidget
===============
A QWidget that displays a single 2-D orthographic slice of a 3-D voxel
volume with an interactive crosshair overlay.

Features
--------
* Two render modes: binary (grayscale burned/unburned) and heatmap (viridis
  LUT applied to the raw distance-field values).
* Interactive crosshair drawn via QPainter on top of the image.  Click or
  drag to move the crosshair; the widget emits ``crosshairMoved(int, int)``
  with the new voxel coordinates.
* Mouse-wheel scrolling emits ``sliceScrolled(int)`` (+1 or -1) so the
  container can advance the out-of-plane slice index.
* Aspect-ratio–preserving scaling: the image is centred and letterboxed.
* Dark-mode aware (mirrors the behaviour in GrainImageWidget).
"""

from PyQt6.QtWidgets import QWidget, QApplication, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint

import numpy as np


def _isDark() -> bool:
    app = QApplication.instance()
    return app is not None and app.isDarkMode()  # type: ignore[attr-defined]


class SliceViewWidget(QWidget):
    """Single-axis orthographic slice viewer with interactive crosshair."""

    # Emitted when the user clicks/drags – (voxel_col, voxel_row)
    crosshairMoved = pyqtSignal(int, int)
    # Emitted on mouse-wheel scroll: +1 (forward) or -1 (backward)
    sliceScrolled = pyqtSignal(int)

    _CH_H_COLOR = QColor(255, 100, 100, 180)  # horizontal line: red
    _CH_V_COLOR = QColor(100, 255, 100, 180)  # vertical line:   green
    _CH_WIDTH   = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(False)

        # Source image dimensions
        self._srcW: int = 1
        self._srcH: int = 1
        # Raw RGBA buffer kept alive so QImage doesn't dangle
        self._imgBuf: 'np.ndarray | None' = None
        # Scaled pixmap ready for drawPixmap
        self._pixmap: 'QPixmap | None' = None
        # Crosshair position in voxel space
        self._chX: int = 0
        self._chY: int = 0
        self._dragging: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setSlice(self, arr: np.ndarray, mode: str, lut: np.ndarray,
                 mapDist: float, maxDist: float,
                 mask: 'np.ndarray | None' = None) -> None:
        """Render a new 2-D slice.

        Parameters
        ----------
        arr      : (rows, cols) float32 regression-map slice
        mode     : 'binary' or 'heatmap'
        lut      : (256, 4) uint8 precomputed viridis RGBA LUT
        mapDist  : normalised regression distance at current burn step
        maxDist  : max value in the full regression map (for heatmap scaling)
        mask     : (rows, cols) bool, True = outside grain boundary (masked out)
        """
        if arr is None or arr.size == 0:
            self._pixmap = None
            self._imgBuf = None
            self.update()
            return

        h, w = arr.shape
        self._srcH = h
        self._srcW = w

        if mode == 'binary':
            rgba = _renderBinary(arr, mapDist, mask)
        else:
            rgba = _renderHeatmap(arr, lut, maxDist, mask)

        # Keep buffer alive – QImage borrows the memory
        self._imgBuf = np.ascontiguousarray(rgba)
        qimg = QImage(self._imgBuf.data, w, h, w * 4,
                      QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def setCrosshair(self, ch_x: int, ch_y: int) -> None:
        """Set crosshair position without emitting a signal."""
        self._chX = int(ch_x)
        self._chY = int(ch_y)
        self.update()

    def getCrosshair(self):
        return self._chX, self._chY

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _getImageRect(self) -> QRect:
        """Letterbox rect where the image is drawn, preserving aspect ratio."""
        ww, wh = self.width(), self.height()
        if self._srcW <= 0 or self._srcH <= 0 or ww <= 0 or wh <= 0:
            return QRect(0, 0, ww, wh)
        scale = min(ww / self._srcW, wh / self._srcH)
        dw = int(self._srcW * scale)
        dh = int(self._srcH * scale)
        ox = (ww - dw) // 2
        oy = (wh - dh) // 2
        return QRect(ox, oy, dw, dh)

    def _widgetToVoxel(self, pos: QPoint):
        """Map a widget-space point to (voxel_col, voxel_row), clamped."""
        r = self._getImageRect()
        if r.width() == 0 or r.height() == 0:
            return 0, 0
        fx = (pos.x() - r.left()) / r.width()
        fy = (pos.y() - r.top())  / r.height()
        vx = int(np.clip(fx * self._srcW, 0, self._srcW - 1))
        vy = int(np.clip(fy * self._srcH, 0, self._srcH - 1))
        return vx, vy

    def _voxelToWidget(self, vx: int, vy: int):
        """Map voxel coordinates to widget-space pixel centre."""
        r = self._getImageRect()
        px = r.left() + (vx + 0.5) / self._srcW * r.width()
        py = r.top()  + (vy + 0.5) / self._srcH * r.height()
        return int(px), int(py)

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        bg = QColor(30, 30, 30) if _isDark() else QColor(50, 50, 50)
        painter.fillRect(self.rect(), bg)

        r = self._getImageRect()
        if self._pixmap is not None:
            painter.drawPixmap(r, self._pixmap)

        # Crosshair lines
        cx_w, cy_w = self._voxelToWidget(self._chX, self._chY)

        pen = QPen()
        pen.setWidth(self._CH_WIDTH)

        pen.setColor(self._CH_H_COLOR)
        painter.setPen(pen)
        painter.drawLine(r.left(), cy_w, r.right(), cy_w)

        pen.setColor(self._CH_V_COLOR)
        painter.setPen(pen)
        painter.drawLine(cx_w, r.top(), cx_w, r.bottom())

        # Slice index label (top-left of image area)
        painter.setPen(QColor(220, 220, 50))
        painter.drawText(r.left() + 4, r.top() + 14,
                         f'{self._chX},{self._chY}')

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            vx, vy = self._widgetToVoxel(event.pos())
            self._chX, self._chY = vx, vy
            self.crosshairMoved.emit(vx, vy)
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            vx, vy = self._widgetToVoxel(event.pos())
            self._chX, self._chY = vx, vy
            self.crosshairMoved.emit(vx, vy)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            self.sliceScrolled.emit(1 if delta > 0 else -1)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ------------------------------------------------------------------
# Pure-function rendering helpers (module-level for easy testing)
# ------------------------------------------------------------------

def _renderBinary(arr: np.ndarray, mapDist: float,
                  mask: 'np.ndarray | None' = None) -> np.ndarray:
    """Return (h, w, 4) RGBA: propellant=grey, core/burned=black.

    Masked-out voxels (outside grain boundary) are rendered transparent
    so the dark widget background shows through.
    """
    remaining = arr > mapDist  # True = propellant still present
    if _isDark():
        # propellant → light-grey, core → near-black
        gray = np.where(remaining, np.uint8(160), np.uint8(30))
    else:
        # propellant → mid-grey, core → black
        gray = np.where(remaining, np.uint8(160), np.uint8(0))
    h, w = gray.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255
    if mask is not None:
        rgba[mask] = (0, 0, 0, 0)  # outside boundary → transparent
    return rgba


def _renderHeatmap(arr: np.ndarray, lut: np.ndarray, maxDist: float,
                   mask: 'np.ndarray | None' = None) -> np.ndarray:
    """Return (h, w, 4) RGBA viridis-mapped distance field.

    Voxels at distance <= 0 (burned core) and masked-out voxels (outside
    grain boundary) are rendered transparent so the dark widget background
    shows through.
    """
    if maxDist <= 0:
        maxDist = 1.0
    normed = np.clip(arr * (255.0 / maxDist), 0, 255).astype(np.uint8)
    rgba = lut[normed].copy()   # (h, w, 4)
    rgba[arr <= 0] = (0, 0, 0, 0)
    if mask is not None:
        rgba[mask] = (0, 0, 0, 0)  # outside boundary → transparent
    return rgba
