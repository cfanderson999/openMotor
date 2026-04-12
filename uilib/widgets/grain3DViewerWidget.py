"""
Grain3DViewerWidget
===================
DICOM-style 2×2 orthogonal slice viewer for 3-D FMM grain regression maps.

Layout (2×2 grid)
-----------------
  ┌─────────────┬─────────────┐
  │  Axial      │  Coronal    │
  │  XY plane   │  XZ plane   │
  │  (z slice)  │  (y slice)  │
  ├─────────────┼─────────────┤
  │  Sagittal   │  (future    │
  │  YZ plane   │   3-D view) │
  │  (x slice)  │             │
  └─────────────┴─────────────┘

Crosshair interaction
---------------------
Each view has orthogonal crosshair lines.  Clicking/dragging in a view
updates the out-of-plane slice indices for the other two views:

  Axial   click (px, py)  → coronal  Y-val = py,  sagittal X-val = px
  Coronal click (px, pz)  → axial    X-val = px,  sagittal Z-val = pz   (pz = row)
  Sagittal click (py, pz) → axial    Y-val = py,  coronal  Z-val = pz   (pz = col)

  (In axial view   cols=X, rows=Y;   regressionMap[z, y, x])
  (In coronal view cols=X, rows=Z;   slice = regMap[:, y, :])
  (In sagittal view cols=Y, rows=Z;  slice = regMap[:, :, x])

Scroll-wheel on a view advances the slice shown in *that* view:
  Axial scroll   → change sliceZ
  Coronal scroll → change sliceY
  Sagittal scroll→ change sliceX

Performance notes (targets 256³ at <50 ms/frame)
-------------------------------------------------
* The regressionMap is only copied once per axis into a contiguous buffer
  at load time (coronal and sagittal require a copy; axial is a zero-copy
  view).  Only the three thin 2-D slices are extracted per update.
* A 16 ms QTimer debounces rapid slider moves so we skip intermediate
  frames when the slider is dragged quickly.
* The viridis RGBA LUT is precomputed once at class definition time as a
  256×4 uint8 array; per-frame rendering is a single numpy array index.
* NumPy buffer references are pinned as instance variables so QImage never
  danewes freed memory.
"""

import numpy as np
from matplotlib import colormaps as _mpl_colormaps

from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSizePolicy, QFrame, QSlider
)
from PyQt6.QtCore import Qt, QTimer

from motorlib.grain import Fmm3DGrain

from .sliceViewWidget import SliceViewWidget


# ---------------------------------------------------------------------------
# Precompute the viridis LUT once at import time
# ---------------------------------------------------------------------------

def _buildLut(name: str = 'viridis') -> np.ndarray:
    """Return (256, 4) uint8 RGBA array for the named matplotlib colormap."""
    cmap = _mpl_colormaps[name]
    indices = np.linspace(0.0, 1.0, 256)
    rgba_f = cmap(indices)           # (256, 4) float64, 0–1
    return (rgba_f * 255).astype(np.uint8)


_VIRIDIS_LUT: np.ndarray = _buildLut('viridis')

# ---------------------------------------------------------------------------
# Helper: axis-label titles for the slice views
# ---------------------------------------------------------------------------

_AXIAL_TITLE    = 'Axial (XY) — Z slice'
_CORONAL_TITLE  = 'Coronal (XZ) — Y slice'
_SAGITTAL_TITLE = 'Sagittal (YZ) — X slice'
_PLACEHOLDER_TITLE = '3-D View (coming soon)'


class Grain3DViewerWidget(QWidget):
    """2×2 DICOM-style orthographic slice viewer for 3-D FMM grains."""

    # Debounce interval for slider updates (ms)
    _DEBOUNCE_MS = 16

    def __init__(self, parent=None):
        super().__init__(parent)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._simResult = None
        # List of (grain_index, grain) for Fmm3DGrain instances only
        self._3dGrains: list = []
        # Currently displayed grain
        self._currentGrain: 'Fmm3DGrain | None' = None
        self._currentGrainIdx: int = -1

        # Regression map for the current grain (3-D array, kept as a reference)
        self._regMap: 'np.ndarray | None' = None
        # Contiguous C-order copies for the non-contiguous axes
        self._regMapContiguous: 'np.ndarray | None' = None
        # Max distance in the regression map (for heatmap normalisation)
        self._maxDist: float = 1.0

        # Current slice indices along each axis
        self._sliceZ: int = 0   # axial view shows this Z slice
        self._sliceY: int = 0   # coronal view shows this Y slice
        self._sliceX: int = 0   # sagittal view shows this X slice

        # Current normalised regression distance (from burn-time slider)
        self._mapDist: float = 0.0

        # Grain boundary mask (3-D bool: True = outside cylinder)
        self._mask: 'np.ndarray | None' = None

        # Display mode
        self._mode: str = 'binary'  # 'binary' or 'heatmap'

        # True when showing t=0 geometry before a full simulation has been run
        self._previewMode: bool = False

        # Debounce timer
        self._debounceTimer = QTimer(self)
        self._debounceTimer.setSingleShot(True)
        self._debounceTimer.setInterval(self._DEBOUNCE_MS)
        self._debounceTimer.timeout.connect(self._renderAll)

        # ------------------------------------------------------------------
        # Build UI
        # ------------------------------------------------------------------
        self._buildUi()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _buildUi(self):
        outerLayout = QVBoxLayout(self)
        outerLayout.setContentsMargins(4, 4, 4, 4)
        outerLayout.setSpacing(4)

        # Controls bar
        ctrlBar = QHBoxLayout()
        ctrlBar.setSpacing(8)

        ctrlBar.addWidget(QLabel('Grain:'))
        self._grainCombo = QComboBox()
        self._grainCombo.setMinimumWidth(160)
        self._grainCombo.currentIndexChanged.connect(self._onGrainSelected)
        ctrlBar.addWidget(self._grainCombo)

        ctrlBar.addWidget(QLabel('Mode:'))
        self._modeCombo = QComboBox()
        self._modeCombo.addItems(['Binary', 'Heatmap'])
        self._modeCombo.currentIndexChanged.connect(self._onModeChanged)
        ctrlBar.addWidget(self._modeCombo)

        self._sliceLabel = QLabel('slice: —')
        ctrlBar.addWidget(self._sliceLabel)

        ctrlBar.addStretch()
        outerLayout.addLayout(ctrlBar)

        # 2×2 grid of views
        grid = QGridLayout()
        grid.setSpacing(2)

        self._axialView   = self._makeView(_AXIAL_TITLE)
        self._coronalView = self._makeView(_CORONAL_TITLE)
        self._sagittalView = self._makeView(_SAGITTAL_TITLE)

        grid.addWidget(self._wrapView(self._axialView,    _AXIAL_TITLE),    0, 0)
        grid.addWidget(self._wrapView(self._coronalView,  _CORONAL_TITLE),  0, 1)
        grid.addWidget(self._wrapView(self._sagittalView, _SAGITTAL_TITLE), 1, 0)
        grid.addWidget(self._makePlaceholder(),                              1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        outerLayout.addLayout(grid, stretch=1)

        # Burn-time slider (below the grid)
        sliderBar = QHBoxLayout()
        sliderBar.setSpacing(4)
        sliderBar.addWidget(QLabel('Burn Time:'))
        self._timeSlider = QSlider(Qt.Orientation.Horizontal)
        self._timeSlider.setMinimum(0)
        self._timeSlider.setMaximum(0)
        sliderBar.addWidget(self._timeSlider, stretch=1)
        self._timeLabel = QLabel('— s')
        self._timeLabel.setMinimumWidth(60)
        sliderBar.addWidget(self._timeLabel)
        outerLayout.addLayout(sliderBar)

        # Connect crosshair signals
        self._axialView.crosshairMoved.connect(self._onAxialCrosshair)
        self._coronalView.crosshairMoved.connect(self._onCoronalCrosshair)
        self._sagittalView.crosshairMoved.connect(self._onSagittalCrosshair)

        # Connect scroll-wheel signals
        self._axialView.sliceScrolled.connect(self._onAxialScroll)
        self._coronalView.sliceScrolled.connect(self._onCoronalScroll)
        self._sagittalView.sliceScrolled.connect(self._onSagittalScroll)

        self._showPlaceholderMessage()

    @staticmethod
    def _makeView(title: str) -> SliceViewWidget:
        v = SliceViewWidget()
        v.setMinimumSize(120, 120)
        return v

    @staticmethod
    def _wrapView(view: SliceViewWidget, title: str) -> QFrame:
        """Wrap a SliceViewWidget in a frame with a title label."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(1)
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet('font-size: 9px; color: gray;')
        vbox.addWidget(lbl)
        vbox.addWidget(view, stretch=1)
        return frame

    @staticmethod
    def _makePlaceholder() -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        vbox = QVBoxLayout(frame)
        lbl = QLabel(_PLACEHOLDER_TITLE)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet('color: gray; font-size: 11px;')
        vbox.addWidget(lbl)
        return frame

    def _showPlaceholderMessage(self, msg: str = 'No 3-D grains in this motor'):
        for v in (self._axialView, self._coronalView, self._sagittalView):
            v.setSlice(None, self._mode, _VIRIDIS_LUT, 0.0, 1.0)
        self._sliceLabel.setText(msg)

    # ------------------------------------------------------------------
    # Public API – called by ResultsWidget
    # ------------------------------------------------------------------

    def showPreview(self, motor) -> None:
        """Show the grain at t=0 after quick results are ready, before a full simulation.

        Builds a throw-away one-step SimulationResult (regression=0 for all grains)
        so the existing showData/updateRegression path can be reused unchanged.
        The burn-time slider is disabled and labelled '(preview)' until a full
        simulation result replaces this data via showData().
        """
        from motorlib.simResult import SimulationResult
        preview = SimulationResult(motor)
        preview.channels['time'].addData(0.0)
        preview.channels['regression'].addData([0.0] * len(motor.grains))
        self.showData(preview)          # populates combo, renders at regression=0
        self._previewMode = True        # must come AFTER showData resets it
        self._timeSlider.setEnabled(False)
        self._timeLabel.setText('0.000 s (preview)')

    def showData(self, simResult) -> None:
        """Populate the grain selector and load the first 3-D grain."""
        self._previewMode = False
        self._timeSlider.setEnabled(True)
        self._simResult = simResult
        self._3dGrains = [
            (i, g)
            for i, g in enumerate(simResult.motor.grains)
            if isinstance(g, Fmm3DGrain) and g.regressionMap is not None
        ]

        self._grainCombo.blockSignals(True)
        self._grainCombo.clear()
        for i, g in self._3dGrains:
            self._grainCombo.addItem(f'Grain {i + 1} ({g.geomName})', userData=(i, g))
        self._grainCombo.blockSignals(False)

        # Set up the burn-time slider range
        numSteps = len(simResult.channels['time'].getData())
        self._timeSlider.blockSignals(True)
        self._timeSlider.setMaximum(max(0, numSteps - 1))
        self._timeSlider.setValue(0)
        self._timeSlider.blockSignals(False)
        self._timeLabel.setText('0.000 s')

        if self._3dGrains:
            self._grainCombo.setCurrentIndex(0)
            self._loadGrain(0)
        else:
            self._currentGrain = None
            self._currentGrainIdx = -1
            self._regMap = None
            self._mask = None
            self._showPlaceholderMessage()

    def updateRegression(self, simResult, sliderIndex: int) -> None:
        """Called when the burn-time slider moves.

        Recomputes mapDist for the selected grain and schedules a redraw.
        Also syncs the local slider position.
        """
        if self._currentGrain is None or simResult is None:
            return

        gid = self._currentGrainIdx
        regDist = simResult.channels['regression'].getPoint(sliderIndex)[gid]
        diameter = self._currentGrain.props['diameter'].getValue()
        self._mapDist = regDist / (0.5 * diameter)

        # Sync local slider without re-triggering
        self._timeSlider.blockSignals(True)
        self._timeSlider.setValue(sliderIndex)
        self._timeSlider.blockSignals(False)
        currentTime = simResult.channels['time'].getPoint(sliderIndex)
        suffix = '  (preview)' if self._previewMode else ''
        self._timeLabel.setText(f'{currentTime:.3f} s{suffix}')

        # Debounce: restart the timer so rapid moves only trigger one redraw
        self._debounceTimer.start()

    def resetView(self) -> None:
        """Clear all state (called when results are reset)."""
        self._simResult = None
        self._3dGrains = []
        self._currentGrain = None
        self._currentGrainIdx = -1
        self._regMap = None
        self._regMapContiguous = None
        self._mask = None
        self._mapDist = 0.0
        self._previewMode = False
        self._grainCombo.blockSignals(True)
        self._grainCombo.clear()
        self._grainCombo.blockSignals(False)
        self._timeSlider.blockSignals(True)
        self._timeSlider.setMaximum(0)
        self._timeSlider.setValue(0)
        self._timeSlider.setEnabled(True)
        self._timeSlider.blockSignals(False)
        self._timeLabel.setText('— s')
        self._showPlaceholderMessage('Run a simulation to see results')

    # ------------------------------------------------------------------
    # Grain loading
    # ------------------------------------------------------------------

    def _loadGrain(self, comboIndex: int) -> None:
        if comboIndex < 0 or comboIndex >= len(self._3dGrains):
            return
        gid, grain = self._3dGrains[comboIndex]
        self._currentGrain = grain
        self._currentGrainIdx = gid
        regMap = grain.regressionMap  # (mapLength, mapDim, mapDim) float32
        self._regMap = regMap

        # Pre-build a contiguous C-order copy for the two non-contiguous axes.
        # At 256³ this is 256³ * 4 bytes ≈ 64 MB – acceptable once per grain.
        # We use the same array for coronal AND sagittal indexing since both
        # need a full copy anyway.
        self._regMapContiguous = np.ascontiguousarray(regMap, dtype=np.float32)
        self._maxDist = float(np.max(regMap)) if regMap.size > 0 else 1.0

        # Store grain boundary mask (True = outside cylinder)
        self._mask = grain.mask  # (mapLength, mapDim, mapDim) bool

        mapLen, mapDim, _ = regMap.shape

        # Initialise crosshairs to the centre of each axis
        self._sliceZ = mapLen  // 2
        self._sliceY = mapDim  // 2
        self._sliceX = mapDim  // 2

        self._syncCrosshairs()
        self._renderAll()

    # ------------------------------------------------------------------
    # Slice extraction
    # ------------------------------------------------------------------

    def _getAxialSlice(self) -> 'np.ndarray | None':
        """Return regressionMap[sliceZ, :, :] — zero-copy view."""
        if self._regMap is None:
            return None
        z = np.clip(self._sliceZ, 0, self._regMap.shape[0] - 1)
        return self._regMap[z, :, :]

    def _getCoronalSlice(self) -> 'np.ndarray | None':
        """Return regressionMap[::-1, sliceY, :] — Z flipped so head end is top row."""
        if self._regMapContiguous is None:
            return None
        y = np.clip(self._sliceY, 0, self._regMapContiguous.shape[1] - 1)
        return np.ascontiguousarray(self._regMapContiguous[::-1, y, :])

    def _getSagittalSlice(self) -> 'np.ndarray | None':
        """Return regressionMap[::-1, :, sliceX] — Z flipped so head end is top row."""
        if self._regMapContiguous is None:
            return None
        x = np.clip(self._sliceX, 0, self._regMapContiguous.shape[2] - 1)
        return np.ascontiguousarray(self._regMapContiguous[::-1, :, x])

    def _getAxialMask(self) -> 'np.ndarray | None':
        """Return mask[sliceZ, :, :] — 2D bool."""
        if self._mask is None:
            return None
        z = np.clip(self._sliceZ, 0, self._mask.shape[0] - 1)
        return np.asarray(self._mask[z, :, :])

    def _getCoronalMask(self) -> 'np.ndarray | None':
        """Return mask[::-1, sliceY, :] — Z flipped to match coronal slice orientation."""
        if self._mask is None:
            return None
        y = np.clip(self._sliceY, 0, self._mask.shape[1] - 1)
        return np.asarray(self._mask[::-1, y, :])

    def _getSagittalMask(self) -> 'np.ndarray | None':
        """Return mask[::-1, :, sliceX] — Z flipped to match sagittal slice orientation."""
        if self._mask is None:
            return None
        x = np.clip(self._sliceX, 0, self._mask.shape[2] - 1)
        return np.asarray(self._mask[::-1, :, x])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _renderAll(self) -> None:
        """Push new images to all three slice views."""
        if self._regMap is None:
            return

        md = self._mapDist
        mx = self._maxDist
        mode = self._mode
        lut = _VIRIDIS_LUT

        axSlice  = self._getAxialSlice()
        corSlice = self._getCoronalSlice()
        sagSlice = self._getSagittalSlice()
        axMask   = self._getAxialMask()
        corMask  = self._getCoronalMask()
        sagMask  = self._getSagittalMask()

        if axSlice is not None:
            self._axialView.setSlice(axSlice, mode, lut, md, mx, axMask)
        if corSlice is not None:
            self._coronalView.setSlice(corSlice, mode, lut, md, mx, corMask)
        if sagSlice is not None:
            self._sagittalView.setSlice(sagSlice, mode, lut, md, mx, sagMask)

        self._updateSliceLabel()

    def _updateSliceLabel(self) -> None:
        if self._regMap is None:
            self._sliceLabel.setText('slice: —')
            return
        sh = self._regMap.shape
        self._sliceLabel.setText(
            f'Z={self._sliceZ}/{sh[0]-1}  Y={self._sliceY}/{sh[1]-1}  X={self._sliceX}/{sh[2]-1}'
        )

    # ------------------------------------------------------------------
    # Crosshair sync helpers
    # ------------------------------------------------------------------

    def _toDisplayZ(self, realZ: int) -> int:
        """Convert real Z (0=aft) <-> display row (0=fore/head).  Self-inverse."""
        if self._regMap is None:
            return realZ
        return (self._regMap.shape[0] - 1) - realZ

    def _syncCrosshairs(self) -> None:
        """Update crosshair positions in all three views without triggering signals."""
        dz = self._toDisplayZ(self._sliceZ)
        # Axial (XY):   horizontal = Y axis (rows), vertical = X axis (cols)
        self._axialView.setCrosshair(self._sliceX, self._sliceY)
        # Coronal (XZ): rows = display_Z (flipped, 0=fore), cols = X
        self._coronalView.setCrosshair(self._sliceX, dz)
        # Sagittal (YZ): rows = display_Z (flipped, 0=fore), cols = Y
        self._sagittalView.setCrosshair(self._sliceY, dz)

    # ------------------------------------------------------------------
    # Crosshair move handlers
    # ------------------------------------------------------------------

    def _onAxialCrosshair(self, col: int, row: int) -> None:
        """Axial view clicked: col=X, row=Y."""
        self._sliceX = col
        self._sliceY = row
        dz = self._toDisplayZ(self._sliceZ)
        self._coronalView.setCrosshair(self._sliceX, dz)
        self._corSlice = None  # invalidate cache entry
        self._sagittalView.setCrosshair(self._sliceY, dz)
        self._sagSlice = None
        # Re-render coronal and sagittal (axial image unchanged)
        self._renderCoronal()
        self._renderSagittal()

    def _onCoronalCrosshair(self, col: int, row: int) -> None:
        """Coronal view clicked: col=X, row=display_Z (0=fore); convert to real Z."""
        self._sliceX = col
        self._sliceZ = self._toDisplayZ(row)  # symmetric: displayZ -> realZ
        self._axialView.setCrosshair(self._sliceX, self._sliceY)
        self._sagittalView.setCrosshair(self._sliceY, row)  # row is already display_Z
        self._renderAxial()
        self._renderSagittal()

    def _onSagittalCrosshair(self, col: int, row: int) -> None:
        """Sagittal view clicked: col=Y, row=display_Z (0=fore); convert to real Z."""
        self._sliceY = col
        self._sliceZ = self._toDisplayZ(row)  # symmetric: displayZ -> realZ
        self._axialView.setCrosshair(self._sliceX, self._sliceY)
        self._coronalView.setCrosshair(self._sliceX, row)  # row is already display_Z
        self._renderAxial()
        self._renderCoronal()

    # ------------------------------------------------------------------
    # Scroll-wheel handlers
    # ------------------------------------------------------------------

    def _clampSlices(self) -> None:
        if self._regMap is None:
            return
        sh = self._regMap.shape   # (mapLen, mapDim, mapDim)
        self._sliceZ = int(np.clip(self._sliceZ, 0, sh[0] - 1))
        self._sliceY = int(np.clip(self._sliceY, 0, sh[1] - 1))
        self._sliceX = int(np.clip(self._sliceX, 0, sh[2] - 1))

    def _onAxialScroll(self, delta: int) -> None:
        """Scroll in axial view → advance Z slice."""
        self._sliceZ += delta
        self._clampSlices()
        dz = self._toDisplayZ(self._sliceZ)
        self._coronalView.setCrosshair(self._sliceX, dz)
        self._sagittalView.setCrosshair(self._sliceY, dz)
        self._renderAxial()
        self._updateSliceLabel()

    def _onCoronalScroll(self, delta: int) -> None:
        """Scroll in coronal view → advance Y slice."""
        self._sliceY += delta
        self._clampSlices()
        dz = self._toDisplayZ(self._sliceZ)
        self._axialView.setCrosshair(self._sliceX, self._sliceY)
        self._sagittalView.setCrosshair(self._sliceY, dz)
        self._renderCoronal()
        self._updateSliceLabel()

    def _onSagittalScroll(self, delta: int) -> None:
        """Scroll in sagittal view → advance X slice."""
        self._sliceX += delta
        self._clampSlices()
        dz = self._toDisplayZ(self._sliceZ)
        self._axialView.setCrosshair(self._sliceX, self._sliceY)
        self._coronalView.setCrosshair(self._sliceX, dz)
        self._renderSagittal()
        self._updateSliceLabel()

    # ------------------------------------------------------------------
    # Per-view render helpers (only re-render what changed)
    # ------------------------------------------------------------------

    def _renderAxial(self) -> None:
        sl = self._getAxialSlice()
        if sl is not None:
            self._axialView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                     self._mapDist, self._maxDist,
                                     self._getAxialMask())

    def _renderCoronal(self) -> None:
        sl = self._getCoronalSlice()
        if sl is not None:
            self._coronalView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                       self._mapDist, self._maxDist,
                                       self._getCoronalMask())

    def _renderSagittal(self) -> None:
        sl = self._getSagittalSlice()
        if sl is not None:
            self._sagittalView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                        self._mapDist, self._maxDist,
                                        self._getSagittalMask())

    # ------------------------------------------------------------------
    # Control signal handlers
    # ------------------------------------------------------------------

    def _onGrainSelected(self, index: int) -> None:
        self._loadGrain(index)

    def _onModeChanged(self, index: int) -> None:
        self._mode = 'heatmap' if index == 1 else 'binary'
        self._renderAll()
