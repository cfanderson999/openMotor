"""
Grain3DViewerWidget
===================
DICOM-style orthogonal slice viewer for 3-D FMM grain regression maps, or for the whole motor as one stack.

Layout
------
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Grain: [..] Mode: [..] Core: [..]  slice                 value at cross │
  ├─────────────┬─────────────────────────────────────────────────────────┤
  │  Axial      │  Coronal  (XZ plane, head end left, nozzle end right)    │
  │  XY plane   ├─────────────────────────────────────────────────────────┤
  │  (square)   │  Sagittal (YZ plane, head end left, nozzle end right)    │
  ├─────────────┴─────────────────────────────────────────────────────────┤
  │ unit   colour bar with ticks (when the core is shaded)                 │
  │ Burn Time: ────────────────── slider ──────────────────────────  time │
  └───────────────────────────────────────────────────────────────────────┘

Motors are long and round, so the side views are turned to put the long axis across the window and the axial view is a
square beside them. The dividers between the four views can be dragged, and "Reset Views" puts everything back.

The Core menu shades the burned core by a quantity from the simulation (pressure, mass flux, Mach number, ...), one colour
per slice of the grain, on a colour range fixed over the whole burn. The bar under the views gives the colours, and the
top right gives the value at the slice the crosshair is on.

The Grain menu lists each 3-D grain and "Full Motor", which joins all of them end to end as one stack.

Crosshair interaction
---------------------
Each view has orthogonal crosshair lines.  Clicking/dragging in a view
updates the out-of-plane slice indices for the other two views:

  Axial   click (px, py)  → coronal  Y-val = py,  sagittal X-val = px
  Coronal click (pz, px)  → axial    X-val = px,  sagittal Z-val = pz   (pz = column)
  Sagittal click (pz, py) → axial    Y-val = py,  coronal  Z-val = pz   (pz = column)

  (In axial view   cols=X, rows=Y;   regressionMap[z, y, x])
  (In coronal view cols=Z, rows=X;   slice = regMap[:, y, :].T)
  (In sagittal view cols=Z, rows=Y;  slice = regMap[:, :, x].T)

Scroll-wheel on a view advances the slice shown in *that* view:
  Axial scroll   → change sliceZ
  Coronal scroll → change sliceY
  Sagittal scroll→ change sliceX

Performance notes
-----------------
* The regression map is held once as a contiguous stack, and only the three thin 2-D slices are extracted per update.
* A 16 ms QTimer throttles rapid slider moves, so the newest state is drawn while the slider moves rather than only when
  it stops.
* The per-slice depth and colour of the core are worked out once per step and metric, and shared by all four views.
* The 3-D pane draws on a worker thread and only the newest request is drawn.
* The viridis table is built once at import, so drawing a slice is a single numpy array index.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QFrame, QSlider, QSplitter, QPushButton
)
from PyQt6.QtCore import Qt, QTimer

from motorlib.grain import Fmm3DGrain

from uilib import coreShading
from uilib.grainStack import GrainStack

from .sliceViewWidget import SliceViewWidget
from .coreColorBar import CoreColorBar
from .grain3DPane import Core3DPane


_VIRIDIS_LUT: np.ndarray = coreShading.buildLut('viridis')


_AXIAL_TITLE    = 'Axial (XY) — Z slice'
_CORONAL_TITLE  = 'Coronal (XZ) — Y slice'
_SAGITTAL_TITLE = 'Sagittal (YZ) — X slice'
_FULL_MOTOR = 'Full Motor'
_UNIT_COLUMN_WIDTH = 100


_SPLITTER_STYLE = (
    'QSplitter::handle { background-color: rgba(128, 128, 128, 70); }'
    'QSplitter::handle:hover { background-color: rgba(128, 128, 128, 160); }'
)


def _splitter(orientation, *widgets) -> QSplitter:
    """A splitter of `widgets`, whose dividers can be dragged to resize them."""
    splitter = QSplitter(orientation)
    splitter.setHandleWidth(5)
    splitter.setChildrenCollapsible(False)
    splitter.setStyleSheet(_SPLITTER_STYLE)
    for widget in widgets:
        splitter.addWidget(widget)
        splitter.setStretchFactor(splitter.count() - 1, 1)
    return splitter


class Grain3DViewerWidget(QWidget):
    """Orthographic slice viewer for 3-D FMM grains, singly or as the full motor."""

    # Longest wait before a moving slider is drawn (ms)
    _DEBOUNCE_MS = 16

    def __init__(self, parent=None):
        super().__init__(parent)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._simResult = None
        # List of (grain_index, grain) for Fmm3DGrain instances only
        self._3dGrains: list = []
        # What the views show: the grains in the stack, aft end first, and their joined maps
        self._stack: 'GrainStack | None' = None
        # Regression map of the stack (3-D array: slice from the aft end, y, x)
        self._regMap: 'np.ndarray | None' = None
        # Max distance in the regression map (for heatmap normalisation)
        self._maxDist: float = 1.0

        # Current slice indices along each axis
        self._sliceZ: int = 0   # axial view shows this Z slice
        self._sliceY: int = 0   # coronal view shows this Y slice
        self._sliceX: int = 0   # sagittal view shows this X slice

        # Grain boundary mask (3-D bool: True = outside cylinder)
        self._mask: 'np.ndarray | None' = None

        # Display mode
        self._mode: str = 'binary'  # 'binary' or 'heatmap'

        # Core shading: the metric key shown (None for none), the result's profiles, and the recorded step shown
        self._coreKey: 'str | None' = None
        self._shading: 'coreShading.CoreShading | None' = None
        self._step: int = 0
        self._scene = (None, None)  # what _coreScene was last worked out for, and the result
        self._preferences = None

        # True when showing t=0 geometry before a full simulation has been run
        self._previewMode: bool = False
        self._laidOut: bool = False

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

        ctrlBar.addWidget(QLabel('Core:'))
        self._coreCombo = QComboBox()
        self._coreCombo.setMinimumWidth(140)
        self._coreCombo.currentIndexChanged.connect(self._onCoreMetricChanged)
        ctrlBar.addWidget(self._coreCombo)

        self._sliceLabel = QLabel('slice: —')
        ctrlBar.addWidget(self._sliceLabel)

        ctrlBar.addStretch()
        self._readoutLabel = QLabel('')
        self._readoutLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctrlBar.addWidget(self._readoutLabel)
        self._resetButton = QPushButton('Reset Views')
        self._resetButton.setToolTip('Put the four views back to their starting sizes, positions and angles')
        self._resetButton.clicked.connect(self._resetViews)
        ctrlBar.addWidget(self._resetButton)
        outerLayout.addLayout(ctrlBar)

        # Axial view and 3-D view in the left column, the long coronal and sagittal views in the right, all resizable
        self._axialView = self._makeView()
        self._coronalView = self._makeView()
        self._sagittalView = self._makeView()
        self._pane3D = Core3DPane()

        self._leftSplit = _splitter(Qt.Orientation.Vertical, self._wrapView(self._axialView, _AXIAL_TITLE), self._pane3D)
        self._rightSplit = _splitter(Qt.Orientation.Vertical, self._wrapView(self._coronalView, _CORONAL_TITLE),
                                     self._wrapView(self._sagittalView, _SAGITTAL_TITLE))
        self._viewSplit = _splitter(Qt.Orientation.Horizontal, self._leftSplit, self._rightSplit)
        self._viewSplit.setStretchFactor(0, 0)
        self._leftSplit.setMinimumWidth(160)
        outerLayout.addWidget(self._viewSplit, stretch=1)

        # Colour bar above the burn-time slider, in the same columns so that both span the same width
        bottom = QGridLayout()
        bottom.setHorizontalSpacing(4)
        bottom.setVerticalSpacing(0)
        bottom.setColumnMinimumWidth(0, _UNIT_COLUMN_WIDTH)
        bottom.setColumnMinimumWidth(2, _UNIT_COLUMN_WIDTH)
        self._unitLabel = QLabel('')
        self._unitLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        bottom.addWidget(self._unitLabel, 0, 0)
        self._colorBar = CoreColorBar()
        bottom.addWidget(self._colorBar, 0, 1)
        bottom.addWidget(QLabel('Burn Time:'), 1, 0)
        self._timeSlider = QSlider(Qt.Orientation.Horizontal)
        self._timeSlider.setMinimum(0)
        self._timeSlider.setMaximum(0)
        bottom.addWidget(self._timeSlider, 1, 1)
        self._timeLabel = QLabel('— s')
        self._timeLabel.setMinimumWidth(60)
        bottom.addWidget(self._timeLabel, 1, 2)
        bottom.setColumnStretch(1, 1)
        outerLayout.addLayout(bottom)

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
    def _makeView() -> SliceViewWidget:
        v = SliceViewWidget()
        v.setMinimumSize(120, 80)
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

    def showEvent(self, event):
        super().showEvent(event)
        if not self._laidOut:
            self._laidOut = True
            QTimer.singleShot(0, self._resetLayout)

    def _resetLayout(self) -> None:
        """The starting sizes of the views: the axial and 3-D views are squares in the left column."""
        height = self._viewSplit.height()
        side = max(160, (height - self._leftSplit.handleWidth()) // 2)
        self._leftSplit.setSizes([side, side])
        self._rightSplit.setSizes([height, height])
        self._viewSplit.setSizes([side, max(1, self._viewSplit.width() - side)])

    def _resetViews(self) -> None:
        """Put the four views back to how they start: their sizes, the slices at the middle, and the 3-D view's angle."""
        self._resetLayout()
        self._pane3D.resetView()
        if self._regMap is not None:
            sh = self._regMap.shape
            self._sliceZ, self._sliceY, self._sliceX = sh[0] // 2, sh[1] // 2, sh[2] // 2
            self._syncCrosshairs()
            self._renderAll()

    def _showPlaceholderMessage(self, msg: str = 'No 3-D grains in this motor'):
        for v in (self._axialView, self._coronalView, self._sagittalView):
            v.setSlice(None, self._mode, _VIRIDIS_LUT, 0.0, 1.0)
        self._sliceLabel.setText(msg)
        self._readoutLabel.setText('')
        self._colorBar.clear()
        self._unitLabel.setText('')
        self._pane3D.clear()

    # ------------------------------------------------------------------
    # Public API – called by ResultsWidget
    # ------------------------------------------------------------------

    def setPreferences(self, preferences) -> None:
        """The preferences that choose the units the colour bar and readout show."""
        self._preferences = preferences
        self._colorBar.setPreferences(preferences)

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
        self._step = 0
        self._loadCoreMetrics(simResult)
        self._3dGrains = [
            (i, g)
            for i, g in enumerate(simResult.motor.grains)
            if isinstance(g, Fmm3DGrain) and g.regressionMap is not None
        ]

        self._grainCombo.blockSignals(True)
        self._grainCombo.clear()
        for i, g in self._3dGrains:
            self._grainCombo.addItem(f'Grain {i + 1} ({g.geomName})')
        if len(self._3dGrains) > 1:
            self._grainCombo.addItem(_FULL_MOTOR)
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
            self._stack = None
            self._regMap = None
            self._mask = None
            self._showPlaceholderMessage()

    def updateRegression(self, simResult, sliderIndex: int) -> None:
        """Called when the burn-time slider moves.

        Records the step to show and schedules a redraw.
        Also syncs the local slider position.
        """
        if self._stack is None or simResult is None:
            return

        self._step = sliderIndex

        # Sync local slider without re-triggering
        self._timeSlider.blockSignals(True)
        self._timeSlider.setValue(sliderIndex)
        self._timeSlider.blockSignals(False)
        currentTime = simResult.channels['time'].getPoint(sliderIndex)
        suffix = '  (preview)' if self._previewMode else ''
        self._timeLabel.setText(f'{currentTime:.3f} s{suffix}')

        # Throttle: while the slider moves, draw the newest state every _DEBOUNCE_MS rather than only when it stops
        if not self._debounceTimer.isActive():
            self._debounceTimer.start()

    def resetView(self) -> None:
        """Clear all state (called when results are reset)."""
        self._simResult = None
        self._3dGrains = []
        self._stack = None
        self._regMap = None
        self._mask = None
        self._step = 0
        self._shading = None
        self._loadCoreMetrics(None)
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
        """Show the grain at `comboIndex` of the grain menu, or the full motor, which comes after the grains."""
        if comboIndex < 0 or comboIndex > len(self._3dGrains) or not self._3dGrains:
            return
        if comboIndex == len(self._3dGrains):
            self._stack = GrainStack.fullMotor(self._3dGrains)
        else:
            self._stack = GrainStack.single(*self._3dGrains[comboIndex])
        self._regMap = self._stack.regressionMap  # (slices, mapDim, mapDim) float32, contiguous
        self._maxDist = float(np.max(self._regMap)) if self._regMap.size > 0 else 1.0

        # Store grain boundary mask (True = outside cylinder)
        self._mask = self._stack.mask  # (slices, mapDim, mapDim) bool
        self._pane3D.setVolume(self._regMap, self._mask)

        mapLen, mapDim, _ = self._regMap.shape

        # Initialise crosshairs to the centre of each axis
        self._sliceZ = mapLen  // 2
        self._sliceY = mapDim  // 2
        self._sliceX = mapDim  // 2

        self._syncCrosshairs()
        self._renderAll()

    # ------------------------------------------------------------------
    # Core shading
    # ------------------------------------------------------------------

    def _loadCoreMetrics(self, simResult) -> None:
        """Offer the metrics the result can shade, keeping the chosen one when it is still on offer."""
        self._shading = coreShading.CoreShading(simResult) if simResult is not None else None
        keep = self._coreKey
        self._coreCombo.blockSignals(True)
        self._coreCombo.clear()
        self._coreCombo.addItem('None', userData=None)
        for metric in (self._shading.metrics if self._shading is not None else []):
            self._coreCombo.addItem(metric.name, userData=metric)
        keys = [self._coreCombo.itemData(i) for i in range(self._coreCombo.count())]
        index = next((i for i, m in enumerate(keys) if m is not None and m.key == keep), 0)
        self._coreCombo.setCurrentIndex(index)
        self._coreCombo.blockSignals(False)
        self._coreKey = keys[index].key if keys[index] is not None else None

    def _coreScene(self) -> tuple:
        """The depth and the colour of the core on every slice of the stack, for the step and metric on show. They are kept
        until one of those changes, as every view needs them."""
        state = (self._stack, self._shading, self._step, self._coreKey)
        if self._scene[0] != state:
            self._scene = (state, (self._depths(), self._coreColors()))
        return self._scene[1]

    def _depths(self) -> np.ndarray:
        """Normalised regression to draw the core at on every slice of the stack. A result whose burn model reported the
        regression slice by slice is drawn so; otherwise each grain has the one regression of the simulation's log."""
        regressions = self._simResult.channels['regression'].getPoint(self._step)

        def perGrain(gid, grain):
            half = 0.5 * grain.props['diameter'].getValue()
            reported = self._shading.depths(self._step, gid) if self._shading is not None else None
            return reported / half if reported is not None else np.full(grain.regressionMap.shape[0], regressions[gid] / half)

        return self._stack.join(perGrain, 0.0)

    def _coreColors(self) -> 'np.ndarray | None':
        """Per-slice colours of the chosen metric for the stack and step on show, or None to draw the plain core. A grain
        that has no core to shade at this step is grey."""
        if self._coreKey is None or self._shading is None or self._stack is None:
            return None
        colors = self._stack.join(lambda gid, grain: self._shading.colors(self._coreKey, self._step, gid),
                                  coreShading.NO_VALUE)
        reported = any(self._shading.profile(self._step, gid) is not None for gid in self._stack.gids)
        return colors.astype(np.uint8) if reported else None

    def _metricValues(self, key: str) -> np.ndarray:
        """The metric on every slice of the stack at the step on show, NaN where there is none."""
        def perGrain(gid, grain):
            profile = self._shading.profile(self._step, gid)
            return profile.metrics[key] if profile is not None else None

        return self._stack.join(perGrain, np.nan)

    def _updateColorBar(self) -> None:
        metric = self._coreCombo.currentData()
        if metric is None or self._shading is None:
            self._colorBar.clear()
            self._unitLabel.setText('')
            return
        self._colorBar.setMetric(metric.unit, *self._shading.range(metric.key), coreShading.CORE_LUT)
        self._unitLabel.setText(self._colorBar.displayUnit)

    def _updateReadout(self) -> None:
        """The value of the shaded metric at the slice the crosshair is on, in the top bar."""
        metric = self._coreCombo.currentData()
        if metric is None or self._shading is None or self._stack is None:
            self._readoutLabel.setText('')
            return
        z = int(np.clip(self._sliceZ, 0, self._regMap.shape[0] - 1))
        value = self._metricValues(metric.key)[z]
        self._readoutLabel.setText('{} at slice {}: {} {}'.format(
            metric.name, z, self._colorBar.format(value), self._colorBar.displayUnit).rstrip())

    def _axialCoreColors(self, colors) -> 'np.ndarray | None':
        if colors is None or self._regMap is None:
            return None
        z = int(np.clip(self._sliceZ, 0, self._regMap.shape[0] - 1))
        return coreShading.axialImage(colors, z, self._regMap.shape[1:])

    def _sideCoreColors(self, colors) -> 'np.ndarray | None':
        if colors is None or self._regMap is None:
            return None
        return coreShading.sideImage(colors, self._regMap.shape[1])

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
        """Return regressionMap[::-1, sliceY, :].T — Z flipped so the head end is the left column."""
        if self._regMap is None:
            return None
        y = np.clip(self._sliceY, 0, self._regMap.shape[1] - 1)
        return np.ascontiguousarray(self._regMap[::-1, y, :].T)

    def _getSagittalSlice(self) -> 'np.ndarray | None':
        """Return regressionMap[::-1, :, sliceX].T — Z flipped so the head end is the left column."""
        if self._regMap is None:
            return None
        x = np.clip(self._sliceX, 0, self._regMap.shape[2] - 1)
        return np.ascontiguousarray(self._regMap[::-1, :, x].T)

    def _getAxialMask(self) -> 'np.ndarray | None':
        """Return mask[sliceZ, :, :] — 2D bool."""
        if self._mask is None:
            return None
        z = np.clip(self._sliceZ, 0, self._mask.shape[0] - 1)
        return np.asarray(self._mask[z, :, :])

    def _getCoronalMask(self) -> 'np.ndarray | None':
        """Return mask[::-1, sliceY, :].T — Z flipped to match coronal slice orientation."""
        if self._mask is None:
            return None
        y = np.clip(self._sliceY, 0, self._mask.shape[1] - 1)
        return np.asarray(self._mask[::-1, y, :].T)

    def _getSagittalMask(self) -> 'np.ndarray | None':
        """Return mask[::-1, :, sliceX].T — Z flipped to match sagittal slice orientation."""
        if self._mask is None:
            return None
        x = np.clip(self._sliceX, 0, self._mask.shape[2] - 1)
        return np.asarray(self._mask[::-1, :, x].T)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _distances(self) -> tuple:
        """The normalised regression to draw the core at, for the axial plane (one value) and for the side planes (one
        value per column, the slices running left to right from the head end)."""
        depths = self._coreScene()[0]
        z = int(np.clip(self._sliceZ, 0, self._regMap.shape[0] - 1))
        return depths[z], depths[::-1][None, :]

    def _renderAxial(self) -> None:
        sl = self._getAxialSlice()
        if sl is not None:
            self._axialView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                     self._distances()[0], self._maxDist,
                                     self._getAxialMask(), self._axialCoreColors(self._coreScene()[1]))
            self._updateReadout()

    def _renderCoronal(self) -> None:
        sl = self._getCoronalSlice()
        if sl is not None:
            self._coronalView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                       self._distances()[1], self._maxDist,
                                       self._getCoronalMask(), self._sideCoreColors(self._coreScene()[1]))

    def _renderSagittal(self) -> None:
        sl = self._getSagittalSlice()
        if sl is not None:
            self._sagittalView.setSlice(sl, self._mode, _VIRIDIS_LUT,
                                        self._distances()[1], self._maxDist,
                                        self._getSagittalMask(), self._sideCoreColors(self._coreScene()[1]))

    def _renderAll(self) -> None:
        """Push new images to all three slice views."""
        if self._regMap is None:
            return
        self._renderAxial()
        self._renderCoronal()
        self._renderSagittal()
        self._pane3D.setCore(*self._coreScene())
        self._updateColorBar()
        self._updateCursor()

    def _updateCursor(self) -> None:
        """Show the slice of each view in the top bar, and put the 3-D view's cutting planes there."""
        if self._regMap is None:
            self._sliceLabel.setText('slice: —')
            return
        sh = self._regMap.shape
        self._sliceLabel.setText(
            f'Z={self._sliceZ}/{sh[0]-1}  Y={self._sliceY}/{sh[1]-1}  X={self._sliceX}/{sh[2]-1}'
        )
        self._pane3D.setCursor(self._sliceZ, self._sliceY, self._sliceX)

    # ------------------------------------------------------------------
    # Crosshair sync helpers
    # ------------------------------------------------------------------

    def _toDisplayZ(self, realZ: int) -> int:
        """Convert real Z (0=aft) <-> display column (0=fore/head).  Self-inverse."""
        if self._regMap is None:
            return realZ
        return (self._regMap.shape[0] - 1) - realZ

    def _placeAxial(self) -> None:
        # Axial (XY): columns = X, rows = Y
        self._axialView.setCrosshair(self._sliceX, self._sliceY)

    def _placeCoronal(self, displayZ: int) -> None:
        # Coronal (XZ): columns = display_Z (flipped, 0=fore), rows = X
        self._coronalView.setCrosshair(displayZ, self._sliceX)

    def _placeSagittal(self, displayZ: int) -> None:
        # Sagittal (YZ): columns = display_Z (flipped, 0=fore), rows = Y
        self._sagittalView.setCrosshair(displayZ, self._sliceY)

    def _syncCrosshairs(self) -> None:
        """Update crosshair positions in all three views without triggering signals."""
        dz = self._toDisplayZ(self._sliceZ)
        self._placeAxial()
        self._placeCoronal(dz)
        self._placeSagittal(dz)

    # ------------------------------------------------------------------
    # Crosshair move handlers
    # ------------------------------------------------------------------

    def _onAxialCrosshair(self, col: int, row: int) -> None:
        """Axial view clicked: col=X, row=Y."""
        self._sliceX = col
        self._sliceY = row
        dz = self._toDisplayZ(self._sliceZ)
        self._placeCoronal(dz)
        self._placeSagittal(dz)
        # Re-render coronal and sagittal (axial image unchanged)
        self._renderCoronal()
        self._renderSagittal()
        self._updateCursor()

    def _onCoronalCrosshair(self, col: int, row: int) -> None:
        """Coronal view clicked: col=display_Z (0=fore), row=X; convert to real Z."""
        self._sliceX = row
        self._sliceZ = self._toDisplayZ(col)  # symmetric: displayZ -> realZ
        self._placeAxial()
        self._placeSagittal(col)  # col is already display_Z
        self._renderAxial()
        self._renderSagittal()
        self._updateCursor()

    def _onSagittalCrosshair(self, col: int, row: int) -> None:
        """Sagittal view clicked: col=display_Z (0=fore), row=Y; convert to real Z."""
        self._sliceY = row
        self._sliceZ = self._toDisplayZ(col)  # symmetric: displayZ -> realZ
        self._placeAxial()
        self._placeCoronal(col)  # col is already display_Z
        self._renderAxial()
        self._renderCoronal()
        self._updateCursor()

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
        self._placeCoronal(dz)
        self._placeSagittal(dz)
        self._renderAxial()
        self._updateCursor()

    def _onCoronalScroll(self, delta: int) -> None:
        """Scroll in coronal view → advance Y slice."""
        self._sliceY += delta
        self._clampSlices()
        dz = self._toDisplayZ(self._sliceZ)
        self._placeAxial()
        self._placeSagittal(dz)
        self._renderCoronal()
        self._updateCursor()

    def _onSagittalScroll(self, delta: int) -> None:
        """Scroll in sagittal view → advance X slice."""
        self._sliceX += delta
        self._clampSlices()
        dz = self._toDisplayZ(self._sliceZ)
        self._placeAxial()
        self._placeCoronal(dz)
        self._renderSagittal()
        self._updateCursor()

    # ------------------------------------------------------------------
    # Control signal handlers
    # ------------------------------------------------------------------

    def _onGrainSelected(self, index: int) -> None:
        self._loadGrain(index)

    def _onCoreMetricChanged(self, index: int) -> None:
        metric = self._coreCombo.itemData(index)
        self._coreKey = metric.key if metric is not None else None
        self._renderAll()

    def _onModeChanged(self, index: int) -> None:
        self._mode = 'heatmap' if index == 1 else 'binary'
        self._renderAll()
