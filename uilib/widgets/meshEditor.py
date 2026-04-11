import math
import itertools
import os

from PyQt6.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFileDialog, QApplication
from PyQt6.QtCore import pyqtSignal

import motorlib
import motorlib.units
import trimesh
import numpy as np

try:
    import cadquery as cq
    _HAS_CADQUERY = True
except ImportError:
    _HAS_CADQUERY = False

class MeshEditor(QWidget):

    meshChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setMaximumWidth(320)
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)

        buttonRow = QHBoxLayout()
        self.selectButton = QPushButton('Select')
        self.selectButton.pressed.connect(self.loadMesh)
        buttonRow.addWidget(self.selectButton)
        buttonRow.addStretch()
        self.layout().addLayout(buttonRow)

        self.infoLabel = QLabel('')
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.infoLabel.setMaximumWidth(260)
        self.infoLabel.hide()
        self.layout().addWidget(self.infoLabel)

        self.faces = []
        self.vertices = []
        self._bounds = None   # raw extents in mesh-file units (dx, dy, dz)
        self._filename = ''
        self.sourcePath = ''

        self.meshUnit = 'm'   # unit the loaded mesh is in; updated via setMeshUnit()
        self.preferences = None

    def loadMesh(self, path=None):
        if path is None:
            path = QFileDialog.getOpenFileName(
                None, 'Load core geometry', '',
                'All Mesh Files (*.stl *.STL *.step *.stp *.STEP *.STP)'
                ';;STL Files (*.stl *.STL)'
                ';;STEP Files (*.step *.stp *.STEP *.STP)'
            )[0]
        if path == '':
            return

        alerts = []
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.step', '.stp'):
            if not _HAS_CADQUERY:
                QApplication.instance().outputMessage(
                    'cadquery is required for STEP import.\nInstall with: pip install cadquery',
                    'STEP import error'
                )
                return
            result = cq.importers.importStep(path)
            all_verts = []
            all_faces = []
            for solid in result.solids().vals():
                verts, faces = solid.tessellate(0.0001)
                offset = len(all_verts)
                all_verts.extend([[v.x, v.y, v.z] for v in verts])
                all_faces.extend([[t[0] + offset, t[1] + offset, t[2] + offset] for t in faces])
            self.vertices = np.array(all_verts, dtype=float)
            self.faces = np.array(all_faces, dtype=int)

            check = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
            if not check.is_watertight:
                alerts.append('Mesh must be closed, export the entire core geometry')
        else:
            mesh = trimesh.load(path, force='mesh')

            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if len(meshes) == 0:
                    QApplication.instance().outputMessage(
                        'Selected file does not contain any mesh geometry.',
                        'Mesh import error'
                    )
                    return
                mesh = trimesh.util.concatenate(meshes)

            if isinstance(mesh, list):
                meshes = [g for g in mesh if isinstance(g, trimesh.Trimesh)]
                if len(meshes) == 0:
                    QApplication.instance().outputMessage(
                        'Selected file does not contain any mesh geometry.',
                        'Mesh import error'
                    )
                    return
                mesh = trimesh.util.concatenate(meshes)

            if not isinstance(mesh, trimesh.Trimesh):
                QApplication.instance().outputMessage(
                    'Unable to parse the selected mesh file.',
                    'Mesh import error'
                )
                return

            if not mesh.is_watertight:
                alerts.append('Mesh must be closed, export the entire core geometry')
            self.faces = mesh.faces
            self.vertices = mesh.vertices

        verts = np.asarray(self.vertices)
        if len(verts) > 0:
            self._bounds = verts.max(axis=0) - verts.min(axis=0)
        else:
            self._bounds = None
        self.sourcePath = path
        self._filename = os.path.basename(path)
        self._updateLabel()

        self.meshChanged.emit()

        if len(alerts) > 0:
            QApplication.instance().outputMessage('\n'.join(alerts), 'Mesh import warnings')

    def setMeshUnit(self, unit):
        """Called externally when the user changes the mesh-file unit dropdown."""
        self.meshUnit = unit
        self._updateLabel()

    def setMeshData(self, faces, vertices, sourcePath=''):
        """Set mesh data loaded from properties so status can be shown without re-import."""
        self.faces = faces
        self.vertices = vertices
        self.sourcePath = sourcePath
        self._filename = os.path.basename(sourcePath) if sourcePath else ''

        verts = np.asarray(self.vertices)
        if len(verts) > 0:
            self._bounds = verts.max(axis=0) - verts.min(axis=0)
            if self._filename == '':
                self._filename = '<saved mesh>'
        else:
            self._bounds = None

        self._updateLabel()

    def _updateLabel(self):
        if self._bounds is None or self._filename == '':
            self.infoLabel.setText('')
            self.infoLabel.hide()
            return

        # Convert raw bounds (in mesh-file unit) → metres → display unit
        dispUnit = 'm'
        if self.preferences is not None:
            dispUnit = self.preferences.getUnit('m')

        scale = motorlib.units.getConversion(self.meshUnit, 'm')
        bounds_m = self._bounds * scale
        bounds_disp = [motorlib.units.convert(v, 'm', dispUnit) for v in bounds_m]

        dims = ' x '.join('{:.4g}'.format(v) for v in bounds_disp)
        filename = self._filename
        if len(filename) > 32:
            filename = filename[:29] + '...'

        self.infoLabel.setText('{} loaded:\n{} {}'.format(filename, dims, dispUnit))
        self.infoLabel.show()

    # Keep old name as an alias for any external callers
    loadSTL = loadMesh
