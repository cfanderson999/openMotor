import math
import itertools

from PyQt6.QtWidgets import QWidget, QPushButton, QHBoxLayout, QFileDialog, QApplication
from PyQt6.QtCore import pyqtSignal

import motorlib
import trimesh
import numpy as np

class MeshEditor(QWidget):

    meshChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())

        self.selectButton = QPushButton('Select')
        self.selectButton.pressed.connect(self.loadSTL)
        self.layout().addWidget(self.selectButton)

        self.faces = []
        self.vertices = []

        self.preferences = None

    def loadSTL(self, path=None):
        if path is None:
            path = QFileDialog.getOpenFileName(None, 'Load core geometry', '', 'STL Files (*.stl *.STL)')[0]
        if path != '': # If they cancel the dialog, path will be an empty string
            mesh = trimesh.load(path)

            alerts = []

            if not mesh.is_watertight:
                alerts.append("Mesh must be closed, export the entire core geometry")

            self.faces = mesh.faces
            self.vertices = mesh.vertices
    
            self.meshChanged.emit()

            if len(alerts) > 0:
                QApplication.instance().outputMessage('\n'.join(alerts), "STL import warnings")
