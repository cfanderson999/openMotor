"""3D Custom Grain submodule"""

import matplotlib.pyplot as plt #FIXME

import pyvista as pv
import numpy as np
import hashlib
import pickle

from ..grain import Fmm3DGrain
from ..properties import MeshProperty, EnumProperty, FloatProperty
from ..simResult import SimAlert, SimAlertLevel, SimAlertType
from ..units import getAllConversions, convert

class custom3d(Fmm3DGrain):
    """Custom grains can have any core shape. They define their geometry using a polygon property, which tracks a list
    of polygons that each consist of a number of points. The polygons are scaled according to user specified units and
    drawn onto the core map."""
    geomName = 'Custom 3D Grain'
    def __init__(self):
        super().__init__()
        self.props['mesh'] = MeshProperty('Core geometry')
        self.props['stlUnit'] = EnumProperty('STL Unit', getAllConversions('m'))
        self.faces = []
        self.vertices = []

    def hashCoreMapInputs(self):
        mapDim = self.mapDim
        inUnit = self.props['stlUnit'].getValue()
        faces = self.props['mesh'].getValue()[0]
        vertices = self.props['mesh'].getValue()[1]
        diameter = self.props['diameter'].getValue()
        length = self.props['length'].getValue()

        byte_string = pickle.dumps([mapDim, inUnit, faces, vertices, diameter, length])
        return hashlib.sha256(byte_string).hexdigest()

    def generateCoreMap(self):
        # newCoreMapHash = self.hashCoreMapInputs()
        # if newCoreMapHash == self.coreMapHash:
        #     return 0
        # else:
        #     self.coreMapHash = newCoreMapHash

        inUnit = self.props['stlUnit'].getValue()

        self.faces, self.vertices = self.props['mesh'].getValue()

        mesh = pv.PolyData(self.vertices, np.insert(self.faces, 0, 3, axis=1))

        mesh = mesh.scale(3 * [convert(1, inUnit, 'm')], inplace=False)

        voxelized = pv.voxelize_volume(mesh.extract_surface(), density=self.props['diameter'].getValue()/self.mapDim)

        x,_y,_z = voxelized.meshgrid

        voxelized = voxelized.cell_data_to_point_data()

        coreArray = np.array(voxelized.point_data["InsideMesh"]).reshape(x.shape, order='F') > 0.5
    
        coreArray = np.rot90(coreArray, axes=(1, 0))
        coreArray = np.logical_not(coreArray)

        # The following code adds the endburner, if it exists, on top of the voxelized core and then afterwards 
        # manually pads values onto the core until its dims match the initGeometry coremap dims
        # I really dont like this way of doing this, there is definitely a better solution, also if your
        # voxelized core is an odd width and mapDim is even then it will be slightly off centered

        bounds = np.array(mesh.bounds)
        bounds = bounds[1::2] - bounds[::2]
        self.totalLength = FloatProperty('Length', 'm', 0, 10)
        self.totalLength.setValue(self.props['length'].getValue() + bounds[1])

        self.mapLength = np.ceil(self.lengthToMap(self.totalLength.getValue())).astype(int)
        self.mapZ, self.mapX, self.mapY = np.meshgrid(np.linspace(-1, 1, self.mapLength), np.linspace(-1, 1, self.mapDim), np.linspace(-1, 1, self.mapDim), indexing='ij')
        self.mask = self.mapX**2 + self.mapY**2 > 1

        self.coreMap = np.ones_like(self.mapZ)
        coreBlankShape = np.array(self.coreMap.shape)
        coreNegativeShape = np.array(coreArray.shape)

        print(coreBlankShape, coreNegativeShape, self.mapZ.shape)

        before = (coreBlankShape - coreNegativeShape) // 2
        after = coreBlankShape - before - coreNegativeShape

        before[0] = 0
        after[0] = coreBlankShape[0] - coreNegativeShape[0]

        for i in range(3):
            if coreBlankShape[i] <= coreNegativeShape[i]:
                before[i], after[i] = 0,0

        coreArray = np.pad(coreArray, pad_width=((before[0], after[0]), (before[1], after[1]), (before[2], after[2])), mode='constant', constant_values=1)
        
        # coreArray = np.flip(coreArray, axis=0)

        self.coreMap = coreArray