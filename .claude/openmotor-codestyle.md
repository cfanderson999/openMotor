# openMotor Code Style & Development Guide

## Naming Conventions

### General Rule: camelCase Everywhere

openMotor uses **camelCase** consistently for nearly all identifiers:

- **Methods:** `getSurfaceAreaAtRegression()`, `calcKN()`, `getGeometryErrors()`
- **Variables:** `regDist`, `dRegDist`, `surfArea`, `numGrains`, `coreRadius`
- **Instance attributes:** `self.wallWeb`, `self.mapDim`, `self.dispName`, `self.geomName`
- **Module-level constants:** `gasConstant`, `atmosphericPressure`, `maximumRefDiameter`
- **Property keys (strings):** `'coreDiameter'`, `'inhibitedEnds'`, `'finLength'`

### Exceptions

- **Classes:** PascalCase — `BatesGrain`, `PerforatedGrain`, `PropertyCollection`
- **Enum members:** UPPER_CASE — `SimAlertLevel.ERROR`, `SimAlertType.GEOMETRY`

### Method Prefixes

| Prefix   | Purpose                            | Example                              |
|----------|------------------------------------|--------------------------------------|
| `get`    | Return a computed or stored value  | `getValue()`, `getPortArea()`        |
| `set`    | Assign a value                     | `setValue()`, `setProperties()`      |
| `calc`   | Perform a calculation              | `calcKN()`, `calcIdealPressure()`    |
| `is`     | Boolean query                      | `isWebLeft()`                        |
| `apply`  | Apply a dict/change to the object  | `applyDict()`, `applyChanges()`     |
| `do`     | Execute a conversion/action        | `doConversion()`, `doMigration()`   |
| `check`  | Validate / return bool             | `checkRequirements()`               |

---

## Import Style

**Order:** Standard library → Third-party → Local (relative)

```python
from abc import abstractmethod
from typing import Tuple, List, Union

import numpy as np
import skfmm
from scipy import interpolate

from . import geometry
from .properties import EnumProperty, FloatProperty, PropertyCollection
from .simResult import SimAlert, SimAlertLevel, SimAlertType
from .constants import maximumRefDiameter, maximumRefLength
```

- **Relative imports** within `motorlib` and `uilib` packages (`.`, `..`)
- **Absolute imports** for sibling packages (`import motorlib` from `uilib`)
- Blank line between each group

---

## Property System

All configurable objects inherit from `PropertyCollection` and declare properties in `__init__`:

```python
class BatesGrain(PerforatedGrain):
    geomName = "BATES"

    def __init__(self):
        super().__init__()
        self.props['coreDiameter'] = FloatProperty('Core Diameter', 'm', 0, 5)
```

### Property Types

| Class              | Use case                          |
|--------------------|-----------------------------------|
| `FloatProperty`    | Numeric with unit, min/max range  |
| `IntProperty`      | Integer with clamping             |
| `EnumProperty`     | Restricted string list            |
| `StringProperty`   | Unrestricted text                 |
| `BooleanProperty`  | Boolean flag                      |
| `PolygonProperty`  | List of polygon vertices          |
| `TabularProperty`  | Multi-tab data (propellant tabs)  |

### Reading/Writing Properties

```python
# Single property
value = obj.getProperty('coreDiameter')
obj.setProperty('coreDiameter', 0.05)

# Bulk get/set (for serialization)
d = obj.getProperties()           # -> {'coreDiameter': 0.05, ...}
obj.setProperties({'coreDiameter': 0.05, 'length': 0.14})

# Direct access
obj.props['coreDiameter'].getValue()
obj.props['coreDiameter'].setValue(0.05)
```

All internal values are stored in **SI base units** (meters, Pascals, kg, etc.). Unit conversion is only done at the UI display/input boundary.

---

## Units System

Conversion is table-based in `motorlib/units.py`. Internal storage is always SI.

```python
from motorlib import units

units.convert(0.05, 'm', 'in')        # -> 1.9685
units.getConversion('Pa', 'psi')       # -> conversion ratio
units.convFormat(101325, 'Pa', 'psi')  # -> "14.696 psi"
```

UI code converts at boundaries:

```python
# Display: SI → user preference
dispUnit = preferences.getUnit(self.prop.unit)
displayVal = motorlib.units.convert(self.prop.getValue(), self.prop.unit, dispUnit)

# Input: user preference → SI
siVal = motorlib.units.convert(editor.value(), dispUnit, self.prop.unit)
```

---

## Grain Architecture

### Class Hierarchy

```
PropertyCollection
  └── Grain (abstract)
        ├── EndBurningGrain
        └── PerforatedGrain (abstract)
              ├── BatesGrain (analytical)
              └── FmmGrain (abstract — fast marching method)
                    ├── Finocyl
                    ├── StarGrain
                    ├── ConicalGrain
                    ├── CGrain / DGrain
                    ├── MoonBurner
                    ├── RodTubeGrain
                    ├── XCore
                    └── CustomGrain
```

### Required Abstract Methods (Grain)

Every grain must implement:

| Method                         | Returns                                      |
|--------------------------------|----------------------------------------------|
| `getSurfaceAreaAtRegression()` | Burning surface area at a regression distance |
| `getVolumeAtRegression()`      | Remaining propellant volume                   |
| `getWebLeft()`                 | Shortest remaining web thickness              |
| `getMassFlux()`                | Mass flux at a position along the grain       |
| `getEndPositions()`            | `(startPos, endPos)` tuple                    |
| `getPortArea()`                | Cross-sectional port area                     |
| `simulationSetup()`            | Pre-simulation initialization                 |

### PerforatedGrain adds:

| Method              | Returns                                |
|---------------------|----------------------------------------|
| `getCorePerimeter()` | Perimeter of the core cross-section   |
| `getFaceArea()`      | Area of an exposed grain face          |

### FmmGrain adds:

| Method                   | Purpose                                |
|--------------------------|----------------------------------------|
| `generateCoreMap()`      | Build 2D boolean map of initial core   |
| `generateRegressionMap()`| Compute signed distance field via FMM  |

### Registration

Grains are registered in `motorlib/grains/__init__.py` via a class-level `geomName` attribute:

```python
grainTypes = {}
for grainType in [BatesGrain, Finocyl, ...]:
    grainTypes[grainType.geomName] = grainType

# Instantiate by name:
grain = grainTypes["BATES"]()
```

---

## Error / Validation System

Validation uses `SimAlert` objects with level and type enums:

```python
def getGeometryErrors(self) -> List[SimAlert]:
    errors = []
    if self.props['coreDiameter'].getValue() >= self.props['diameter'].getValue():
        errors.append(SimAlert(
            SimAlertLevel.ERROR,
            SimAlertType.GEOMETRY,
            'Core diameter must be less than grain diameter'
        ))
    return errors
```

- `SimAlertLevel`: `ERROR` (blocks sim), `WARNING`, `MESSAGE`
- `SimAlertType`: `GEOMETRY`, `CONSTRAINT`, `VALUE`
- Optional `location` string to identify which component has the issue

Grains call `super().getGeometryErrors()` and append their own checks.

---

## Serialization (File I/O)

Files use **YAML** format with version tagging (`.ric` extension):

```yaml
version: [0, 6, 1]
type: 3
data:
  nozzle: { throat: 0.01428, exit: 0.03, ... }
  propellant: { name: 'KNDX', density: 1785, tabs: [...] }
  grains:
    - type: BATES
      properties: { diameter: 0.083, length: 0.14, coreDiameter: 0.05, ... }
  config: { maxPressure: 1500000, timestep: 0.03, ... }
```

### Motor dict round-trip

```python
# Save
motorData = motor.getDict()
saveFile(path, motorData, fileTypes.MOTOR)

# Load
data = loadFile(path, fileTypes.MOTOR)
motor = Motor()
motor.applyDict(data)
```

### Version Migration

Old file versions are migrated through a chain of migration functions:

```python
def migrateMotor_0_5_0_to_0_6_0(data):
    data['config']['sepPressureRatio'] = 0.4
    for grain in data['grains']:
        if grain['type'] == 'Finocyl':
            grain['properties']['invertedFins'] = False
    return data
```

---

## UI Patterns (PyQt6)

### Signal/Slot Architecture

```python
class FileManager(QObject):
    fileNameChanged = pyqtSignal(str, bool)
    newMotor = pyqtSignal(object)

    def save(self):
        saveFile(self.fileName, ...)
        self.fileNameChanged.emit(self.fileName, self.isSaved())
```

Connections are made in the main window constructor:

```python
self.app.fileManager.fileNameChanged.connect(self.updateWindowTitle)
self.app.simulationManager.newSimulationResult.connect(self.updateMotorStats)
```

### Tool Pattern

Tools are `QDialog` subclasses that take user input via `PropertyCollection`, optionally require a simulation run, then modify the motor:

```python
class MyTool(Tool):
    def __init__(self, manager):
        props = {
            'targetValue': FloatProperty('Target', 'Pa', 0, 7e7)
        }
        super().__init__(manager, 'Tool Name', 'Description', props, needsSimulation=True)

    def applyChanges(self, inp, motor, simulation):
        # inp is a dict of property values from the user
        motor.nozzle.setProperty('throat', computedValue)
        self.manager.updateMotor(motor)
```

Tools are registered in `ToolManager` under category keys (`'Set'`, `'Optimize'`, `'Design'`, `'Analyze'`).

### Converter Pattern

Exporters/importers inherit from `Exporter`/`Importer` and override `doConversion()` and `checkRequirements()`:

```python
class MyExporter(Exporter):
    def __init__(self, manager):
        super().__init__(manager, 'Format Name', 'Description', {'.ext': 'File Type'})

    def doConversion(self, path, config):
        with open(path, 'w') as f:
            f.write(self.manager.simRes.getCSV(...))

    def checkRequirements(self):
        return self.manager.simRes is not None
```

---

## Docstrings

Brief triple-quoted descriptions. No `@param`/`@return` tags — types are conveyed via type hints instead.

```python
def getSurfaceAreaAtRegression(self, regDist: float) -> float:
    """Returns the surface area of the grain after it has regressed a linear distance of 'regDist'."""
```

Module-level docstrings describe the module's purpose:

```python
"""This module includes the base classes from which all grain classes should inherit."""
```

---

## Type Hints

Core `motorlib` code uses type hints from `typing` (`List`, `Tuple`, `Union`, `Dict`). NumPy arrays use `NDArray` from `numpy.typing`.

```python
def getMassFlux(self, massIn: float, dTime: float, regDist: float,
                dRegDist: float, position: float, density: float) -> float:

geomName: Union[str, None] = None
```

`uilib` code is less consistently typed — UI glue code often omits hints.

---

## Testing

Uses `unittest.TestCase`. Tests mirror `motorlib` structure under `test/unit/`.

```python
class TestMotorMethods(unittest.TestCase):
    def test_calcKN(self):
        tm = motorlib.motor.Motor()
        bg = motorlib.grains.BatesGrain()
        bg.setProperties({'diameter': 0.083, 'length': 0.14, 'coreDiameter': 0.05, ...})
        tm.grains.append(bg)
        bg.simulationSetup(motorlib.motor.MotorConfig())
        tm.nozzle.setProperties({'throat': 0.01428})

        self.assertAlmostEqual(tm.calcKN([0], 0), 180, 0)
```

- `assertAlmostEqual()` for floating point, `assertEqual()` for exact
- Set up motor/grain objects inline per test
- One assertion focus per test method

---

## Key Dependencies

| Package    | Purpose                           |
|------------|-----------------------------------|
| `numpy`    | Array math, grain maps            |
| `scipy`    | Interpolation, optimization, FMM  |
| `skfmm`    | Fast marching method for grains   |
| `PyQt6`    | UI framework                      |
| `PyYAML`   | File serialization                |
| `trimesh`  | STL mesh loading (custom3d grain) |
| `matplotlib` | Plotting in UI                  |
