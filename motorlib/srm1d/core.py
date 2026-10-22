"""Quasi-1D transient bore flow with Ma et al. (2020) erosive burning.

Ignition-free: every cell with burning surface burns from t=0. Geometry
(port area, hydraulic diameter, burning area per unit length) is an input held
constant during `advanceFlow`; the caller updates it between calls from the
time-averaged burn rate that `advanceFlow` returns.
"""

from dataclasses import dataclass
import numpy as np

from ._kernels import advanceFlow  # noqa: F401
from .surface import surfaceTables, DEFAULT_AP_VOLUME_FRACTION, DEFAULT_AP_MASS_FRACTION, DEFAULT_BINDER_MASS_FRACTION


@dataclass
class FlowProperties:
    """Combustion gas and propellant inputs, SI units. Gas transport values are the CEA frozen-composition chamber values;
    equilibrium (effective) values overpredict the erosive rate by about 45%."""
    rhoPropellant: float
    cpPropellant: float
    tFlame: float
    tInitial: float
    gamma: float
    molarMass: float
    cpGas: float
    muGas: float
    kGas: float
    tabMinP: np.ndarray
    tabMaxP: np.ndarray
    tabA: np.ndarray
    tabN: np.ndarray
    roughness: float = 32e-6
    kappa: float = 0.44
    kSolid: float = 0.271
    kappaZn: float = 1.0
    apVolumeFraction: float = DEFAULT_AP_VOLUME_FRACTION
    apMassFraction: float = DEFAULT_AP_MASS_FRACTION
    binderMassFraction: float = DEFAULT_BINDER_MASS_FRACTION
    metalMassFractions: dict = None
    condensedPhaseHeat: bool = False

    def surfaceTables(self):
        """(ln burn rate, surface temperature, surface energy) tables from the ingredient laws. `apVolumeFraction` is the
        volume fraction of the propellant that is AP. The surface energy is the sensible cps (Ts - Ti) of Ma et al. unless
        `condensedPhaseHeat` is set, which uses ingredient energies from `apMassFraction`, `binderMassFraction` and
        `metalMassFractions` (any remainder costed with Al's table) including the condensed-phase heat release."""
        if self.condensedPhaseHeat:
            return surfaceTables(self.apVolumeFraction, self.tInitial, None, self.apMassFraction, self.binderMassFraction,
                                  self.metalMassFractions)
        return surfaceTables(self.apVolumeFraction, self.tInitial, cps=self.cpPropellant)

    @property
    def rSpecific(self):
        return 8.314462 / self.molarMass

    @property
    def solidDiffusivity(self):
        return self.kSolid / (self.rhoPropellant * self.cpPropellant)

    @property
    def prandtl(self):
        return self.muGas * self.cpGas / self.kGas
