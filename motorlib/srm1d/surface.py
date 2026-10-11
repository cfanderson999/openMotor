"""Propellant surface temperature as a function of burn rate, from ingredient pyrolysis laws.

The surface of a burning composite gasifies at a temperature that adjusts to the regression rate: a faster surface
leaves less residence time for the condensed-phase decomposition, so it runs hotter. Each gasifying ingredient follows
its own law, and the propellant value is their volume-weighted mean. That follows the time-averaged composite burning
picture of Lengelle (RTO-EN-023, Eqs. 1-6), in which AP and binder are traversed in proportion to their volume
fractions. Metals and other non-AP solids sit in the binder matrix and are counted with the binder.
"""

import numpy as np
from scipy.optimize import brentq

GAS_CONSTANT = 8.314462
CAL = 4.184

# Ammonium perchlorate: mass flux = A exp(-E / (R Ts)). Lengelle, Combustion of Solid Propellants, RTO-EN-023, Table 9.
AP_DENSITY = 1950.0
AP_PREFACTOR = 96000.0 * 10.0
AP_ACTIVATION_ENERGY = 20.0e3 * CAL

# HTPB binder: first-order degradation through a thin surface layer, Lengelle RTO-EN-023 Sec. on linear pyrolysis,
#   r^2 = (alpha / xi) Ac exp(-xi) / [(-ln Y)(1 - Ti/Ts - Qs/(cp Ts)) + Qs/(cp Ts)],  xi = E / (R Ts).
# Thermal properties and activation energy are the quoted values; Ac is set so Ts = 1060 K at 0.4 mm/s, the point
# quoted from the infrared pyrometry data, because the printed prefactor does not reproduce it.
BINDER_ACTIVATION_ENERGY = 48.6e3 * CAL
BINDER_DIFFUSIVITY = 1.0e-7
BINDER_CP = 0.39 * 1000.0 * CAL
BINDER_DEGRADATION_HEAT = -450.0 * 1000.0 * CAL
BINDER_RESIDUAL_FRACTION = 0.01
BINDER_PREFACTOR = 2.81e12

# AP volume fraction of the propellant assumed when the formulation is unknown; Ma et al. (2020) propellant 1
# (69 AP, 17 HTPB, 14 Al by mass) gives 0.60.
DEFAULT_AP_VOLUME_FRACTION = 0.60

# Condensed-phase energy needed to bring each ingredient from the initial temperature to gas at its surface [J/kg].
# AP: Lengelle RTO-EN-023 Sec. 4 gives Qc = 175 cal/g at 10 mm/s (Ts = 925 K), which already nets the heating and phase
# changes against the exothermic decomposition of the 70% that reacts in the melt and the sublimation of the other 30%;
# it varies with Ts at the melt heat capacity, 0.328 cal/g/K.
AP_SURFACE_ENTHALPY = 175.0 * 1000.0 * CAL
AP_SURFACE_ENTHALPY_TEMPERATURE = 925.0
AP_MELT_HEAT_CAPACITY = 0.328 * 1000.0 * CAL
# Binder: sensible heating plus the endothermic degradation, h = cp (Ts - Ti) - Qs with Qs = -450 cal/g (750 cal/g at 1060 K).

# Metal enthalpy rise from 298.15 K including fusion, from the NASA CEA thermo polynomials for the reference-element
# condensed phases (thermo.inp: AL(cr)/AL(L) Cox 1989 p217; Mg(cr)/Mg(L) Alcock 1993; Zn(cr)/Zn(L) Cox 1989 p221),
# [K] and [kJ/kg]. No vapor-phase term: at the propellant's chamber pressure the liquid-vapor equilibrium from this
# same thermo data puts Zn's boiling point at 1650-1990 K over 400-1500 psi, above the Ts_binder the surface law
# reaches at realistic burn rates (<1700 K below ~50 mm/s), so the metal leaves the surface as a liquid droplet and
# burns in the gas phase, same picture as Al; this breaks down at high enough rate/pressure that Ts_binder exceeds it.
METAL_TEMPERATURE = np.array([298.15, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000.])
METAL_ENTHALPY = {
    'Al': 1000.0 * np.array([0, 94, 192, 294, 400, 512, 631, 1147, 1265, 1383, 1500, 1618, 1736, 1853, 1971, 2089, 2206, 2324.]),
    'Mg': 1000.0 * np.array([0, 107, 217, 331, 449, 572, 701, 1186, 1327, 1468, 1609, 1750, 1891, 2033, 2174, 2315, 2456, 2597.]),
    'Zn': 1000.0 * np.array([0, 40, 81, 124, 281, 329, 377, 425, 473, 521, 569, 617, 665, 713, 761, 810, 858, 906.]),
}

# Mass fractions assumed when the formulation is unknown; Ma et al. (2020) propellant 1 (69 AP, 17 HTPB, 14 Al).
DEFAULT_AP_MASS_FRACTION = 0.69
DEFAULT_BINDER_MASS_FRACTION = 0.17

RATE_RANGE = (1e-4, 0.5)
RATE_POINTS = 128


def apSurfaceTemperature(rate):
    """AP surface temperature [K] when the AP surface regresses at `rate` [m/s]."""
    return AP_ACTIVATION_ENERGY / GAS_CONSTANT / np.log(AP_PREFACTOR / (AP_DENSITY * rate))


def binderSurfaceTemperature(rate, tInitial):
    """HTPB binder surface temperature [K] when the binder surface regresses at `rate` [m/s]."""
    def rateAt(ts):
        xi = BINDER_ACTIVATION_ENERGY / (GAS_CONSTANT * ts)
        heat = BINDER_DEGRADATION_HEAT / (BINDER_CP * ts)
        denominator = -np.log(BINDER_RESIDUAL_FRACTION) * (1.0 - tInitial / ts - heat) + heat
        return np.sqrt(BINDER_DIFFUSIVITY / xi * BINDER_PREFACTOR * np.exp(-xi) / denominator)
    return brentq(lambda ts: rateAt(ts) - rate, tInitial + 50.0, 4000.0)


def apSurfaceEnthalpy(ts):
    """Energy to bring AP from the initial state to gas at surface temperature `ts` [K], J/kg."""
    return AP_SURFACE_ENTHALPY + AP_MELT_HEAT_CAPACITY * (ts - AP_SURFACE_ENTHALPY_TEMPERATURE)


def binderSurfaceEnthalpy(ts, tInitial):
    """Energy to bring binder from `tInitial` to gas at surface temperature `ts`, J/kg."""
    return BINDER_CP * (ts - tInitial) - BINDER_DEGRADATION_HEAT


def metalEnthalpy(ts, tInitial, species='Al'):
    """Energy to heat a metal (including fusion) from `tInitial` to `ts`, J/kg. `species` selects the ingredient's own
    table (Al, Mg, Zn); an unlisted species falls back to Al's, the only one with no composition-specific data."""
    table = METAL_ENTHALPY.get(species, METAL_ENTHALPY['Al'])
    return np.interp(ts, METAL_TEMPERATURE, table) - np.interp(tInitial, METAL_TEMPERATURE, table)


def surfaceTables(apVolumeFraction, tInitial, cps=None, apMassFraction=None, binderMassFraction=None, metalMassFractions=None):
    """Surface temperature and surface energy against ln(burn rate [m/s]), for linear interpolation.

    Returns (lnRate, temperature, enthalpy). Temperature is the volume-weighted mean of the ingredient laws.
    Enthalpy [J/kg of propellant] is the energy per unit mass that the erosive heat flux must supply. With `cps` given
    it is the sensible heating cps (Ts - Ti) of Ma et al. (2020); otherwise it is the mass-weighted sum of the
    ingredient energies at their own surface temperatures, which includes the condensed-phase heat release.
    `metalMassFractions` is {species: mass fraction} for the non-AP, non-binder solids (e.g. {'Zn': 0.20, 'Mg': 0.15});
    any of it left unassigned is costed with Al's table, the only one without its own entry.
    """
    lnRate = np.linspace(np.log(RATE_RANGE[0]), np.log(RATE_RANGE[1]), RATE_POINTS)
    rates = np.exp(lnRate)
    tAp = np.array([apSurfaceTemperature(r) for r in rates])
    tBinder = np.array([binderSurfaceTemperature(r, tInitial) for r in rates])
    temperature = apVolumeFraction * tAp + (1.0 - apVolumeFraction) * tBinder
    if cps is not None:
        enthalpy = cps * (temperature - tInitial)
    else:
        other = 1.0 - apMassFraction - binderMassFraction
        if other < -1e-9:
            raise ValueError("AP and binder mass fractions exceed one")
        other = max(other, 0.0)
        metalMassFractions = dict(metalMassFractions or {})
        unassigned = other - sum(metalMassFractions.values())
        if unassigned < -1e-9:
            raise ValueError("metal mass fractions exceed the non-AP, non-binder remainder")
        metalMassFractions['Al'] = metalMassFractions.get('Al', 0.0) + max(unassigned, 0.0)
        metals = sum(frac * metalEnthalpy(tBinder, tInitial, species) for species, frac in metalMassFractions.items() if frac > 0.0)
        enthalpy = apMassFraction * apSurfaceEnthalpy(tAp) + binderMassFraction * binderSurfaceEnthalpy(tBinder, tInitial) + metals
    return lnRate, temperature, enthalpy
