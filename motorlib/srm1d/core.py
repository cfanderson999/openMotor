"""Quasi-1D transient bore flow with Ma et al. (2020) erosive burning.

Ignition-free: every cell with burning surface burns from t=0. Geometry
(port area, hydraulic diameter, burning area per unit length) is an input held
constant during `advanceFlow`; the caller updates it between calls from the
time-averaged burn rate that `advanceFlow` returns.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit

from .solver import piso_step, compute_dt_cfl, _nozzle_boundary_flow
from .burn_rate import compute_burn_rates, haaland_friction
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


@njit(cache=True, fastmath=True)
def _postPisoUpdate(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness):
    aMax = 0.0
    for i in range(n):
        uCell[i] = 0.5 * (u[i] + u[i + 1])
        aLocal = (gamma * rSpecific * T[i]) ** 0.5
        Mach[i] = uCell[i] / aLocal
        Re[i] = rho[i] * abs(uCell[i]) * dHyd[i] / muGas
        fDarcy[i] = haaland_friction(Re[i], roughness, dHyd[i])
        if aLocal > aMax:
            aMax = aLocal
    return aMax


@njit(cache=True, fastmath=True)
def advanceFlow(
    rho, u, P, T, aPort, dHyd, burnLength, xCenters,
    rTotal, rErosive, rIntegral, fDarcy, Re, Mach, uCell,
    dx, n, aThroat, pAmbient, tAmbient,
    rhoP, tInitial, tFlame, cpGas, muGas, kThermal, prandtl,
    gamma, rSpecific, roughness, kappa,
    tabMinP, tabMaxP, tabA, tabN, nTabs,
    erosiveEnabled, cfl, dtMax, burnUpdateInterval, portMachCap, tSpan, maxSteps,
    rDyn, rQs, rQsErosive, solidAlpha, kappaZn, tsLnRate, tsValues, hValues,
):
    """Advance the gas state by `tSpan` seconds (or `maxSteps`). `rIntegral` accumulates r_total*dt per cell.

    The quasi-steady Ma burn rate is relaxed toward over the condensed-phase thermal-wave time
    tau = kappaZn * solidAlpha / r^2 (Greatrix lag form); kappaZn = 0 disables the lag. `rDyn` persists between calls.
    `tsLnRate`, `tsValues`, `hValues` tabulate surface temperature and surface energy against ln(burn rate); see `FlowProperties`.

    Returns (elapsed, nozzle mass flow integral [kg], steps)."""
    gammaArr = np.full(n, gamma)
    rArr = np.full(n, rSpecific)
    cpArr = np.full(n, cpGas)
    tCeiling = np.full(n, max(tFlame, T[0]) * 1.01)
    isBurning = burnLength > 0.0
    massSource = np.zeros(n)
    thermalSource = np.zeros(n)
    momentumSource = np.zeros(n + 1)

    aMax = _postPisoUpdate(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
    elapsed = 0.0
    massNozzle = 0.0
    step = 0
    while elapsed < tSpan and step < maxSteps:
        dt = compute_dt_cfl(u, aMax, dx, n + 1, cfl, dtMax)
        if elapsed + dt > tSpan:
            dt = tSpan - elapsed

        if step % burnUpdateInterval == 0:
            rTotalNew, rErosiveNew = compute_burn_rates(
                P, Re, dHyd, xCenters, isBurning, roughness,
                prandtl, kThermal, cpGas, tFlame,
                rhoP, tInitial,
                tabMinP, tabMaxP, tabA, tabN, nTabs, kappa, n, tsLnRate, tsValues, hValues, rQs,
            )
            for i in range(n):
                if erosiveEnabled:
                    rQs[i] = rTotalNew[i]
                    rQsErosive[i] = rErosiveNew[i]
                else:
                    rQs[i] = rTotalNew[i] - rErosiveNew[i]
                    rQsErosive[i] = 0.0

        for i in range(n):
            if not isBurning[i]:
                rDyn[i] = 0.0
            elif rDyn[i] <= 0.0 or kappaZn <= 0.0 or solidAlpha <= 0.0:
                rDyn[i] = rQs[i]
            else:
                tau = kappaZn * solidAlpha / max(rDyn[i], 1e-4) ** 2
                rDyn[i] = rQs[i] + (rDyn[i] - rQs[i]) * np.exp(-dt / tau)
            rTotal[i] = rDyn[i]
            rErosive[i] = rQsErosive[i] * rDyn[i] / rQs[i] if rQs[i] > 1e-12 else 0.0

        for i in range(n):
            massSource[i] = rhoP * rTotal[i] * burnLength[i]
            thermalSource[i] = massSource[i] * tFlame * cpGas
            rIntegral[i] += rTotal[i] * dt

        rho[:], u[:], P[:], T[:] = piso_step(
            rho, u, P, T, aPort, dHyd,
            massSource, thermalSource, momentumSource, fDarcy,
            dx, dt, gammaArr, rArr, cpArr, tCeiling,
            aThroat, pAmbient, tAmbient, n, portMachCap,
        )
        aMax = _postPisoUpdate(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
        mdotNozzle = _nozzle_boundary_flow(P[n - 1], T[n - 1], aThroat, gamma, rSpecific, pAmbient, tAmbient)[0]
        massNozzle += mdotNozzle * dt
        elapsed += dt
        step += 1
    return elapsed, massNozzle, step
