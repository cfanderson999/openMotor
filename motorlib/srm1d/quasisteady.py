"""Quasi-steady bore flow with Ma et al. (2020) erosive burning.

The gas follows the steady limit of the staggered scheme in `solver.py`, solved directly instead of by marching the
acoustic waves to rest: the face mass flows are the running sum of the burn gas minus the gas stored in each cell, the
nozzle fixes the pressure of the last cell, and the discrete momentum equation of each face is marched back toward the
head end. The stored gas keeps the chamber filling and emptying at its own time scale, so the pressure still responds
to a changing burn rate. The temperature stays at the flame temperature because every cell is fed by gas at that
temperature and the energy equation has no work term. The burn rate lags its quasi-steady value exactly as in
`core.advanceFlow`, which is the only other time scale.
"""

import numpy as np
from numba import njit

from .solver import _nozzle_boundary_flow
from .burn_rate import compute_burn_rates, haaland_friction
from .core import _postPisoUpdate

PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-8
PRESSURE_FLOOR = 1.0e3


@njit(cache=True, fastmath=True)
def _solveNozzlePressure(guess, stored, source, T, aThroat, gamma, rSpecific, pAmbient, tAmbient):
    """Pressure of the last cell at which the nozzle outflow balances the source minus the storage.

    The storage is `stored[0] * x + stored[1]` and the nozzle flow rises with x, so Newton's method converges."""
    x = max(guess, PRESSURE_FLOOR)
    for _ in range(40):
        mdot, dmdp, _T, _state = _nozzle_boundary_flow(x, T, aThroat, gamma, rSpecific, pAmbient, tAmbient)
        residual = mdot + stored[0] * x + stored[1] - source
        slope = dmdp + stored[0]
        if slope <= 0.0:
            break
        step = residual / slope
        x_new = max(x - step, PRESSURE_FLOOR)
        converged = abs(x_new - x) <= 1e-12 * x
        x = x_new
        if converged:
            break
    return x


@njit(cache=True, fastmath=True)
def _steadyFlow(P, Pold, rho, u, T, aPort, dHyd, massSource, dx, n, dt, aThroat, pAmbient, tAmbient, gamma, rSpecific,
                muGas, roughness, flux, velocityCell, fDarcy, Re, inertiaScale):
    """Solves the quasi-steady flow for one substep and writes the result into P, rho and u. Returns the nozzle mass flow."""
    Pk = Pold.copy()
    fluxOld = flux.copy()
    Pnew = np.empty(n)
    rhoK = np.empty(n)
    mc = np.empty(n)
    aFace = np.empty(n + 1)
    aFace[0] = aPort[0]
    for j in range(1, n):
        aFace[j] = 0.5 * (aPort[j - 1] + aPort[j])
    aFace[n] = aPort[n - 1]

    source = 0.0
    capacity = 0.0
    for i in range(n):
        source += massSource[i] * dx
        capacity += aPort[i] * dx / (rSpecific * T[i] * dt)

    # Only the pressure x of the last cell is implicit. The rest of the profile keeps the shape it had at the start of the
    # substep and moves with x, so the stored gas q_i = c_i (x - Pold[n-1]) cannot feed back on the profile however short
    # the substep is. That feedback is what makes an iteration on the whole profile diverge once a substep is shorter
    # than the chamber filling time.
    x = _solveNozzlePressure(Pold[n - 1], np.array([capacity, -capacity * Pold[n - 1]]), source, T[n - 1], aThroat,
                             gamma, rSpecific, pAmbient, tAmbient)
    mdotNozzle = _nozzle_boundary_flow(x, T[n - 1], aThroat, gamma, rSpecific, pAmbient, tAmbient)[0]
    # Face mass flows: gas made upstream minus gas stored upstream.
    flux[0] = 0.0
    for i in range(n):
        flux[i + 1] = flux[i] + massSource[i] * dx - aPort[i] * dx / (rSpecific * T[i] * dt) * (x - Pold[n - 1])
    flux[n] = mdotNozzle

    # The flow accelerates or slows as the burn rate changes, which takes pressure: rho du/dt at each face. It is built
    # from the change in face mass flow, which a geometry update does not disturb the way it does the velocity, and is
    # fixed for the substep so it cannot feed back on the density iteration.
    inertiaForce = np.zeros(n + 1)
    densityRise = (x - Pold[n - 1]) / (rSpecific * T[n - 1])
    for j in range(1, n):
        rhoOld = 0.5 * (Pold[j - 1] / (rSpecific * T[j - 1]) + Pold[j] / (rSpecific * T[j]))
        inertiaForce[j] = inertiaScale * ((flux[j] - fluxOld[j]) / aFace[j] - flux[j] / (rhoOld * aFace[j]) * densityRise) / dt

    for _ in range(PICARD_ITERATIONS):
        for i in range(n):
            rhoK[i] = max(Pk[i], PRESSURE_FLOOR) / (rSpecific * T[i])
        u[0] = 0.0
        for j in range(1, n):
            u[j] = flux[j] / (0.5 * (rhoK[j - 1] + rhoK[j]) * aFace[j])
        u[n] = flux[n] / (rhoK[n - 1] * aFace[n])
        for i in range(n):
            velocityCell[i] = 0.5 * (u[i] + u[i + 1])
            mc[i] = rhoK[i] * velocityCell[i] * aPort[i]
            Re[i] = rhoK[i] * abs(velocityCell[i]) * dHyd[i] / muGas
            fDarcy[i] = haaland_friction(Re[i], roughness, dHyd[i])

        Pnew[n - 1] = x
        for j in range(n - 1, 0, -1):
            if mc[j - 1] >= 0.0:
                uLeft = u[j - 1] if j > 1 else 0.0
            else:
                uLeft = u[j]
            if mc[j] >= 0.0:
                uRight = u[j]
            else:
                uRight = u[j + 1] if j < n - 1 else u[j]
            conv = -(mc[j] * uRight - mc[j - 1] * uLeft) / dx
            ff = 0.5 * (fDarcy[j - 1] + fDarcy[j])
            Df = 0.5 * (dHyd[j - 1] + dHyd[j])
            rhoF = 0.5 * (rhoK[j - 1] + rhoK[j])
            friction = -ff / (2.0 * Df) * rhoF * abs(u[j]) * u[j]
            inertia = inertiaForce[j]
            Pnew[j - 1] = max(Pnew[j] - dx * (conv / aFace[j] + friction - inertia), PRESSURE_FLOOR)

        change = 0.0
        for i in range(n):
            change = max(change, abs(Pnew[i] - Pk[i]) / Pk[i])
            Pk[i] = Pnew[i]
        if change < PICARD_TOLERANCE:
            break

    for i in range(n):
        P[i] = Pk[i]
        rho[i] = Pk[i] / (rSpecific * T[i])
    u[0] = 0.0
    for j in range(1, n):
        u[j] = flux[j] / (0.5 * (rho[j - 1] + rho[j]) * aFace[j])
    u[n] = flux[n] / (rho[n - 1] * aFace[n])
    return mdotNozzle


@njit(cache=True, fastmath=True)
def advanceQuasiSteady(
    rho, u, P, T, aPort, dHyd, burnLength, xCenters,
    rTotal, rErosive, rIntegral, fDarcy, Re, Mach, uCell,
    dx, n, aThroat, pAmbient, tAmbient,
    rhoP, tInitial, tFlame, cpGas, muGas, kThermal, prandtl,
    gamma, rSpecific, roughness, kappa,
    tabMinP, tabMaxP, tabA, tabN, nTabs,
    erosiveEnabled, maxSubStep, lagFraction, tSpan,
    rDyn, rQs, rQsErosive, solidAlpha, kappaZn, tsLnRate, tsValues, hValues, flux,
):
    """Advance the gas state by `tSpan` seconds in substeps no longer than `maxSubStep`, and no longer than `lagFraction`
    of the shortest burn rate lag time. Same state arrays as `core.advanceFlow`; `rIntegral` accumulates r_total*dt.

    `flux` holds the face mass flows between calls and is filled from `u` when it is all zero.

    Returns (elapsed, nozzle mass flow integral [kg], substeps)."""
    isBurning = burnLength > 0.0
    massSource = np.zeros(n)
    Pold = np.empty(n)
    if not flux.any():
        for j in range(1, n):
            flux[j] = 0.5 * (rho[j - 1] + rho[j]) * u[j] * 0.5 * (aPort[j - 1] + aPort[j])
        flux[n] = rho[n - 1] * u[n] * aPort[n - 1]
    _postPisoUpdate(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
    elapsed = 0.0
    massNozzle = 0.0
    steps = 0
    while elapsed < tSpan:
        rTotalNew, rErosiveNew = compute_burn_rates(
            P, Re, dHyd, xCenters, isBurning, roughness,
            prandtl, kThermal, cpGas, tFlame,
            rhoP, tInitial,
            tabMinP, tabMaxP, tabA, tabN, nTabs, kappa, n, tsLnRate, tsValues, hValues, rQs,
        )
        shortest = 1e30
        for i in range(n):
            if erosiveEnabled:
                rQs[i] = rTotalNew[i]
                rQsErosive[i] = rErosiveNew[i]
            else:
                rQs[i] = rTotalNew[i] - rErosiveNew[i]
                rQsErosive[i] = 0.0
            if isBurning[i] and kappaZn > 0.0 and solidAlpha > 0.0 and rDyn[i] > 0.0:
                shortest = min(shortest, kappaZn * solidAlpha / max(rDyn[i], 1e-4) ** 2)
        dt = min(maxSubStep, lagFraction * shortest, tSpan - elapsed)
        if 0.0 < tSpan - elapsed - dt < 0.5 * dt:  # fold a sliver of a step into this one
            dt = tSpan - elapsed

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
            massSource[i] = rhoP * rTotal[i] * burnLength[i]
            rIntegral[i] += rTotal[i] * dt

        for i in range(n):
            Pold[i] = P[i]
        rhoSaved, uSaved, fluxSaved = rho.copy(), u.copy(), flux.copy()
        mdot = _steadyFlow(P, Pold, rho, u, T, aPort, dHyd, massSource, dx, n, dt, aThroat, pAmbient, tAmbient, gamma,
                           rSpecific, muGas, roughness, flux, uCell, fDarcy, Re, 1.0)
        if P.min() <= 1.5 * PRESSURE_FLOOR:
            # The momentum march ran out of pressure, as it does when the flow changes abruptly over a very short substep.
            # The inertia term is a correction, so the substep is redone without it.
            rho[:], u[:], flux[:] = rhoSaved, uSaved, fluxSaved
            mdot = _steadyFlow(P, Pold, rho, u, T, aPort, dHyd, massSource, dx, n, dt, aThroat, pAmbient, tAmbient, gamma,
                               rSpecific, muGas, roughness, flux, uCell, fDarcy, Re, 0.0)
        _postPisoUpdate(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
        massNozzle += mdot * dt
        elapsed += dt
        steps += 1
    return elapsed, massNozzle, steps
