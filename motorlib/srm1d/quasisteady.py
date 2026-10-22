"""Quasi-steady bore flow with Ma et al. (2020) erosive burning.

The gas follows the steady limit of the staggered scheme in `solver.py`, solved directly instead of by marching the
acoustic waves to rest: the face mass flows are the running sum of the burn gas minus the gas stored in each cell, the
nozzle fixes the pressure of the last cell, and the discrete momentum equation of each face is marched back toward the
head end. The stored gas keeps the chamber filling and emptying at its own time scale, so the pressure still responds
to a changing burn rate. The temperature stays at the flame temperature because every cell is fed by gas at that
temperature and the energy equation has no work term. The burn rate lags its quasi-steady value exactly as in
`core.advanceFlow`, which is the only other time scale.
"""

from ._kernels import advanceQuasiSteady  # noqa: F401
