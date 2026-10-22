"""
burn_rate — Ma et al. (2020) Erosive Burning Model
=======================================================

PURPOSE:
    Computes the total burn rate (normal + erosive) at each cell in the
    motor grain port. This is the physics that distinguishes a 1D solver
    from a lumped-parameter approach: the burn rate varies along the
    grain because gas velocity (and thus convective heat transfer) varies.

THE MODEL (Ma et al. 2020, Eqs. 2-7):
    The total burn rate at any axial station is:

        r = r₀ + r_e

    where r₀ = a·Pⁿ is the normal (Saint-Robert) burn rate, and r_e is
    the erosive increment driven by convective heat transfer from the
    crossflow gas to the propellant surface.

    The erosive component comes from an energy balance at the surface:

        r_e = (T_flame - T_surface) / (ρ_p · C_ps · (T_surface - T_initial)) · h

    where h is the convective heat transfer coefficient, modified by
    transpiration (mass injection through the burning surface).

THE HEAT TRANSFER CHAIN:
    1. Haaland friction factor (Eq. 15 in Ma) — Darcy-Weisbach friction
       from the Haaland (1983) explicit correlation. Feeds into Gnielinski.

    2. Gnielinski Nusselt number (Eqs. 8-10 in Ma) — Convective heat
       transfer from the Gnielinski (2013) correlation, covering laminar,
       transition, and turbulent regimes. Includes entrance effects and
       a temperature-ratio correction.

    3. Transpiration correction (Eq. 16 in Ma) — The burning surface
       injects mass into the boundary layer, thickening it and reducing
       heat transfer. The correction is h/h₀ = β/(exp(β) - 1) where
       β = ρ_p · r · C_p / h₀.

    4. Bisection solver — The total burn rate r appears on both sides of
       the equation (through the transpiration correction), making it
       implicit. Fixed-point iteration diverges when the Jacobian > 1
       (which happens at moderate erosion). Bisection on the residual
       F(r) = r - r₀ - r_e(r) is monotonic and always converges.

ZERO ARBITRARY CONSTANTS:
    Every parameter in this model traces to either a named correlation
    (Haaland, Gnielinski) or a physical property (from CEA/RPA or
    propellant characterization). There are no tuning factors.

REFERENCES:
    Ma, Y. et al. (2020). "A New Erosive Burning Model of Solid
    Propellant Based on Heat Transfer." Int. J. Aerospace Eng., 2020.

    Haaland, S.E. (1983). "Simple and explicit formulas for the friction
    factor in turbulent pipe flow." J. Fluids Eng., 105(1), 89-90.

    Gnielinski, V. (2013). "On heat transfer in tubes." Int. J. Heat
    Mass Transfer, 63, 134-140.
"""

from ._kernels import (  # noqa: F401
    haaland_friction, gnielinski_nusselt, transpiration_correction, surface_temperature, erosive_energy_factor,
    compute_burn_rates,
)
