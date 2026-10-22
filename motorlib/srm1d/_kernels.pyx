# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: language_level=3

"""
Quasi-1D bore flow with the Ma et al. (2020) erosive burning model: the kernels of `core`, `quasisteady` and `burn_rate`.

The flow is solved on a staggered grid, with scalars (P, rho, T) at the N cell centres and velocities at the N + 1 faces;
face 0 is the head-end wall and face N the nozzle. Two solvers share the burn rate model: `advanceFlow` marches the
compressible equations with the PISO algorithm (Issa 1986), and `advanceQuasiSteady` solves their steady limit directly.

Ported from the numba kernels of srm_1d (https://github.com/eJar0k/srm_1d, MIT), with the same numerics. Everything runs
without the GIL on scalars and typed memoryviews.
"""

from libc.math cimport sqrt, pow, exp, log, log10, fabs
from libc.stdlib cimport malloc, free
import numpy as np

cdef double PRESSURE_FLOOR = 1.0e3
cdef int PICARD_ITERATIONS = 8
cdef double PICARD_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

cdef inline double interp(double x, double* xp, double* fp, Py_ssize_t n) noexcept nogil:
    """Linear interpolation of the table (xp rising, fp) at x, held at the end values beyond the ends."""
    cdef Py_ssize_t low = 0, high = n - 1, middle
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    while high - low > 1:
        middle = (low + high) // 2
        if xp[middle] <= x:
            low = middle
        else:
            high = middle
    return (fp[low + 1] - fp[low]) / (xp[low + 1] - xp[low]) * (x - xp[low]) + fp[low]


cdef struct BurnModel:
    # What a cell needs to find its burn rate, all of it constant over a call
    double roughness, Pr, kThermal, cpGas, tFlame, rhoP, tInitial, kappa
    Py_ssize_t nTabs, nTs
    double* tabMinP
    double* tabMaxP
    double* tabA
    double* tabN
    double* tsLnRate  # surface temperature and surface energy against ln(burn rate)
    double* tsValues
    double* hValues


# ---------------------------------------------------------------------------
# Burn rate: friction, heat transfer, and the root of the erosive balance
# ---------------------------------------------------------------------------

cpdef double haaland_friction(double Re, double roughness, double D) noexcept nogil:
    """Darcy friction factor from Haaland (1983): laminar below Re 2300, Haaland's form of Colebrook-White above 4000, and
    a linear blend between."""
    cdef double eps_rel, f_lam, inv_sqrt_f, f_turb, gamma_t
    Re = max(Re, 1.0)
    eps_rel = roughness / max(D, 1e-10)
    f_lam = 64.0 / Re
    inv_sqrt_f = -1.8 * log10(pow(eps_rel, 1.11) / pow(3.7, 1.11) + 6.9 / Re)
    inv_sqrt_f = max(inv_sqrt_f, 1.0)
    f_turb = 1.0 / (inv_sqrt_f * inv_sqrt_f)
    if Re < 2300.0:
        return f_lam
    elif Re > 4000.0:
        return f_turb
    gamma_t = (Re - 2300.0) / 1700.0
    return (1.0 - gamma_t) * f_lam + gamma_t * f_turb


cpdef double gnielinski_nusselt(double Re, double Pr, double D, double L, double f, double T_gas, double T_surface,
                                double kappa) noexcept nogil:
    """Nusselt number of Gnielinski (2013) for pipe flow, with entrance effects and the gas-heating correction
    (T_gas / T_surface)^kappa (Ma et al. Eqs. 8-10): a laminar composite below Re 2300, the modified Petukhov-Kirillov
    form above 4000, a linear blend between, and never below 3.66. `L` is the distance from the head end."""
    cdef double Re_Pr_D_L, Nu2, Nu3, Nu2_term, Nu_lam, K, entrance, f8, num, den, Nu_turb, gamma_t, Nu
    Re = max(Re, 1.0)
    L = max(L, D)
    Re_Pr_D_L = Re * Pr * D / L

    Nu2 = 1.615 * pow(Re_Pr_D_L, 1.0 / 3.0)
    Nu3 = pow(2.0 / (1.0 + 22.0 * Pr), 1.0 / 6.0) * pow(Re_Pr_D_L, 0.5)
    Nu2_term = max(Nu2 - 0.7, 0.0)
    Nu_lam = pow(3.66 * 3.66 * 3.66 + 0.7 * 0.7 * 0.7 + Nu2_term * Nu2_term * Nu2_term + Nu3 * Nu3 * Nu3, 1.0 / 3.0)

    if kappa > 0.0:
        K = pow(T_gas / max(T_surface, 1.0), kappa)
    else:
        K = 1.0
    entrance = 1.0 + pow(D / L, 2.0 / 3.0)
    f8 = f / 8.0
    num = f8 * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * sqrt(f8) * (pow(Pr, 2.0 / 3.0) - 1.0)
    Nu_turb = max(num / max(den, 1e-10), 0.0) * entrance * K

    gamma_t = min(max((Re - 2300.0) / 1700.0, 0.0), 1.0)
    Nu = (1.0 - gamma_t) * Nu_lam + gamma_t * Nu_turb
    return max(Nu, 3.66)


cpdef double transpiration_correction(double beta) noexcept nogil:
    """h / h0 = beta / (exp(beta) - 1), the reduction of heat transfer by the gas leaving the burning surface
    (Ma et al. Eq. 16), held between 0 and 1."""
    cdef double beta_c, result
    if fabs(beta) < 1e-6:
        return 1.0 - beta / 2.0
    beta_c = min(max(beta, -500.0), 500.0)
    result = beta_c / (exp(beta_c) - 1.0)
    return min(max(result, 0.0), 1.0)


cdef inline Py_ssize_t select_tab_idx(const BurnModel* b, double P) noexcept nogil:
    """The tab whose pressure range holds P, as openMotor's Propellant.getCombustionProperties picks it: strictly inside
    first, else the one with the nearest boundary."""
    cdef Py_ssize_t k, best = 0
    cdef double d, best_dist
    for k in range(b.nTabs):
        if b.tabMinP[k] < P and P < b.tabMaxP[k]:
            return k
    best_dist = min(fabs(P - b.tabMinP[0]), fabs(P - b.tabMaxP[0]))
    for k in range(1, b.nTabs):
        d = min(fabs(P - b.tabMinP[k]), fabs(P - b.tabMaxP[k]))
        if d < best_dist:
            best = k
            best_dist = d
    return best


cdef inline double saint_robert_from_tabs(const BurnModel* b, double P) noexcept nogil:
    """r0 = a(P) P^n(P) with the tab that holds P."""
    cdef double P_pos = max(P, 0.0)
    cdef Py_ssize_t k
    if P_pos == 0.0:
        return 0.0
    k = select_tab_idx(b, P)
    return b.tabA[k] * pow(P_pos, b.tabN[k])


cdef inline double erosive_residual(const BurnModel* b, double r, double r0, double alpha, double h0) noexcept nogil:
    """F(r) = r - r0 - ebf(r) h(r), whose root is the total burn rate; one log and two table searches per call."""
    cdef double beta = alpha * r, h, ln_r, T_s, H
    if beta < 1e-6:
        h = h0 * (1.0 - beta / 2.0)
    elif beta > 500.0:
        h = 0.0
    else:
        h = h0 * beta / (exp(beta) - 1.0)
    ln_r = log(r)
    T_s = interp(ln_r, b.tsLnRate, b.tsValues, b.nTs)
    H = interp(ln_r, b.tsLnRate, b.hValues, b.nTs)
    return r - r0 - max(b.tFlame - T_s, 0.0) / (b.rhoP * H) * h


cdef void burn_rate_cell(const BurnModel* b, double P, double Re_local, double D_hyd, double x_from_head,
                         double r_guess, double* r_total_out, double* r_erosive_out) noexcept nogil:
    """The total burn rate at one cell (Ma et al. Eqs. 2-7): the normal rate r0 plus the erosive gain from convective heat
    transfer, which depends on the rate itself through the blowing of the surface.

    The root of F(r) = r - r0 - ebf(r) h(r) is unique and bracketed by any sign change, as F rises with r (more blowing,
    less heat transfer). Illinois regula falsi converges in about 5 evaluations where bisection needed 24 to 30, and stops
    at the same 1e-8 m/s bracket width. `r_guess`, the previous rate of the cell or 0, narrows the first bracket.
    """
    cdef double r0, f, T_surface, Nu, h0, alpha, r_lo, F_lo, r_hi, F_hi, lo, hi, F_a, F_b, width
    cdef double r_prev, r_total, r_new, F_new
    cdef int side, attempt, iteration
    r0 = saint_robert_from_tabs(b, P)
    r_total_out[0] = r0
    r_erosive_out[0] = 0.0
    if Re_local < 100.0:
        return

    f = haaland_friction(Re_local, b.roughness, D_hyd)
    T_surface = interp(log(r0), b.tsLnRate, b.tsValues, b.nTs)
    Nu = gnielinski_nusselt(Re_local, b.Pr, D_hyd, x_from_head, f, b.tFlame, T_surface, b.kappa)
    h0 = Nu * b.kThermal / D_hyd
    if h0 < 1e-6:
        return
    if T_surface - b.tInitial < 1.0:
        return

    alpha = b.rhoP * b.cpGas / h0  # beta = alpha * r

    r_lo = r0
    F_lo = erosive_residual(b, r_lo, r0, alpha, h0)
    if F_lo >= 0.0:  # transpiration too strong for any erosive burning
        return
    r_hi = r0 * 20.0
    F_hi = erosive_residual(b, r_hi, r0, alpha, h0)

    if r_guess > r0:  # the previous root is usually within a fraction of a percent: try a tight bracket around it first
        for attempt in range(2):
            width = 2e-3 if attempt == 0 else 5e-2
            lo = max(r_guess * (1.0 - width), r0)
            hi = min(r_guess * (1.0 + width), r_hi)
            F_a = erosive_residual(b, lo, r0, alpha, h0)
            F_b = erosive_residual(b, hi, r0, alpha, h0)
            if F_a < 0.0 and F_b >= 0.0:
                r_lo, F_lo, r_hi, F_hi = lo, F_a, hi, F_b
                break
            if F_a >= 0.0:
                r_hi, F_hi = lo, F_a
            elif F_b < 0.0:
                r_lo, F_lo = hi, F_b

    # It stops at the bracket width the bisection used, or when successive iterates coincide, which is what happens once
    # one end of the bracket sits on the root and the other cannot close.
    side = 0
    r_prev = r_lo
    r_total = 0.5 * (r_lo + r_hi)
    for iteration in range(60):
        if F_hi != F_lo:
            r_new = (r_lo * F_hi - r_hi * F_lo) / (F_hi - F_lo)
        else:  # the residuals at the ends have halved down to each other: bisect
            r_new = 0.5 * (r_lo + r_hi)
        F_new = erosive_residual(b, r_new, r0, alpha, h0)
        if F_new < 0.0:
            r_lo = r_new
            F_lo = F_new
            if side == -1:
                F_hi *= 0.5
            side = -1
        else:
            r_hi = r_new
            F_hi = F_new
            if side == 1:
                F_lo *= 0.5
            side = 1
        if (r_hi - r_lo) < 1e-8:
            r_total = 0.5 * (r_lo + r_hi)
            break
        if fabs(r_new - r_prev) < 1e-10:
            r_total = r_new
            break
        r_prev = r_new
        r_total = r_new
    r_total_out[0] = r_total
    r_erosive_out[0] = max(r_total - r0, 0.0)


cdef void burn_rates(const BurnModel* b, double[::1] P, double[::1] Re, double[::1] D_hyd, double[::1] x_centers,
                     const unsigned char* burning, double[::1] r_guess, double[::1] r_total, double[::1] r_erosive,
                     Py_ssize_t N) noexcept nogil:
    """The burn rates of the burning cells, and zero for the others."""
    cdef Py_ssize_t i
    for i in range(N):
        r_total[i] = 0.0
        r_erosive[i] = 0.0
        if burning[i]:
            burn_rate_cell(b, P[i], Re[i], D_hyd[i], x_centers[i], r_guess[i], &r_total[i], &r_erosive[i])


cdef void make_model(BurnModel* b, double roughness, double Pr, double kThermal, double cpGas, double tFlame, double rhoP,
                     double tInitial, double kappa, double[::1] tabMinP, double[::1] tabMaxP, double[::1] tabA,
                     double[::1] tabN, Py_ssize_t nTabs, double[::1] tsLnRate, double[::1] tsValues,
                     double[::1] hValues):
    b.roughness, b.Pr, b.kThermal, b.cpGas, b.tFlame = roughness, Pr, kThermal, cpGas, tFlame
    b.rhoP, b.tInitial, b.kappa, b.nTabs = rhoP, tInitial, kappa, nTabs
    b.tabMinP, b.tabMaxP, b.tabA, b.tabN = &tabMinP[0], &tabMaxP[0], &tabA[0], &tabN[0]
    b.nTs = tsLnRate.shape[0]
    b.tsLnRate, b.tsValues, b.hValues = &tsLnRate[0], &tsValues[0], &hValues[0]


def surface_temperature(double rate, double[::1] ts_ln_rate, double[::1] ts_values):
    """Surface temperature [K] at burn rate `rate` [m/s], interpolated linearly in ln(rate) from a table."""
    return interp(log(rate), &ts_ln_rate[0], &ts_values[0], ts_ln_rate.shape[0])


def erosive_energy_factor(double rate, double T_flame, double rho_p, double[::1] ts_ln_rate, double[::1] ts_values,
                          double[::1] h_values):
    """Erosive gain (T_f - T_s)/(rho_p H) of Ma et al., with T_s and the surface energy H taken at the burn rate."""
    cdef double ln_rate = log(rate), T_s, H
    T_s = interp(ln_rate, &ts_ln_rate[0], &ts_values[0], ts_ln_rate.shape[0])
    H = interp(ln_rate, &ts_ln_rate[0], &h_values[0], ts_ln_rate.shape[0])
    return max(T_flame - T_s, 0.0) / (rho_p * H)


def compute_burn_rates(double[::1] P, double[::1] Re, double[::1] D_hyd, double[::1] x_centers, is_burning,
                       double roughness, double Pr, double k_thermal, double Cp_gas, double T_flame, double rho_p,
                       double T_initial, double[::1] tab_min_p, double[::1] tab_max_p, double[::1] tab_a,
                       double[::1] tab_n, int n_tabs, double kappa, int N, double[::1] ts_ln_rate,
                       double[::1] ts_values, double[::1] h_values, double[::1] r_guess):
    """The total and the erosive burn rate of every cell [m/s] from its pressure, Reynolds number, hydraulic diameter and
    distance from the head end, where `is_burning` is true and zero elsewhere. `r_guess` is the previous total rate of each
    cell, 0 where there is none."""
    cdef BurnModel model
    cdef unsigned char[::1] burning = np.ascontiguousarray(is_burning, dtype=np.uint8)
    total = np.zeros(N)
    erosive = np.zeros(N)
    cdef double[::1] totalView = total, erosiveView = erosive
    make_model(&model, roughness, Pr, k_thermal, Cp_gas, T_flame, rho_p, T_initial, kappa, tab_min_p, tab_max_p, tab_a,
               tab_n, n_tabs, ts_ln_rate, ts_values, h_values)
    with nogil:
        burn_rates(&model, P, Re, D_hyd, x_centers, &burning[0], r_guess, totalView, erosiveView, N)
    return total, erosive


# ---------------------------------------------------------------------------
# Nozzle boundary
# ---------------------------------------------------------------------------

cdef struct NozzleFlow:
    double mdot  # positive out of the chamber, negative for ambient inflow
    double dmdp  # derivative with respect to the pressure of the last cell
    double T  # temperature of the gas that crosses


cdef inline double critical_pressure_ratio(double gamma) noexcept nogil:
    return pow(2.0 / (gamma + 1.0), gamma / (gamma - 1.0))


cdef inline double critical_flow_function(double gamma) noexcept nogil:
    return sqrt(gamma) * pow(2.0 / (gamma + 1.0), (gamma + 1.0) / (2.0 * (gamma - 1.0)))


cdef inline double subsonic_throat_mdot_mag(double P0, double T0, double Pb, double A_throat, double gamma,
                                            double R_specific) noexcept nogil:
    """Magnitude of the isentropic subsonic throat flow from a reservoir to a back pressure."""
    cdef double pr, e1, e2, f, coeff
    if (P0 <= 0.0 or T0 <= 0.0 or Pb <= 0.0 or Pb >= P0 or A_throat <= 0.0 or gamma <= 1.0 or R_specific <= 0.0):
        return 0.0
    pr = Pb / P0
    e1 = 2.0 / gamma
    e2 = (gamma + 1.0) / gamma
    f = pow(pr, e1) - pow(pr, e2)
    if f <= 0.0:
        return 0.0
    coeff = 2.0 * gamma / ((gamma - 1.0) * R_specific * T0)
    return A_throat * P0 * sqrt(coeff * f)


cdef inline double subsonic_outflow_dmdp(double P_cell, double T_cell, double P_ambient, double A_throat, double gamma,
                                         double R_specific) noexcept nogil:
    """d(mdot_out)/dP_cell on the subsonic outflow branch."""
    cdef double pr = P_ambient / P_cell, e1 = 2.0 / gamma, e2 = (gamma + 1.0) / gamma
    cdef double f = pow(pr, e1) - pow(pr, e2), coeff, root, dfdpr, dmdp
    if f <= 1.0e-16:
        return 0.0
    coeff = 2.0 * gamma / ((gamma - 1.0) * R_specific * T_cell)
    root = sqrt(coeff * f)
    dfdpr = e1 * pow(pr, e1 - 1.0) - e2 * pow(pr, e2 - 1.0)
    dmdp = A_throat * (root - 0.5 * coeff * pr * dfdpr / root)
    return dmdp if dmdp >= 0.0 else 0.0


cdef inline double subsonic_inflow_dmdp(double P_cell, double T_ambient, double P_ambient, double A_throat, double gamma,
                                        double R_specific) noexcept nogil:
    """d(signed mdot)/dP_cell on the subsonic ambient-inflow branch."""
    cdef double pr = P_cell / P_ambient, e1 = 2.0 / gamma, e2 = (gamma + 1.0) / gamma
    cdef double f = pow(pr, e1) - pow(pr, e2), coeff, root, dfdpr, dmdp
    if f <= 1.0e-16:
        return 0.0
    coeff = 2.0 * gamma / ((gamma - 1.0) * R_specific * T_ambient)
    root = sqrt(coeff * f)
    dfdpr = e1 * pow(pr, e1 - 1.0) - e2 * pow(pr, e2 - 1.0)
    dmdp = -0.5 * A_throat * coeff * dfdpr / root
    return dmdp if dmdp >= 0.0 else 0.0


cdef NozzleFlow nozzle_boundary_flow(double P_cell, double T_cell, double A_throat, double gamma, double R_specific,
                                     double P_ambient, double T_ambient) noexcept nogil:
    """The quasi-steady isentropic flow through the throat, choked or subsonic, out of the chamber or back in from the
    ambient."""
    cdef NozzleFlow out
    cdef double Pc, Tc, Pa, Ta, balance_tol, gamma_fn, crit_pr, pr, coeff, mdot
    out.mdot = 0.0
    out.dmdp = 0.0
    out.T = T_cell
    if A_throat <= 0.0 or gamma <= 1.0 or R_specific <= 0.0:
        return out

    Pc = max(P_cell, 1.0)
    Tc = max(T_cell, 1.0)
    Pa = max(P_ambient, 1.0)
    Ta = max(T_ambient, 1.0)
    out.T = Tc
    balance_tol = max(1.0e-6, 1.0e-10 * Pa)
    if fabs(Pc - Pa) <= balance_tol:
        return out

    gamma_fn = critical_flow_function(gamma)
    crit_pr = critical_pressure_ratio(gamma)

    if Pc > Pa:
        pr = Pa / Pc
        if pr <= crit_pr:
            coeff = A_throat * gamma_fn / sqrt(R_specific * Tc)
            out.mdot = Pc * coeff
            out.dmdp = coeff
            return out
        out.mdot = subsonic_throat_mdot_mag(Pc, Tc, Pa, A_throat, gamma, R_specific)
        out.dmdp = subsonic_outflow_dmdp(Pc, Tc, Pa, A_throat, gamma, R_specific)
        return out

    out.T = Ta
    pr = Pc / Pa
    if pr <= crit_pr:
        out.mdot = -Pa * A_throat * gamma_fn / sqrt(R_specific * Ta)
        return out
    out.mdot = -subsonic_throat_mdot_mag(Pa, Ta, Pc, A_throat, gamma, R_specific)
    out.dmdp = subsonic_inflow_dmdp(Pc, Ta, Pa, A_throat, gamma, R_specific)
    return out


# ---------------------------------------------------------------------------
# Transient solver: one PISO step
# ---------------------------------------------------------------------------

cdef struct PisoWork:
    # Scratch for one step, each of length N + 1
    double* aFace
    double* uStar
    double* dFace
    double* aSub
    double* aDiag
    double* aSup
    double* bRhs
    double* pPrime
    double* pNew
    double* uNew
    double* rhoNew1
    double* tNew
    double* cPrime
    double* dPrime


cdef PisoWork* piso_work_new(Py_ssize_t N) noexcept nogil:
    cdef PisoWork* w = <PisoWork*>malloc(sizeof(PisoWork))
    cdef double* block = <double*>malloc(14 * (N + 1) * sizeof(double))
    cdef Py_ssize_t m = N + 1
    w.aFace, w.uStar, w.dFace, w.aSub, w.aDiag, w.aSup, w.bRhs = block, block + m, block + 2 * m, block + 3 * m, \
        block + 4 * m, block + 5 * m, block + 6 * m
    w.pPrime, w.pNew, w.uNew, w.rhoNew1, w.tNew, w.cPrime, w.dPrime = block + 7 * m, block + 8 * m, block + 9 * m, \
        block + 10 * m, block + 11 * m, block + 12 * m, block + 13 * m
    return w


cdef void piso_work_free(PisoWork* w) noexcept nogil:
    free(w.aFace)  # the block starts at aFace
    free(w)


cdef void thomas_solve(double* a, double* b, double* c, double* d, Py_ssize_t N, double* x, double* c_prime,
                       double* d_prime) noexcept nogil:
    """Solve the tridiagonal system a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i] with the Thomas algorithm."""
    cdef Py_ssize_t i
    cdef double denominator
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, N):
        denominator = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator
    x[N - 1] = d_prime[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]


cdef void piso_step(double[::1] rho, double[::1] u, double[::1] P, double[::1] T, double[::1] A_port,
                    double[::1] D_hyd, double[::1] mass_source, double[::1] thermal_source, double[::1] f_darcy,
                    double dx, double dt, double[::1] gamma_arr, double[::1] R_arr, double[::1] Cp_arr,
                    double[::1] T_ceiling_arr, double A_throat, double P_ambient, double T_ambient, Py_ssize_t N,
                    double port_mach_cap, PisoWork* w) noexcept nogil:
    """One PISO step on the staggered grid, advancing (rho, u, P, T) in place.

    A momentum predictor with the old pressure is followed by two pressure corrections, whose tridiagonal systems come
    from continuity at each cell with the velocity corrections u' = -dt/rho_f dP'/dx, and then by the energy equation, which
    advects the sensible enthalpy Cp T so that cells of different Cp conserve energy, and the equation of state. The
    nozzle boundary uses the mixture of the last cell.
    """
    cdef Py_ssize_t i, j
    cdef double gamma_bnd = gamma_arr[N - 1], R_bnd = R_arr[N - 1]
    cdef double rho_f, A_f, dPdx, u_at_jm1, u_at_j, mdot_jm1, mdot_j, u_upwind_left, u_upwind_right, conv, f_f, D_f
    cdef double friction, RT_local, a_t, coeff_w, coeff_e, rho_w, rho_e, mdot_star_w, mdot_star_e
    cdef double gamma_f, R_f, T_f, a_f, u_lim, Cp_i, old_mass, new_mass, h_old, flux_h_w, flux_h_e, mdot_w, mdot_e
    cdef double convective, h_new, T_raw, T_floor, T_clipped, mdot_boundary, rho_boundary
    cdef int sweep
    cdef NozzleFlow nozzle

    w.aFace[0] = A_port[0]
    for j in range(1, N):
        w.aFace[j] = 0.5 * (A_port[j - 1] + A_port[j])
    w.aFace[N] = A_port[N - 1]

    # Momentum predictor at the faces 1 to N - 1, with the pressure gradient of the two adjacent cells.
    w.uStar[0] = 0.0
    for j in range(1, N):
        rho_f = 0.5 * (rho[j - 1] + rho[j])
        A_f = w.aFace[j]
        if rho_f < 1e-10 or A_f < 1e-10:
            w.uStar[j] = 0.0
            continue
        dPdx = (P[j] - P[j - 1]) / dx
        u_at_jm1 = 0.5 * (u[j - 1] + u[j])
        u_at_j = 0.5 * (u[j] + u[j + 1])
        mdot_jm1 = rho[j - 1] * u_at_jm1 * A_port[j - 1]
        mdot_j = rho[j] * u_at_j * A_port[j]
        if mdot_jm1 >= 0:
            u_upwind_left = u[j - 1] if j > 1 else 0.0
        else:
            u_upwind_left = u[j]
        if mdot_j >= 0:
            u_upwind_right = u[j]
        else:
            u_upwind_right = u[j + 1] if j < N - 1 else u[j]
        conv = -(mdot_j * u_upwind_right - mdot_jm1 * u_upwind_left) / dx
        f_f = 0.5 * (f_darcy[j - 1] + f_darcy[j])
        D_f = 0.5 * (D_hyd[j - 1] + D_hyd[j])
        friction = -f_f / (2.0 * D_f) * rho_f * fabs(u[j]) * u[j]
        w.uStar[j] = u[j] + dt / rho_f * (-dPdx + conv / A_f + friction)
    w.uStar[N] = w.uStar[N - 1]

    # Pressure corrections. The second repeats the first on the corrected fields, which gives PISO its temporal accuracy
    # without outer iterations. Sweep 0 starts from the old fields and sweep 1 from the first correction.
    for sweep in range(2):
        w.dFace[0] = 0.0
        w.dFace[N] = 0.0
        for j in range(1, N):
            if sweep == 0:
                rho_f = 0.5 * (rho[j - 1] + rho[j])
            else:
                rho_f = 0.5 * (w.rhoNew1[j - 1] + w.rhoNew1[j])
            w.dFace[j] = dt / max(rho_f, 1e-6)

        for i in range(N):
            RT_local = R_arr[i] * T[i]
            a_t = A_port[i] * dx / (RT_local * dt)
            coeff_w = w.aFace[i] * w.aFace[i] * w.dFace[i] / dx if i > 0 else 0.0
            coeff_e = w.aFace[i + 1] * w.aFace[i + 1] * w.dFace[i + 1] / dx if i < N - 1 else 0.0
            w.aSub[i] = -coeff_w
            w.aSup[i] = -coeff_e
            w.aDiag[i] = a_t + coeff_w + coeff_e

            if i > 0:
                if sweep == 0:
                    rho_w = 0.5 * (rho[i - 1] + rho[i])
                    mdot_star_w = rho_w * w.uStar[i] * w.aFace[i]
                else:
                    rho_w = 0.5 * (w.rhoNew1[i - 1] + w.rhoNew1[i])
                    mdot_star_w = rho_w * w.uNew[i] * w.aFace[i]
            else:
                mdot_star_w = 0.0
            if i < N - 1:
                if sweep == 0:
                    rho_e = 0.5 * (rho[i] + rho[i + 1])
                    mdot_star_e = rho_e * w.uStar[i + 1] * w.aFace[i + 1]
                else:
                    rho_e = 0.5 * (w.rhoNew1[i] + w.rhoNew1[i + 1])
                    mdot_star_e = rho_e * w.uNew[i + 1] * w.aFace[i + 1]
            else:  # the nozzle flow responds to the pressure correction of the last cell
                if sweep == 0:
                    nozzle = nozzle_boundary_flow(P[i], T[i], A_throat, gamma_bnd, R_bnd, P_ambient, T_ambient)
                else:
                    nozzle = nozzle_boundary_flow(w.pNew[i], T[i], A_throat, gamma_bnd, R_bnd, P_ambient, T_ambient)
                w.aDiag[i] += nozzle.dmdp
                mdot_star_e = nozzle.mdot
            w.bRhs[i] = mass_source[i] * dx - (mdot_star_e - mdot_star_w)

        thomas_solve(w.aSub, w.aDiag, w.aSup, w.bRhs, N, w.pPrime, w.cPrime, w.dPrime)

        if sweep == 0:
            for i in range(N):
                w.pNew[i] = P[i] + w.pPrime[i]
            for j in range(N + 1):
                w.uNew[j] = w.uStar[j]
            for j in range(1, N):
                w.uNew[j] = w.uStar[j] - w.dFace[j] * (w.pPrime[j] - w.pPrime[j - 1]) / dx
            w.uNew[0] = 0.0
            for i in range(N):
                w.rhoNew1[i] = w.pNew[i] / (R_arr[i] * T[i])
        else:
            for i in range(N):
                w.pNew[i] = w.pNew[i] + w.pPrime[i]
            for j in range(1, N):
                w.uNew[j] = w.uNew[j] - w.dFace[j] * (w.pPrime[j] - w.pPrime[j - 1]) / dx
            w.uNew[0] = 0.0

    nozzle = nozzle_boundary_flow(w.pNew[N - 1], T[N - 1], A_throat, gamma_bnd, R_bnd, P_ambient, T_ambient)
    mdot_boundary = nozzle.mdot
    if mdot_boundary >= 0.0:
        rho_boundary = w.rhoNew1[N - 1]
    else:
        rho_boundary = P_ambient / (R_bnd * max(nozzle.T, 1.0))
    w.uNew[N] = mdot_boundary / max(rho_boundary * w.aFace[N], 1.0e-12)

    # Aerodynamic choking limiter. The pressure-based PISO has no choking limit in the port and does not capture shocks,
    # so a violent fill could drive the face velocities to supersonic values. With `port_mach_cap` above 0, the interior
    # face velocities are held within that many local speeds of sound, before the energy advection so the enthalpy flux
    # is bounded too. The head wall and the nozzle face are left alone.
    if port_mach_cap > 0.0:
        for j in range(1, N):
            gamma_f = 0.5 * (gamma_arr[j - 1] + gamma_arr[j])
            R_f = 0.5 * (R_arr[j - 1] + R_arr[j])
            T_f = 0.5 * (T[j - 1] + T[j])
            a_f = sqrt(gamma_f * R_f * max(T_f, 1.0))
            u_lim = port_mach_cap * a_f
            if w.uNew[j] > u_lim:
                w.uNew[j] = u_lim
            elif w.uNew[j] < -u_lim:
                w.uNew[j] = -u_lim

    # Energy: the mass-conservative upwind update of the enthalpy of each cell, with the temperature held between the
    # ambient and the cell's ceiling.
    for i in range(N):
        Cp_i = Cp_arr[i]
        old_mass = rho[i] * A_port[i] * dx
        new_mass = max(w.rhoNew1[i] * A_port[i] * dx, 1e-16)
        h_old = Cp_i * T[i]

        if i == 0:
            flux_h_w = 0.0
        else:
            rho_w = 0.5 * (w.rhoNew1[i - 1] + w.rhoNew1[i])
            mdot_w = rho_w * w.uNew[i] * w.aFace[i]
            if mdot_w >= 0:
                flux_h_w = mdot_w * Cp_arr[i - 1] * T[i - 1]
            else:
                flux_h_w = mdot_w * Cp_i * T[i]

        if i < N - 1:
            rho_e = 0.5 * (w.rhoNew1[i] + w.rhoNew1[i + 1])
            mdot_e = rho_e * w.uNew[i + 1] * w.aFace[i + 1]
            if mdot_e >= 0:
                flux_h_e = mdot_e * Cp_i * T[i]
            else:
                flux_h_e = mdot_e * Cp_arr[i + 1] * T[i + 1]
        else:
            # Outflow carries the Cp of the last cell. So does inflow, standing in for the ambient Cp: the flow is small
            # there.
            nozzle = nozzle_boundary_flow(w.pNew[i], T[i], A_throat, gamma_bnd, R_bnd, P_ambient, T_ambient)
            flux_h_e = nozzle.mdot * Cp_i * nozzle.T

        convective = flux_h_w - flux_h_e
        h_new = (old_mass * h_old + dt * (convective + thermal_source[i] * dx)) / new_mass
        T_raw = h_new / Cp_i if Cp_i > 0.0 else h_new
        T_floor = max(1.0, T_ambient)
        T_clipped = T_raw
        if T_clipped < T_floor:
            T_clipped = T_floor
        if T_clipped > T_ceiling_arr[i]:
            T_clipped = T_ceiling_arr[i]
        w.tNew[i] = T_clipped

    for i in range(N):
        P[i] = max(w.pNew[i], PRESSURE_FLOOR)
        T[i] = w.tNew[i]
        rho[i] = P[i] / (R_arr[i] * T[i])
    for j in range(N + 1):
        u[j] = w.uNew[j]


cdef inline double compute_dt_cfl(double[::1] u, double a_sound, double dx, Py_ssize_t N, double cfl_target,
                                  double dt_max) noexcept nogil:
    """dt = CFL dx / max(|u| + a) over the faces, at least the speed of sound, and no more than `dt_max`."""
    cdef double max_wave_speed = a_sound, ws, dt
    cdef Py_ssize_t i
    for i in range(N):
        ws = fabs(u[i]) + a_sound
        if ws > max_wave_speed:
            max_wave_speed = ws
    dt = cfl_target * dx / max_wave_speed
    return min(dt, dt_max)


cdef double post_piso_update(double[::1] rho, double[::1] u, double[::1] P, double[::1] T, double[::1] dHyd,
                             double[::1] Re, double[::1] Mach, double[::1] uCell, double[::1] fDarcy, Py_ssize_t n,
                             double muGas, double gamma, double rSpecific, double roughness) noexcept nogil:
    """The cell velocity, Mach number, Reynolds number and friction factor from the state, and the largest speed of sound."""
    cdef double aMax = 0.0, aLocal
    cdef Py_ssize_t i
    for i in range(n):
        uCell[i] = 0.5 * (u[i] + u[i + 1])
        aLocal = sqrt(gamma * rSpecific * T[i])
        Mach[i] = uCell[i] / aLocal
        Re[i] = rho[i] * fabs(uCell[i]) * dHyd[i] / muGas
        fDarcy[i] = haaland_friction(Re[i], roughness, dHyd[i])
        if aLocal > aMax:
            aMax = aLocal
    return aMax


cdef inline void relax_burn_rate(Py_ssize_t i, double dt, double rhoP, double solidAlpha, double kappaZn,
                                 double[::1] burnLength, double[::1] rQs, double[::1] rQsErosive, double[::1] rDyn,
                                 double[::1] rTotal, double[::1] rErosive) noexcept nogil:
    """Relax the burn rate of cell i toward its quasi-steady value over the thermal-wave time of the condensed phase,
    tau = kappaZn alpha_s / r^2 (the Greatrix lag form). kappaZn = 0 removes the lag."""
    cdef double tau
    if not burnLength[i] > 0.0:
        rDyn[i] = 0.0
    elif rDyn[i] <= 0.0 or kappaZn <= 0.0 or solidAlpha <= 0.0:
        rDyn[i] = rQs[i]
    else:
        tau = kappaZn * solidAlpha / pow(max(rDyn[i], 1e-4), 2.0)
        rDyn[i] = rQs[i] + (rDyn[i] - rQs[i]) * exp(-dt / tau)
    rTotal[i] = rDyn[i]
    rErosive[i] = rQsErosive[i] * rDyn[i] / rQs[i] if rQs[i] > 1e-12 else 0.0


cdef inline void take_burn_rates(Py_ssize_t n, bint erosiveEnabled, double[::1] rTotalNew, double[::1] rErosiveNew,
                                 double[::1] rQs, double[::1] rQsErosive) noexcept nogil:
    """The quasi-steady burn rates: with erosive burning the total and its erosive part, without it the normal rate alone."""
    cdef Py_ssize_t i
    for i in range(n):
        if erosiveEnabled:
            rQs[i] = rTotalNew[i]
            rQsErosive[i] = rErosiveNew[i]
        else:
            rQs[i] = rTotalNew[i] - rErosiveNew[i]
            rQsErosive[i] = 0.0


def advanceFlow(
    double[::1] rho, double[::1] u, double[::1] P, double[::1] T, double[::1] aPort, double[::1] dHyd,
    double[::1] burnLength, double[::1] xCenters,
    double[::1] rTotal, double[::1] rErosive, double[::1] rIntegral, double[::1] fDarcy, double[::1] Re,
    double[::1] Mach, double[::1] uCell,
    double dx, Py_ssize_t n, double aThroat, double pAmbient, double tAmbient,
    double rhoP, double tInitial, double tFlame, double cpGas, double muGas, double kThermal, double prandtl,
    double gamma, double rSpecific, double roughness, double kappa,
    double[::1] tabMinP, double[::1] tabMaxP, double[::1] tabA, double[::1] tabN, Py_ssize_t nTabs,
    bint erosiveEnabled, double cfl, double dtMax, Py_ssize_t burnUpdateInterval, double portMachCap, double tSpan,
    Py_ssize_t maxSteps,
    double[::1] rDyn, double[::1] rQs, double[::1] rQsErosive, double solidAlpha, double kappaZn,
    double[::1] tsLnRate, double[::1] tsValues, double[::1] hValues,
):
    """Advance the gas state by `tSpan` seconds (or `maxSteps`). `rIntegral` accumulates r_total*dt per cell.

    The quasi-steady Ma burn rate is relaxed toward over the condensed-phase thermal-wave time
    tau = kappaZn * solidAlpha / r^2 (Greatrix lag form); kappaZn = 0 disables the lag. `rDyn` persists between calls.
    `tsLnRate`, `tsValues`, `hValues` tabulate surface temperature and surface energy against ln(burn rate); see `FlowProperties`.

    Returns (elapsed, nozzle mass flow integral [kg], steps)."""
    cdef BurnModel model
    cdef PisoWork* work
    cdef Py_ssize_t i, step = 0
    cdef double aMax, dt, elapsed = 0.0, massNozzle = 0.0, tCeiling
    cdef NozzleFlow nozzle
    gammaArr, rArr, cpArr = np.full(n, gamma), np.full(n, rSpecific), np.full(n, cpGas)
    tCeiling = max(tFlame, T[0]) * 1.01
    ceilingArr = np.full(n, tCeiling)
    massSource, thermalSource = np.zeros(n), np.zeros(n)
    rTotalNew, rErosiveNew = np.zeros(n), np.zeros(n)
    burning = np.ascontiguousarray(np.asarray(burnLength) > 0.0, dtype=np.uint8)
    cdef double[::1] gammaView = gammaArr, rView = rArr, cpView = cpArr, ceilingView = ceilingArr
    cdef double[::1] massView = massSource, thermalView = thermalSource, totalNewView = rTotalNew, erosiveNewView = rErosiveNew
    cdef unsigned char[::1] burningView = burning
    make_model(&model, roughness, prandtl, kThermal, cpGas, tFlame, rhoP, tInitial, kappa, tabMinP, tabMaxP, tabA, tabN,
               nTabs, tsLnRate, tsValues, hValues)
    work = piso_work_new(n)
    with nogil:
        aMax = post_piso_update(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
        while elapsed < tSpan and step < maxSteps:
            dt = compute_dt_cfl(u, aMax, dx, n + 1, cfl, dtMax)
            if elapsed + dt > tSpan:
                dt = tSpan - elapsed

            if step % burnUpdateInterval == 0:
                burn_rates(&model, P, Re, dHyd, xCenters, &burningView[0], rQs, totalNewView, erosiveNewView, n)
                take_burn_rates(n, erosiveEnabled, totalNewView, erosiveNewView, rQs, rQsErosive)

            for i in range(n):
                relax_burn_rate(i, dt, rhoP, solidAlpha, kappaZn, burnLength, rQs, rQsErosive, rDyn, rTotal, rErosive)
                massView[i] = rhoP * rTotal[i] * burnLength[i]
                thermalView[i] = massView[i] * tFlame * cpGas
                rIntegral[i] += rTotal[i] * dt

            piso_step(rho, u, P, T, aPort, dHyd, massView, thermalView, fDarcy, dx, dt, gammaView, rView, cpView,
                      ceilingView, aThroat, pAmbient, tAmbient, n, portMachCap, work)
            aMax = post_piso_update(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
            nozzle = nozzle_boundary_flow(P[n - 1], T[n - 1], aThroat, gamma, rSpecific, pAmbient, tAmbient)
            massNozzle += nozzle.mdot * dt
            elapsed += dt
            step += 1
    piso_work_free(work)
    return elapsed, massNozzle, step


# ---------------------------------------------------------------------------
# Quasi-steady solver
# ---------------------------------------------------------------------------

cdef double solve_nozzle_pressure(double guess, double stored0, double stored1, double source, double T, double aThroat,
                                  double gamma, double rSpecific, double pAmbient, double tAmbient) noexcept nogil:
    """Pressure of the last cell at which the nozzle outflow balances the source minus the storage.

    The storage is `stored0 * x + stored1` and the nozzle flow rises with x, so Newton's method converges."""
    cdef double x = max(guess, PRESSURE_FLOOR), residual, slope, x_new
    cdef NozzleFlow nozzle
    cdef bint converged
    cdef int k
    for k in range(40):
        nozzle = nozzle_boundary_flow(x, T, aThroat, gamma, rSpecific, pAmbient, tAmbient)
        residual = nozzle.mdot + stored0 * x + stored1 - source
        slope = nozzle.dmdp + stored0
        if slope <= 0.0:
            break
        x_new = max(x - residual / slope, PRESSURE_FLOOR)
        converged = fabs(x_new - x) <= 1e-12 * x
        x = x_new
        if converged:
            break
    return x


cdef inline double min_of(double[::1] values, Py_ssize_t n) noexcept nogil:
    cdef double lowest = values[0]
    cdef Py_ssize_t i
    for i in range(1, n):
        if values[i] < lowest:
            lowest = values[i]
    return lowest


cdef double steady_flow(double[::1] P, double[::1] Pold, double[::1] rho, double[::1] u, double[::1] T,
                        double[::1] aPort, double[::1] dHyd, double[::1] massSource, double dx, Py_ssize_t n, double dt,
                        double aThroat, double pAmbient, double tAmbient, double gamma, double rSpecific, double muGas,
                        double roughness, double[::1] flux, double[::1] velocityCell, double[::1] fDarcy,
                        double[::1] Re, double inertiaScale, double* scratch) noexcept nogil:
    """Solve the quasi-steady flow for one substep and write the result into P, rho and u. Returns the nozzle mass flow.

    `scratch` holds 7 arrays of n + 1."""
    cdef double* Pk = scratch
    cdef double* fluxOld = scratch + (n + 1)
    cdef double* Pnew = scratch + 2 * (n + 1)
    cdef double* rhoK = scratch + 3 * (n + 1)
    cdef double* mc = scratch + 4 * (n + 1)
    cdef double* aFace = scratch + 5 * (n + 1)
    cdef double* inertiaForce = scratch + 6 * (n + 1)
    cdef Py_ssize_t i, j
    cdef int iteration
    cdef double source = 0.0, capacity = 0.0, x, mdotNozzle, densityRise, rhoOld, uLeft, uRight, conv, ff, Df, rhoF
    cdef double friction, inertia, change
    cdef NozzleFlow nozzle

    for i in range(n):
        Pk[i] = Pold[i]
    for j in range(n + 1):
        fluxOld[j] = flux[j]
    aFace[0] = aPort[0]
    for j in range(1, n):
        aFace[j] = 0.5 * (aPort[j - 1] + aPort[j])
    aFace[n] = aPort[n - 1]

    for i in range(n):
        source += massSource[i] * dx
        capacity += aPort[i] * dx / (rSpecific * T[i] * dt)

    # Only the pressure x of the last cell is implicit. The rest of the profile keeps the shape it had at the start of the
    # substep and moves with x, so the stored gas q_i = c_i (x - Pold[n-1]) cannot feed back on the profile however short
    # the substep is. That feedback is what makes an iteration on the whole profile diverge once a substep is shorter
    # than the chamber filling time.
    x = solve_nozzle_pressure(Pold[n - 1], capacity, -capacity * Pold[n - 1], source, T[n - 1], aThroat, gamma, rSpecific,
                              pAmbient, tAmbient)
    nozzle = nozzle_boundary_flow(x, T[n - 1], aThroat, gamma, rSpecific, pAmbient, tAmbient)
    mdotNozzle = nozzle.mdot
    # Face mass flows: gas made upstream minus gas stored upstream.
    flux[0] = 0.0
    for i in range(n):
        flux[i + 1] = flux[i] + massSource[i] * dx - aPort[i] * dx / (rSpecific * T[i] * dt) * (x - Pold[n - 1])
    flux[n] = mdotNozzle

    # The flow accelerates or slows as the burn rate changes, which takes pressure: rho du/dt at each face. It is built
    # from the change in face mass flow, which a geometry update does not disturb the way it does the velocity, and is
    # fixed for the substep so it cannot feed back on the density iteration.
    for j in range(n + 1):
        inertiaForce[j] = 0.0
    densityRise = (x - Pold[n - 1]) / (rSpecific * T[n - 1])
    for j in range(1, n):
        rhoOld = 0.5 * (Pold[j - 1] / (rSpecific * T[j - 1]) + Pold[j] / (rSpecific * T[j]))
        inertiaForce[j] = inertiaScale * ((flux[j] - fluxOld[j]) / aFace[j] - flux[j] / (rhoOld * aFace[j]) * densityRise) / dt

    for iteration in range(PICARD_ITERATIONS):
        for i in range(n):
            rhoK[i] = max(Pk[i], PRESSURE_FLOOR) / (rSpecific * T[i])
        u[0] = 0.0
        for j in range(1, n):
            u[j] = flux[j] / (0.5 * (rhoK[j - 1] + rhoK[j]) * aFace[j])
        u[n] = flux[n] / (rhoK[n - 1] * aFace[n])
        for i in range(n):
            velocityCell[i] = 0.5 * (u[i] + u[i + 1])
            mc[i] = rhoK[i] * velocityCell[i] * aPort[i]
            Re[i] = rhoK[i] * fabs(velocityCell[i]) * dHyd[i] / muGas
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
            friction = -ff / (2.0 * Df) * rhoF * fabs(u[j]) * u[j]
            inertia = inertiaForce[j]
            Pnew[j - 1] = max(Pnew[j] - dx * (conv / aFace[j] + friction - inertia), PRESSURE_FLOOR)

        change = 0.0
        for i in range(n):
            change = max(change, fabs(Pnew[i] - Pk[i]) / Pk[i])
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


def advanceQuasiSteady(
    double[::1] rho, double[::1] u, double[::1] P, double[::1] T, double[::1] aPort, double[::1] dHyd,
    double[::1] burnLength, double[::1] xCenters,
    double[::1] rTotal, double[::1] rErosive, double[::1] rIntegral, double[::1] fDarcy, double[::1] Re,
    double[::1] Mach, double[::1] uCell,
    double dx, Py_ssize_t n, double aThroat, double pAmbient, double tAmbient,
    double rhoP, double tInitial, double tFlame, double cpGas, double muGas, double kThermal, double prandtl,
    double gamma, double rSpecific, double roughness, double kappa,
    double[::1] tabMinP, double[::1] tabMaxP, double[::1] tabA, double[::1] tabN, Py_ssize_t nTabs,
    bint erosiveEnabled, double maxSubStep, double lagFraction, double tSpan,
    double[::1] rDyn, double[::1] rQs, double[::1] rQsErosive, double solidAlpha, double kappaZn,
    double[::1] tsLnRate, double[::1] tsValues, double[::1] hValues, double[::1] flux,
):
    """Advance the gas state by `tSpan` seconds in substeps no longer than `maxSubStep`, and no longer than `lagFraction`
    of the shortest burn rate lag time. Same state arrays as `advanceFlow`; `rIntegral` accumulates r_total*dt.

    `flux` holds the face mass flows between calls and is filled from `u` when it is all zero.

    Returns (elapsed, nozzle mass flow integral [kg], substeps)."""
    cdef BurnModel model
    cdef Py_ssize_t i, j, steps = 0
    cdef double elapsed = 0.0, massNozzle = 0.0, shortest, dt, remaining, mdot
    cdef bint anyFlux = False
    massSource, Pold = np.zeros(n), np.zeros(n)
    rTotalNew, rErosiveNew = np.zeros(n), np.zeros(n)
    rhoSaved, uSaved, fluxSaved = np.zeros(n), np.zeros(n + 1), np.zeros(n + 1)
    scratch = np.zeros(7 * (n + 1))
    burning = np.ascontiguousarray(np.asarray(burnLength) > 0.0, dtype=np.uint8)
    cdef double[::1] massView = massSource, poldView = Pold, totalNewView = rTotalNew, erosiveNewView = rErosiveNew
    cdef double[::1] rhoSavedView = rhoSaved, uSavedView = uSaved, fluxSavedView = fluxSaved, scratchView = scratch
    cdef unsigned char[::1] burningView = burning
    make_model(&model, roughness, prandtl, kThermal, cpGas, tFlame, rhoP, tInitial, kappa, tabMinP, tabMaxP, tabA, tabN,
               nTabs, tsLnRate, tsValues, hValues)
    with nogil:
        for j in range(n + 1):
            if flux[j] != 0.0:
                anyFlux = True
        if not anyFlux:
            for j in range(1, n):
                flux[j] = 0.5 * (rho[j - 1] + rho[j]) * u[j] * 0.5 * (aPort[j - 1] + aPort[j])
            flux[n] = rho[n - 1] * u[n] * aPort[n - 1]
        post_piso_update(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
        while elapsed < tSpan:
            burn_rates(&model, P, Re, dHyd, xCenters, &burningView[0], rQs, totalNewView, erosiveNewView, n)
            take_burn_rates(n, erosiveEnabled, totalNewView, erosiveNewView, rQs, rQsErosive)
            shortest = 1e30
            for i in range(n):
                if burnLength[i] > 0.0 and kappaZn > 0.0 and solidAlpha > 0.0 and rDyn[i] > 0.0:
                    shortest = min(shortest, kappaZn * solidAlpha / pow(max(rDyn[i], 1e-4), 2.0))
            remaining = tSpan - elapsed
            dt = min(min(maxSubStep, lagFraction * shortest), remaining)
            if 0.0 < remaining - dt < 0.5 * dt:  # fold a sliver of a step into this one
                dt = remaining

            for i in range(n):
                relax_burn_rate(i, dt, rhoP, solidAlpha, kappaZn, burnLength, rQs, rQsErosive, rDyn, rTotal, rErosive)
                massView[i] = rhoP * rTotal[i] * burnLength[i]
                rIntegral[i] += rTotal[i] * dt

            for i in range(n):
                poldView[i] = P[i]
                rhoSavedView[i] = rho[i]
            for j in range(n + 1):
                uSavedView[j] = u[j]
                fluxSavedView[j] = flux[j]
            mdot = steady_flow(P, poldView, rho, u, T, aPort, dHyd, massView, dx, n, dt, aThroat, pAmbient, tAmbient, gamma,
                               rSpecific, muGas, roughness, flux, uCell, fDarcy, Re, 1.0, &scratchView[0])
            if min_of(P, n) <= 1.5 * PRESSURE_FLOOR:
                # The momentum march ran out of pressure, as it does when the flow changes abruptly over a very short
                # substep. The inertia term is a correction, so the substep is redone without it.
                for i in range(n):
                    rho[i] = rhoSavedView[i]
                for j in range(n + 1):
                    u[j] = uSavedView[j]
                    flux[j] = fluxSavedView[j]
                mdot = steady_flow(P, poldView, rho, u, T, aPort, dHyd, massView, dx, n, dt, aThroat, pAmbient, tAmbient,
                                   gamma, rSpecific, muGas, roughness, flux, uCell, fDarcy, Re, 0.0, &scratchView[0])
            post_piso_update(rho, u, P, T, dHyd, Re, Mach, uCell, fDarcy, n, muGas, gamma, rSpecific, roughness)
            massNozzle += mdot * dt
            elapsed += dt
            steps += 1
    return elapsed, massNozzle, steps
