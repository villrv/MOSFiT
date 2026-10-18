"""Physics light-curve solver used by the `physicsni` engine (from sn-modal-lc).


modal_lc.py -- Pinto & Eastman (2000)-style modal solver for radioactively
powered light curves in homologous expansion.

Physics (comoving frame, radiation dominated, homologous, gray optical opacity):

    de/dt + 4e/t = div(D grad e) + rho*eps,      D = c / (3 kappa rho)

With x = v/v_max, rho = M/(v_max^3 t^3) * eta(x),
kappa = kappa0 * k(t) * khat(x), and u = e t^4:

    du/dt = (t / t_d^2) Lop[u] + t^4 rho eps
    Lop[u] = x^-2 d/dx ( x^2 p du/dx ),   p = 1 / (khat * eta)
    t_d^2  = 3 kappa0 k(t) M / (c v_max)

Boundary: u(1) = 0 (Dirichlet), regular at x = 0.

Lop is self-adjoint with weight x^2 and depends ONLY on the profile shapes
eta(x), khat(x) -- not on M, E_k, kappa0 or the Ni distribution. So the
eigenproblem Lop phi_n = -lam_n phi_n is solved once per profile shape
(i.e. once per fit), and each mode amplitude obeys

    da_n/dtheta = -lam_n a_n + R_n,     dtheta = t dt / t_d^2(t)

which is Arnett's integral, one per mode. Uniform density + heating shaped
like the fundamental mode reproduces Arnett (1982) exactly.

Luminosity (flux through x = 1):
    L = (4 pi v_max^3 / t_d^2) * sum_n lam_n a_n c_n,   c_n = int phi_n x^2 dx

Gamma rays: gray absorption (kappa_gamma), angle-averaged over all ray
directions from each zone (not outward-only). Positrons deposited locally.

Units: cgs internally; inputs M in Msun, E_k in erg, times in days.
"""
import numpy as np
from scipy.linalg import eigh, solve_banded
from scipy.special import erfc

C = 2.99792458e10
MSUN = 1.989e33
DAY = 86400.0
SIGMA_SB = 5.670374e-5
A_RAD = 7.5657e-15
EPS_NI = 3.90e10          # erg/s per g of initial 56Ni (Nadyozhin 1994 form)
EPS_CO = 6.78e9
TAU_NI = 8.77 * DAY
TAU_CO = 111.3 * DAY
F_POS = 0.032             # fraction of 56Co decay energy in positron KE


def heating_per_gram(t):
    """(gamma, positron) specific heating, erg/s per g of initial Ni; t in s."""
    ni = (EPS_NI - EPS_CO) * np.exp(-t / TAU_NI)
    co = EPS_CO * np.exp(-t / TAU_CO)
    return ni + (1.0 - F_POS) * co, F_POS * co


def density_shape(kind, x, h, **kw):
    """Normalized eta(x): sum 4 pi x^2 eta h = 1."""
    if kind == "uniform":
        eta = np.ones_like(x)
    elif kind == "exponential":           # rho ~ exp(-v/v_e), cut at v_max
        eta = np.exp(-x * kw.get("vmax_over_ve", 10.0))
    elif kind == "scalzo":                # Scalzo et al. 2014a "Run F": [1+(v/v_KE)^3]^-3
        eta = (1.0 + (x * kw.get("vmax_over_vke", 8.0)) ** 3) ** -3
    elif kind == "tabulated":             # e.g. HESMA: v_tab (any units), rho_tab
        vt, rt = np.asarray(kw["v_tab"], float), np.asarray(kw["rho_tab"], float)
        eta = np.exp(np.interp(x * kw.get("v_max", vt.max()), vt, np.log(rt)))
    elif kind == "broken_power_law":      # Matzner & McKee style
        xt, d, n = kw.get("x_t", 0.3), kw.get("delta", 1.0), kw.get("n", 10.0)
        eta = np.where(x < xt, (x / xt) ** -d, (x / xt) ** -n)
    else:
        raise ValueError(f"unknown profile {kind}")
    return eta / np.sum(4 * np.pi * x**2 * eta * h)


def _phi_funcs(hh):
    """Exponential-integrator weights for linear-in-theta source.
    a1 = e^-h a0 + dth [R0 (phi1 - psi) + R1 psi]."""
    hh = np.asarray(hh, float)
    small = hh < 1e-4
    hs = np.where(small, 1.0, hh)
    phi1 = -np.expm1(-hs) / hs
    g2 = (1.0 - np.exp(-hs) * (1.0 + hs)) / hs**2
    phi1 = np.where(small, 1 - hh / 2 + hh**2 / 6, phi1)
    g2 = np.where(small, 0.5 - hh / 3 + hh**2 / 8, g2)
    return phi1, phi1 - g2


def _recurrence_flux_py(E, B, wts):
    """a_0 = 0; a_{i+1} = E_i * a_i + B_i;  returns S_i = sum_n wts_n a_{i,n} (len nt)."""
    nt1, n = E.shape
    out = np.zeros(nt1 + 1)
    a = np.zeros(n)
    for i in range(nt1):
        a = E[i] * a + B[i]
        out[i + 1] = a @ wts
    return out


try:                                    # optional JIT (~10x on the recurrence)
    from numba import njit

    @njit(cache=False)
    def _recurrence_flux_nb(E, B, wts):
        nt1, n = E.shape
        out = np.zeros(nt1 + 1)
        a = np.zeros(n)
        for i in range(nt1):
            s = 0.0
            for j in range(n):
                a[j] = E[i, j] * a[j] + B[i, j]
                s += wts[j] * a[j]
            out[i + 1] = s
        return out

    def _recurrence_flux(E, B, wts):
        return _recurrence_flux_nb(np.ascontiguousarray(E), np.ascontiguousarray(B),
                                   np.ascontiguousarray(wts, dtype=np.float64))
except ImportError:                     # pure numpy fallback
    _recurrence_flux = _recurrence_flux_py


def kappa_recomb(T, X_ige, k_ige_hot=0.15, k_ige_cold=0.03, k_ime=0.05,
                 T_rec=7000.0, dT=1000.0):
    """Illustrative gray effective opacity (cm^2/g): IGE opacity drops from
    k_ige_hot to k_ige_cold across T_rec (Fe III -> II); IME opacity flat.
    Parameters are placeholders to be calibrated (e.g. against SEDONA)."""
    s = 0.5 * (1.0 + np.tanh((T - T_rec) / dT))
    return X_ige * (k_ige_cold + (k_ige_hot - k_ige_cold) * s) + (1 - X_ige) * k_ime


class ModalModel:
    """Build once per profile shape; call light_curve() many times."""

    def __init__(self, profile="exponential", n_grid=400, n_modes=30,
                 kappa_shape=None, n_mu=16, n_s=32, keep_all_modes=False,
                 **profile_kw):
        N = n_grid
        h = 1.0 / N
        x = (np.arange(N) + 0.5) * h
        xf = np.arange(N + 1) * h
        eta = density_shape(profile, x, h, **profile_kw)
        khat = np.ones(N) if kappa_shape is None else kappa_shape(x)
        self.N, self.h, self.x, self.xf = N, h, x, xf
        self.eta, self.khat = eta, khat
        self.W = x**2 * h                                   # quadrature weight
        self.fm = 4 * np.pi * x**2 * eta * h               # mass fraction/cell
        self.qc = np.cumsum(self.fm) - 0.5 * self.fm        # mass coordinate
        self.I4 = np.sum(4 * np.pi * x**4 * eta * h)        # E_k = M v^2 I4 / 2

        # Stiffness matrix K (symmetric, positive): W Lop u = -K u
        p = 1.0 / (khat * eta)
        pf = 2 * p[:-1] * p[1:] / (p[:-1] + p[1:])          # harmonic mean
        g = pf * xf[1:-1] ** 2 / h
        diag = np.zeros(N)
        diag[:-1] += g
        diag[1:] += g
        self.kb = 2.0 * p[-1] / h                           # Dirichlet at x=1
        diag[-1] += self.kb
        self.Kdiag, self.Koff = diag, -g

        K = np.diag(diag) + np.diag(-g, 1) + np.diag(-g, -1)
        nkeep = N if keep_all_modes else n_modes
        lam, phi = eigh(K, np.diag(self.W), subset_by_index=[0, nkeep - 1])
        self.lam, self.phi = lam, phi                       # phi^T W phi = I
        self.cn = phi.T @ self.W
        self.n_modes = n_modes

        # Gamma-ray geometry: G(x, mu) = int eta ds along ray to surface
        mu, wmu = np.polynomial.legendre.leggauss(n_mu)
        sg, ws = np.polynomial.legendre.leggauss(n_s)
        xi, m = x[:, None], mu[None, :]
        smax = -xi * m + np.sqrt(xi**2 * m**2 + 1 - xi**2)
        s = smax[..., None] * (sg + 1) / 2
        r = np.sqrt(xi[..., None] ** 2 + s**2 + 2 * xi[..., None] * m[..., None] * s)
        self.G = 0.5 * smax * np.sum(ws * np.interp(r, x, eta), axis=-1)
        self.wmu = wmu / 2.0

        # Optical-depth shape from surface, at faces (for photosphere)
        dshape = khat * eta * h
        self.tau_shape_f = np.concatenate([np.cumsum(dshape[::-1])[::-1], [0.0]])

    # ------------------------------------------------------------------
    def ni_distribution(self, M_ni, f_mix, w_mix):
        """Ni mass per cell (g): smoothed step in mass coordinate."""
        shape = 0.5 * erfc((self.qc - f_mix) / (np.sqrt(2) * max(w_mix, 1e-6)))
        frac = self.fm * shape
        X = M_ni / (self._M * frac.sum()) * shape
        if X.max() > 1 + 1e-9:
            raise ValueError("Ni mass fraction > 1: increase f_mix or w_mix")
        return M_ni * frac / frac.sum()

    def _fdep_coarse(self, t, v, kappa_gamma, mask, n_coarse=120):
        """Deposition fraction on a coarse log-t grid for masked cells, plus the
        (index, weight) needed to interpolate it onto t."""
        tc = np.geomspace(t[0], t[-1], n_coarse)
        # f_cell depends on time only through A = kappa_g M / (v t)^2, so it is
        # tabulated once per profile in log A (0.01 dex; interpolation error < 1e-5).
        if not hasattr(self, "_ftab"):
            lA = np.linspace(-8.0, 8.0, 1601)
            tau = 10.0 ** lA[:, None, None] * self.G[None]
            self._ftab_lA = lA
            self._ftab = np.einsum("aim,m->ai", -np.expm1(-tau), self.wmu)
        lA = self._ftab_lA
        x = np.log10(kappa_gamma * self._M / (v * tc) ** 2)
        x = np.clip(x, lA[0], lA[-1])
        j = np.clip(((x - lA[0]) / (lA[1] - lA[0])).astype(int), 0, len(lA) - 2)
        u = ((x - lA[j]) / (lA[1] - lA[0]))[:, None]
        fc = self._ftab[j][:, mask] * (1 - u) + self._ftab[j + 1][:, mask] * u
        lt, ltc = np.log(t), np.log(tc)
        k = np.clip(np.searchsorted(ltc, lt) - 1, 0, n_coarse - 2)
        w = np.clip((lt - ltc[k]) / (ltc[k + 1] - ltc[k]), 0.0, 1.0)[:, None]
        return k, w, fc

    def _fdep(self, t, v, kappa_gamma, mask, n_coarse=120):
        """Gamma deposition fraction (nt, N); interpolated from coarse grid."""
        f = np.ones((len(t), self.N))
        if kappa_gamma is None:
            return f
        tc = np.geomspace(t[0], t[-1], n_coarse)
        Gm = self.G[mask]                                         # (nm, n_mu)
        tau = (kappa_gamma * self._M / (v * tc) ** 2)[:, None, None] * Gm[None]
        fc = np.einsum("jim,m->ji", -np.expm1(-tau), self.wmu)     # (n_coarse, nm)
        # shared linear interpolation in log t for all columns
        lt, ltc = np.log(t), np.log(tc)
        k = np.clip(np.searchsorted(ltc, lt) - 1, 0, n_coarse - 2)
        w = np.clip((lt - ltc[k]) / (ltc[k + 1] - ltc[k]), 0.0, 1.0)[:, None]
        f[:, mask] = fc[k] * (1 - w) + fc[k + 1] * w
        return f

    # ------------------------------------------------------------------
    def light_curve(self, t_days, M_ej, E_k, M_ni, f_mix=0.5, w_mix=0.05,
                    kappa0=0.1, kappa_gamma=0.03, k_of_t=None, ni_shape=None,
                    n_modes=None, solver="modes", n_int=2000, t_min=1e-3,
                    T_floor=0.0, static_correction=True, kappa_func=None,
                    kappa_kw=None, X_ige=None):
        """Return dict with L, Q_dep, Q_tot (erg/s), R_ph (cm), T_ph (K).

        k_of_t: callable(t_days) -> multiplier on kappa0 (spatially uniform,
                time-dependent effective opacity; keeps the modal speedup).
        ni_shape: optional per-cell relative Ni mass (overrides f_mix/w_mix).
        solver: "modes" (fast) or "pde" (Crank-Nicolson reference).
        kappa_func: optional callable(T, X_ige, **kappa_kw) -> kappa (cm^2/g).
                Self-consistent closure: the local kappa(T, X) field is
                evaluated each step from the mode-reconstructed temperature,
                and its energy-weighted ratio to the basis opacity
                (kappa0 * khat) sets the global k(t), lagged one step. Build
                the basis with kappa_shape matching the hot-state composition.
        """
        self._M = M_ej * MSUN
        M = self._M
        v = np.sqrt(2 * E_k / (M * self.I4))
        Mni = M_ni * MSUN
        t_days = np.atleast_1d(t_days)
        t = np.geomspace(t_min, t_days.max() * 1.0001, n_int) * DAY

        if ni_shape is not None:
            dmni = Mni * ni_shape / ni_shape.sum()
        else:
            dmni = self.ni_distribution(Mni, f_mix, w_mix)
        mask = dmni > 1e-12 * Mni

        qg, qp = heating_per_gram(t)
        Q_tot = Mni * (qg + qp)
        k = np.ones_like(t) if k_of_t is None else np.asarray(k_of_t(t / DAY))
        td2 = 3 * kappa0 * k * M / (C * v)
        integrand = t / td2
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
        theta += integrand[0] * t[0] / 2

        fast = solver == "modes" and kappa_func is None
        if fast:
            # Project heating onto the modes on the coarse gamma-time grid, then
            # interpolate in log t (identical to building the (nt, N) array first,
            # because the deposition fraction is itself interpolated that way).
            n = n_modes or self.n_modes
            phi_n = self.phi[:, :n]
            base = dmni @ phi_n                                   # full trapping
            if kappa_gamma is None:
                Pg = np.broadcast_to(base, (len(t), n))
                Qg = np.full(len(t), dmni.sum())
            else:
                kk, ww, fc = self._fdep_coarse(t, v, kappa_gamma, mask)
                dmm, dmo = dmni[mask], dmni[~mask]
                Pc = (fc * dmm) @ phi_n[mask] + dmo @ phi_n[~mask]
                Qc = fc @ dmm + dmo.sum()
                Pg = Pc[kk] * (1 - ww) + Pc[kk + 1] * ww
                Qg = Qc[kk] * (1 - ww[:, 0]) + Qc[kk + 1] * ww[:, 0]
            Q_dep = qg * Qg + qp * dmni.sum()
            R_fast = (td2 / (4 * np.pi * v**3))[:, None] * (qg[:, None] * Pg + qp[:, None] * base)
            Qcell = src = None
        else:
            fdep = self._fdep(t, v, kappa_gamma, mask)
            Qcell = dmni[None, :] * (qg[:, None] * fdep + qp[:, None])   # erg/s
            Q_dep = Qcell.sum(1)
            src = (td2 / (4 * np.pi * v**3))[:, None] * Qcell          # = W r

        if solver == "modes" and kappa_func is not None:
            kw = kappa_kw or {}
            X = dmni / (M * self.fm) if X_ige is None else X_ige
            kbase = kappa0 * self.khat
            n = n_modes or self.n_modes
            lam, phi, cn = self.lam[:n], self.phi[:, :n], self.cn[:n]
            Qphi = Qcell @ phi
            a = np.zeros(n); L = np.zeros(len(t)); kc = 1.0
            k = np.ones(len(t))
            td0 = 3 * kappa0 * M / (C * v)
            for i in range(len(t) - 1):
                td2i = td0 * kc
                pref = td2i / (4 * np.pi * v**3)
                dth = 0.5 * (t[i] + t[i + 1]) * (t[i + 1] - t[i]) / td2i
                hh = lam * dth
                p1, ps = _phi_funcs(hh)
                a = np.exp(-hh) * a + dth * pref * (Qphi[i] * (p1 - ps) + Qphi[i + 1] * ps)
                L[i + 1] = 4 * np.pi * v**3 / td2i * np.sum(lam * a * cn)
                if static_correction:
                    L[i + 1] += Q_dep[i + 1] - np.sum(Qphi[i + 1] * cn)
                u = phi @ a
                up = np.maximum(u, 0.0)
                wsum = np.sum(up * self.W)
                if wsum > 0:
                    T = (up / t[i + 1] ** 4 / A_RAD) ** 0.25
                    kc = np.sum(kappa_func(T, X, **kw) / kbase * up * self.W) / wsum
                k[i + 1] = kc
            td2 = td0 * k
        elif solver == "modes":
            n = n_modes or self.n_modes
            lam, phi, cn = self.lam[:n], self.phi[:, :n], self.cn[:n]
            R = R_fast if fast else src @ phi
            # All step sizes are known up front: build the exponential-integrator
            # weights for every (step, mode) at once, then run the linear recurrence
            # a_{i+1} = E_i a_i + B_i (numba if available, else a light numpy loop).
            dth = np.diff(theta)
            hh = dth[:, None] * lam[None, :]
            p1, ps = _phi_funcs(hh)
            E = np.exp(-hh)
            B = dth[:, None] * (R[:-1] * (p1 - ps) + R[1:] * ps)
            Lsum = _recurrence_flux(E, B, lam * cn)
            L = 4 * np.pi * v**3 / td2 * Lsum
            if static_correction:
                # Mode acceleration: truncated high modes treated as
                # quasi-static (a_n ~ R_n/lam_n). By completeness their summed
                # flux is Q_dep minus the projected part of the heating.
                L = L + Q_dep - 4 * np.pi * v**3 / td2 * (R @ cn)
        elif solver == "pde":
            L = self._pde(t, theta, src, v, td2)
        else:
            raise ValueError(solver)

        # Photosphere (tau = 2/3 from surface)
        A = kappa0 * k * M / (v * t) ** 2
        x_ph = np.interp((2 / 3) / A, self.tau_shape_f[::-1], self.xf[::-1])
        R_ph = x_ph * v * t
        with np.errstate(divide="ignore", invalid="ignore"):
            T_ph = (L / (4 * np.pi * SIGMA_SB * R_ph**2)) ** 0.25
        if T_floor > 0:
            low = ~(T_ph > T_floor)
            T_ph[low] = T_floor
            R_ph[low] = np.sqrt(np.maximum(L[low], 0) / (4 * np.pi * SIGMA_SB * T_floor**4))

        ti = lambda y: np.interp(np.log(t_days * DAY), np.log(t), y)
        return dict(t=t_days, L=ti(L), Q_dep=ti(Q_dep), Q_tot=ti(Q_tot),
                    R_ph=ti(R_ph), T_ph=ti(T_ph), v_max=v, k=ti(k),
                    _t=t / DAY, _L=L, _Q_dep=Q_dep, _Q_tot=Q_tot)

    def _pde(self, t, theta, src, v, td2, n_be=20):
        """Crank-Nicolson in theta on the same grid (independent of modes)."""
        N, W, d, o = self.N, self.W, self.Kdiag, self.Koff
        u = np.zeros(N)
        L = np.zeros(len(t))
        ab = np.zeros((3, N))
        for i in range(len(t) - 1):
            dth = theta[i + 1] - theta[i]
            c = dth if i < n_be else 0.5 * dth         # backward-Euler start
            ab[0, 1:] = c * o
            ab[1] = W + c * d
            ab[2, :-1] = c * o
            if i < n_be:
                rhs = W * u + dth * src[i + 1]
            else:
                Ku = d * u
                Ku[:-1] += o * u[1:]
                Ku[1:] += o * u[:-1]
                rhs = W * u - c * Ku + c * (src[i] + src[i + 1])
            u = solve_banded((1, 1), ab, rhs)
            L[i + 1] = 4 * np.pi * v**3 / td2[i + 1] * self.kb * u[-1]
        return L

    # ------------------------------------------------------------------
    def light_curve_kT(self, t_days, M_ej, E_k, M_ni, f_mix=0.5, w_mix=0.05,
                       kappa_func=kappa_recomb, kappa_kw=None, X_ige=None,
                       kappa_gamma=0.03, n_int=20000, t_min=1e-3, kref=0.1,
                       n_store=200, ni_shape=None):
        """Reference solver with local kappa(T, X): backward Euler in theta,
        opacity lagged one step (semi-implicit). T from radiation energy
        density (LTE, radiation dominated). Returns L plus diagnostics,
        including three candidate global kappa_eff(t) definitions."""
        kappa_kw = kappa_kw or {}
        self._M = M = M_ej * MSUN
        v = np.sqrt(2 * E_k / (M * self.I4))
        Mni = M_ni * MSUN
        t_days = np.atleast_1d(t_days)
        t = np.geomspace(t_min, t_days.max() * 1.0001, n_int) * DAY
        if ni_shape is not None:
            dmni = Mni * ni_shape / ni_shape.sum()
        else:
            dmni = self.ni_distribution(Mni, f_mix, w_mix)
        Xni = dmni / (M * self.fm)
        X = Xni if X_ige is None else X_ige
        qg, qp = heating_per_gram(t)
        fdep = self._fdep(t, v, kappa_gamma, dmni > 1e-12 * Mni)
        Qcell = dmni[None, :] * (qg[:, None] * fdep + qp[:, None])
        td2 = 3 * kref * M / (C * v)
        theta = t**2 / (2 * td2)
        src = td2 / (4 * np.pi * v**3) * Qcell
        N, h, W, eta, xf = self.N, self.h, self.W, self.eta, self.xf
        u = np.zeros(N)
        L = np.zeros(n_int)
        istore = set(np.unique(np.geomspace(1, n_int - 1, n_store).astype(int)))
        st = dict(t=[], kappa=[], T=[], u=[])
        keff = np.full((n_int, 3), np.nan)
        ab = np.zeros((3, N))
        for i in range(n_int - 1):
            T = (np.maximum(u, 0) / t[i] ** 4 / A_RAD) ** 0.25
            kap = kappa_func(T, X, **kappa_kw)
            keff[i] = [np.sum(kap * self.fm),                              # mass-weighted
                       np.sum(kap * u * W) / max(np.sum(u * W), 1e-300),    # energy-weighted
                       1.0 / np.sum(self.fm / kap)]                        # harmonic (mass)
            if i in istore:
                st["t"].append(t[i] / DAY); st["kappa"].append(kap); st["T"].append(T); st["u"].append(u.copy())
            p = kref / (kap * eta)
            pf = 2 * p[:-1] * p[1:] / (p[:-1] + p[1:])
            g = pf * xf[1:-1] ** 2 / h
            d = np.zeros(N); d[:-1] += g; d[1:] += g
            kb = 2 * p[-1] / h; d[-1] += kb
            dth = theta[i + 1] - theta[i]
            ab[0, 1:] = -dth * g; ab[1] = W + dth * d; ab[2, :-1] = -dth * g
            u = solve_banded((1, 1), ab, W * u + dth * src[i + 1])
            L[i + 1] = 4 * np.pi * v**3 / td2 * kb * u[-1]
        keff[-1] = keff[-2]
        ti = lambda y: np.interp(np.log(t_days * DAY), np.log(t), y)
        return dict(t=t_days, L=ti(L), Q_dep=ti(Qcell.sum(1)), v_max=v,
                    keff_mass=ti(keff[:, 0]), keff_energy=ti(keff[:, 1]),
                    keff_harm=ti(keff[:, 2]), X_ige=X,
                    store={k: np.array(val) for k, val in st.items()},
                    _t=t / DAY, _L=L, _keff=keff)


# --- Energy conservation for thermonuclear ejecta (Maeda & Iwamoto 2009; as used by Scalzo et al. 2014a)
E_NI, E_FE, E_IME = 1.56e51, 1.74e51, 1.24e51   # erg per Msun burned from C/O

def wd_binding(M):
    """Approximate C/O white-dwarf binding energy (erg): ~0.46e51 at 1.38 Msun,
    scaling as M^(7/3) (polytrope-like). Approximate to ~20%."""
    return 0.46e51 * (M / 1.38) ** (7 / 3)

def kinetic_energy(M_ej, M_ni, M_fe=0.1, M_co=0.02):
    """E_k = E_nuc - E_bind, with the remainder burned to IMEs."""
    M_ime = M_ej - M_ni - M_fe - M_co
    if M_ime < 0:
        return -1.0
    return E_NI * M_ni + E_FE * M_fe + E_IME * M_ime - wd_binding(M_ej)
