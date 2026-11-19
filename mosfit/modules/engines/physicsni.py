"""MOSFiT engine: physics-model Type Ia light curve (eigenmode diffusion in the Scalzo et al. 2014a density
profile, angle-averaged gamma-ray transport, E_k from energy conservation). Replaces nickelcobalt + diffusion.
Solver: mosfit/solvers/eigenmode_lc.py; E_k from the `ia_energetics` module.

Parameters: mejecta (Msun), fnickel (M_Ni / M_ej), fmix (0-1, Ni outer edge between its minimum
allowed value and the surface), kappa, kappagamma (cm^2/g), texplosion; E_k from `ia_energetics`.
Outputs dense luminosities, the bolometric peak time `tpeakbol` (rest days since explosion) and the peak-to-quarter-peak time
`tquarterbol` (rest days) for `hybridsed`.
"""
import numpy as np

from mosfit.modules.engines.engine import Engine
from mosfit.solvers.eigenmode_lc import EigenmodeSolver


# Important: Only define one ``Module`` class per file.


class PhysicsNi(Engine):
    """Radioactive engine + diffusion from the physics model (mosfit/solvers/eigenmode_lc.py)."""

    PROFILE = 'scalzo'        # Scalzo et al. 2014a density profile
    N_INT = 500
    PEAK_GRID = np.arange(4.0, 90.01, 0.5)
    # Nickel edge: smoothed-step width (mass coordinate) and the innermost allowed edge q_min = a M_Ni/M_ej + b.
    W_MIX = 0.05
    FLOOR_A, FLOOR_B = 1.02, 0.01
    _models = {}

    _REFERENCES = [
        {'bibcode': '2000ApJ...530..757P'},
        {'bibcode': '2014MNRAS.445.2535S'}
    ]

    def process(self, **kwargs):
        """Process module."""
        times = np.asarray(kwargs[self.key('dense_times')], dtype=float)
        t_exp = kwargs[self.key('resttexplosion')]
        m_ej = kwargs[self.key('mejecta')]
        m_ni = kwargs[self.key('fnickel')] * m_ej
        fmin = min(0.999, self.FLOOR_A * m_ni / m_ej + self.FLOOR_B)
        f_mix = fmin + (1.0 - fmin) * kwargs[self.key('fmix')]
        kappa = kwargs[self.key('kappa')]
        kappa_g = kwargs[self.key('kappagamma')]
        lum = np.zeros_like(times)
        e_k = kwargs[self.key('ekinetic')]
        ts = times - t_exp
        good = ts > 1e-3
        n_good = int(good.sum())
        t_peak = t_quarter = np.nan
        if e_k > 0 and m_ni < 0.95 * m_ej and n_good:
            if self.PROFILE not in self._models:
                self._models[self.PROFILE] = EigenmodeSolver(self.PROFILE, n_modes=30)
            mod = self._models[self.PROFILE]
            try:
                # Extra fixed grid to locate the bolometric peak (anchors the `hybridsed` phase).
                r = mod.light_curve(np.r_[ts[good], self.PEAK_GRID], m_ej, e_k, m_ni, f_mix=f_mix,
                                    w_mix=self.W_MIX, kappa0=kappa, kappa_gamma=kappa_g,
                                    n_int=self.N_INT)
                lum[good] = np.maximum(r['L'][:n_good], 0.0)
                v_max = r['v_max']
                t_peak = self._peak_time(r['L'][n_good:])
                t_quarter = self._quarter_time(r['L'][n_good:])
            except ValueError:
                v_max = 0.0
        else:
            v_max = 0.0
        lum[~np.isfinite(lum)] = 0.0
        return {self.dense_key('luminosities'): lum,
                self.key('vmax'): v_max / 1e5,
                self.key('tpeakbol'): t_peak,
                self.key('tquarterbol'): t_quarter}

    def _peak_time(self, lum):
        """Bolometric peak (rest days since explosion), parabolic refinement on PEAK_GRID."""
        g = self.PEAK_GRID
        i = int(np.clip(np.argmax(lum), 1, len(g) - 2))
        y0, y1, y2 = lum[i - 1:i + 2]
        den = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        return float(g[i] + np.clip(shift, -1.0, 1.0) * (g[1] - g[0]))

    def _quarter_time(self, lum):
        """Rest days from bolometric peak until L falls to a quarter of peak, on PEAK_GRID (sets the colour clock)."""
        g = self.PEAK_GRID
        i = int(np.argmax(lum)); j = i + int(np.argmax(lum[i:] < 0.25 * lum[i]))
        if j == i:
            return np.nan
        f = (lum[j - 1] - 0.25 * lum[i]) / (lum[j - 1] - lum[j])
        return float(g[j - 1] + f * (g[j] - g[j - 1]) - g[i])
