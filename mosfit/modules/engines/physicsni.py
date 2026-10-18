"""MOSFiT engine: physics-model light curve (eigenmode diffusion, any density profile,
angle-averaged gamma transport, E_k from energy conservation). Replaces nickelcobalt + diffusion.
Solver: mosfit/physics_lc.py (sn-modal-lc, github.com/vvillar/sn-modal-lc).

Parameters: mejecta (Msun), fnickel (M_Ni / M_ej), fmix (0-1, Ni outer edge between its minimum
allowed value and the surface), kappa, kappagamma (cm^2/g), mfe (stable Fe, Msun), texplosion.
Outputs dense luminosities and dense photospheric radii (tau = 2/3) for `physicsphotosphere`,
the bolometric peak time `tpeakbol` (rest days since explosion) and the peak-to-quarter-peak time
`tquarterbol` (rest days) for `hybridsed`.
"""
import numpy as np

from mosfit.modules.engines.engine import Engine
from mosfit.physics_lc import ModalModel, kinetic_energy


# Important: Only define one ``Module`` class per file.


class PhysicsNi(Engine):
    """Radioactive engine + diffusion from the physics model (modal_lc.py)."""

    PROFILE = 'scalzo'
    N_INT = 500
    PEAK_GRID = np.arange(4.0, 90.01, 0.5)
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
        m_fe = kwargs.get(self.key('mfe'), 0.1)
        fmin = min(0.999, 1.02 * m_ni / m_ej + 0.01)
        f_mix = fmin + (1.0 - fmin) * kwargs[self.key('fmix')]
        kappa = kwargs[self.key('kappa')]
        kappa_g = kwargs[self.key('kappagamma')]

        lum = np.zeros_like(times)
        rph = np.zeros_like(times)
        e_k = kinetic_energy(m_ej, m_ni, M_fe=m_fe)
        ts = times - t_exp
        good = ts > 1e-3
        n_good = int(good.sum())
        t_peak = t_quarter = np.nan
        if e_k > 0 and m_ni < 0.95 * m_ej and n_good:
            if self.PROFILE not in self._models:
                self._models[self.PROFILE] = ModalModel(self.PROFILE, n_modes=30)
            mod = self._models[self.PROFILE]
            try:
                # Extra fixed grid to locate the bolometric peak (anchors the `physicssed` phase).
                r = mod.light_curve(np.r_[ts[good], self.PEAK_GRID], m_ej, e_k, m_ni, f_mix=f_mix,
                                    w_mix=0.05, kappa0=kappa, kappa_gamma=kappa_g, n_int=self.N_INT)
                lum[good] = np.maximum(r['L'][:n_good], 0.0)
                rph[good] = r['R_ph'][:n_good]
                v_max = r['v_max']
                t_peak = self._peak_time(r['L'][n_good:])
                t_quarter = self._quarter_time(r['L'][n_good:])
            except ValueError:
                v_max = 0.0
        else:
            v_max = 0.0
        lum[~np.isfinite(lum)] = 0.0
        return {self.dense_key('luminosities'): lum,
                self.dense_key('physradius'): rph,
                self.key('vmax'): v_max / 1e5,
                self.key('ekinetic'): e_k,
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
