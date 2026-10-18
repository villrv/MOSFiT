"""MOSFiT engine: physics-model light curve (eigenmode diffusion, any density profile,
angle-averaged gamma transport, E_k from energy conservation). Replaces nickelcobalt + diffusion.
Solver: mosfit/physics_lc.py (sn-modal-lc, github.com/vvillar/sn-modal-lc).

Parameters: mejecta (Msun), fnickel (M_Ni / M_ej), fmix (0-1, Ni outer edge between its minimum
allowed value and the surface), kappa, kappagamma (cm^2/g), mfe (stable Fe, Msun), texplosion.
Outputs dense luminosities and dense photospheric radii (tau = 2/3) for `physicsphotosphere`.
"""
import numpy as np

from mosfit.modules.engines.engine import Engine
from mosfit.physics_lc import ModalModel, kinetic_energy


# Important: Only define one ``Module`` class per file.


class PhysicsNi(Engine):
    """Radioactive engine + diffusion from the physics model (modal_lc.py)."""

    PROFILE = 'scalzo'
    N_INT = 500
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
        if e_k > 0 and m_ni < 0.95 * m_ej and good.any():
            if self.PROFILE not in self._models:
                self._models[self.PROFILE] = ModalModel(self.PROFILE, n_modes=30)
            mod = self._models[self.PROFILE]
            try:
                r = mod.light_curve(ts[good], m_ej, e_k, m_ni, f_mix=f_mix, w_mix=0.05,
                                    kappa0=kappa, kappa_gamma=kappa_g, n_int=self.N_INT)
                lum[good] = np.maximum(r['L'], 0.0)
                rph[good] = r['R_ph']
                v_max = r['v_max']
            except ValueError:
                v_max = 0.0
        else:
            v_max = 0.0
        lum[~np.isfinite(lum)] = 0.0
        return {self.dense_key('luminosities'): lum,
                self.dense_key('physradius'): rph,
                self.key('vmax'): v_max / 1e5,
                self.key('ekinetic'): e_k}
