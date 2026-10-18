"""Definitions for the `HybridSED` class."""
import os

import numpy as np
from mosfit.constants import BOL_BAND_INDEX
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class HybridSED(SED):
    """Learned rest-frame Type Ia spectrum, luminosity-conserving.

    L_lambda(t) = L(t) * S(lambda, phase), with S the Hsiao et al. (2007) template multiplied by a smooth
    correction learned from CSP photometry (sn-modal-lc, build_hybrid_table.py) and normalised so that
    int S dlambda = 1 over 1500-25000 A. The luminosity L(t) comes from the engine (e.g. `physicsni`);
    this module only decides how that light is spread over wavelength. The table was trained with the
    luminosities of the `physicsni` engine itself. phase = days since B maximum in the rest frame
    = (t - t_explosion) - tpeakbol - tbpeak, where tpeakbol is the engine's bolometric peak and tbpeak the
    B-max delay after it (free; prior from the training sample, stored in the table as tbpeak_mean/sd).
    The table covers -15 to +70 d; outside it the nearest edge is used. Because S is a rest-frame
    spectrum, it works for any filters and any redshift. $MOSFIT_HYBRIDSED_TABLE overrides the table
    (e.g. one trained without the object being fitted).
    """

    TABLE = os.environ.get('MOSFIT_HYBRIDSED_TABLE') or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'hybridsed_table.npz')
    _cache = {}

    _REFERENCES = [
        {'bibcode': '2007ApJ...663.1187H'}
    ]

    def _load(self):
        if self.TABLE not in self._cache:
            d = np.load(self.TABLE)
            self._cache[self.TABLE] = (d['phase'], d['lam'], d['shape'])
        return self._cache[self.TABLE]

    def process(self, **kwargs):
        """Process module."""
        lum_key = self.key('luminosities')
        kwargs = self.prepare_input(lum_key, **kwargs)
        lum = np.asarray(kwargs[lum_key], dtype=float)
        times = np.asarray(kwargs['rest_times'], dtype=float)
        t_exp = kwargs[self.key('resttexplosion')]
        t_bmax = kwargs[self.key('tpeakbol')] + kwargs[self.key('tbpeak')]
        if not np.isfinite(t_bmax):
            t_bmax = 18.0
        band_indices = np.asarray(kwargs['all_band_indices'])
        frequencies = np.asarray(kwargs['all_frequencies'])
        zp1 = 1.0 + kwargs[self.key('redshift')]
        ph_tab, lam, shape = self._load()

        # Spectrum shape at each observation's rest-frame phase (linear in phase).
        phase = np.clip(times - t_exp - t_bmax, ph_tab[0], ph_tab[-1])
        k = np.clip(np.searchsorted(ph_tab, phase) - 1, 0, len(ph_tab) - 2)
        f = ((phase - ph_tab[k]) / (ph_tab[k + 1] - ph_tab[k]))[:, None]
        spec = shape[k] * (1 - f) + shape[k + 1] * f                 # (n_obs, n_lam), 1/A

        def at_wavs(rows, rest_wavs):
            rest_wavs = np.atleast_2d(rest_wavs)
            x = np.log(rest_wavs); xl = np.log(lam)
            j = np.clip(np.searchsorted(xl, x) - 1, 0, len(lam) - 2)
            w = (x - xl[j]) / (xl[j + 1] - xl[j])
            out = spec[rows][:, j] * (1 - w) + spec[rows][:, j + 1] * w
            if out.ndim == 3:                                       # (rows, 1, n_w) -> (rows, n_w)
                out = out[:, 0, :]
            out[:, (rest_wavs[0] < lam[0]) | (rest_wavs[0] > lam[-1])] = 0.0
            return out

        seds = self.alloc_seds(len(lum))
        good = (lum > 0) & np.isfinite(lum)
        banded = (band_indices >= 0) & good
        for bi in np.unique(band_indices[banded]):
            idx = np.flatnonzero(banded & (band_indices == bi))
            rest_wavs = np.asarray(self._sample_wavelengths[int(bi)], dtype=float) / zp1
            seds[idx, :len(rest_wavs)] = lum[idx, None] * at_wavs(idx, rest_wavs[None, :])
        freq_rows = (band_indices < 0) & (band_indices != BOL_BAND_INDEX) & good
        for i in np.flatnonzero(freq_rows):
            rest_wav = self.C_OVER_ANG / frequencies[i] / zp1
            seds[i, 0] = lum[i] * at_wavs(np.array([i]), np.array([[rest_wav]]))[0, 0]

        seds = self.add_to_existing_seds(seds, **kwargs)
        # Units of `seds` is ergs / s / Angstrom.
        return {'sample_wavelengths': self._sample_wavelengths, self.key('seds'): seds,
                'luminosities_out': lum, 'times_out': times}
