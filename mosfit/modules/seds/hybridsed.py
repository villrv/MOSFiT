"""Definitions for the `HybridSED` class."""
import os

import numpy as np
from mosfit.constants import BOL_BAND_INDEX
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class HybridSED(SED):
    """Learned rest-frame Type Ia spectrum, luminosity-conserving.

    L_lambda(t) = L(t) * S(lambda, phase): the Hsiao et al. (2007) template times a smooth correction learned from
    CSP photometry, normalised so that int S dlambda = 1 over 1500-25000 A. L(t) comes from the `physicsni` engine;
    this module only spreads that light over wavelength. phase = rest-frame days since B maximum
    = (t - t_explosion) - tpeakbol - tbpeak (tpeakbol: the engine's bolometric peak; tbpeak: the B-max delay, free).
    The spectrum's clock and its colours are both indexed by the engine's peak-to-quarter-peak time w = `tquarterbol`
    (stretch s = exp(b + alpha (ln w - <ln w>)); colour S = H exp(W0 + x W1), x = standardised ln w), so they add no
    free parameter. The table covers -15 to +70 d; outside it the nearest edge is used. Works for any filter and
    redshift.
    """

    TABLE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hybridsed_table.npz')
    _cache = {}

    _REFERENCES = [
        {'bibcode': '2007ApJ...663.1187H'}
    ]

    def _load(self):
        if self.TABLE not in self._cache:
            d = np.load(self.TABLE)
            phys = (float(d['phys_alpha']), float(d['phys_b']), float(d['phys_lnw_mean'])) \
                if 'phys_alpha' in d.files else None
            color = (d['hsiao'], d['w0'], d['w1'], np.gradient(d['lam']), float(d['color_lnw_mean']),
                     float(d['color_lnw_sd'])) if 'w1' in d.files else None
            self._cache[self.TABLE] = (d['phase'], d['lam'], d['shape'], phys, color)
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
        ph_tab, lam, shape, phys, color = self._load()
        w = kwargs.get(self.key('tquarterbol'), np.nan)
        if color is not None and np.isfinite(w) and w > 0:
            # Colour indexed by the physics: S = H exp(W0 + x W1), x = standardised ln(peak->quarter time),
            # renormalised at every phase so the spectrum still carries exactly L(t).
            hs, w0, w1, dlam, lnw_mean, lnw_sd = color
            x = (np.log(w) - lnw_mean) / lnw_sd
            e = hs * np.exp(np.clip(w0 + x * w1, -10, 10))
            shape = e / (e * dlam).sum(1)[:, None]
        if phys is not None:
            # Clock set by the physics: s = exp(b + alpha (ln w - <ln w>)), w = engine's peak->quarter time.
            alpha, b, lnw_mean = phys
            stretch = float(np.exp(b + alpha * (np.log(w) - lnw_mean))) if np.isfinite(w) and w > 0 else 1.0
        else:
            stretch = 1.0

        # Spectrum shape at each observation's rest-frame phase (linear in phase), clock stretched by the physics.
        phase = np.clip((times - t_exp - t_bmax) / stretch, ph_tab[0], ph_tab[-1])
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
