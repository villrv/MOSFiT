"""Optical/IR/UV photometry given as a flux is fit in flux, with its errors.

Loads LSQ12dlf (Vega U/B/V/R/I magnitudes) into the ``ia_physics`` model as
magnitudes, as AB flux densities in Jy and in µJy, and as count rates with
arbitrary zeropoints, and checks that

* flux points with no magnitude are loaded (not dropped as X-ray/radio) and
  fit in flux (``magcount``), whatever units they came in;
* data, model and errors all end up in µJy (AB zeropoint 23.9), so Jy and
  µJy input give the same likelihood, and so do two zeropoints for the same
  count rates;
* the model flux of an AB flux point on a Vega band is the AB flux of the
  band magnitude the model predicts (the band's AB offset is applied);
* errors are the given flux errors, in µJy;
* a point with both a magnitude and a count rate is fit in magnitudes unless
  ``--flux`` (``prefer_fluxes``) is set;
* negative fluxes (forced photometry before explosion) are kept and fit.
"""
from __future__ import print_function

import copy
import json
import os
import sys

import numpy as np
from schwimmbad import SerialPool

MAG_TO_FLUX_ERR = 0.4 * np.log(10.0)
AB_ZP_UJY = 2.5 * np.log10(3631.0e6)


def _detections(data, name):
    return [
        p for p in data[name]['photometry']
        if 'magnitude' in p and 'e_magnitude' in p and
        not p.get('upperlimit') and 'band' in p]


def _event(data, name, photometry):
    new = copy.deepcopy(data)
    new[name]['photometry'] = photometry
    return new


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from mosfit.fitter import Fitter
    from mosfit.model import Model
    from mosfit.modules.datas.transient import _is_band_flux

    pool = SerialPool()
    data = json.load(open(os.path.join('mosfit', 'tests', 'LSQ12dlf.json')))
    name = list(data.keys())[0]
    dets = _detections(data, name)
    failures = []

    def check(cond, msg):
        print(('ok    ' if cond else 'FAIL  ') + msg, flush=True)
        if not cond:
            failures.append(msg)

    def load(event, prefer_fluxes=False):
        fitter = Fitter(test=True, quiet=True, exit_on_prompt=True,
                        prefer_cache=True, prefer_fluxes=prefer_fluxes)
        model = Model(model='ia_physics', data=event, test=True,
                      printer=fitter._printer, fitter=fitter, pool=pool)
        if not model.load_data(event, event_name=name, pool=pool):
            return None, None
        x = np.full(model._num_free_parameters, 0.5)
        return model, x

    def run(model, x):
        out = model.run_stack(x, root='objective')
        obs = np.asarray(out['observed'], dtype=bool)
        return out, obs

    # Magnitudes: reference model predictions and band AB offsets.
    mag_model, x = load(data)
    out_m, obs_m = run(mag_model, x)
    phot = mag_model._modules['photometry']
    bands = np.asarray(out_m['all_bands'])[obs_m]
    offset_of = {}
    for bi in np.unique(out_m['all_band_indices']):
        if bi >= 0:
            offset_of[phot._unique_bands[bi]['name']] = phot._band_offsets[bi]
    model_mag = np.asarray(out_m['model_observations'], dtype=float)[obs_m]
    model_mag_ab = model_mag + np.array([offset_of[b] for b in bands])
    check(len(model_mag) == len(dets) and len(dets) > 50,
          'magnitude event loads all {} detections'.format(len(dets)))

    def flux_density_event(unit, scale, n_negative=0):
        phot_list = []
        for p in dets:
            m_ab = float(p['magnitude']) + offset_of[p['band']]
            f_ujy = 10.0 ** (-0.4 * (m_ab - AB_ZP_UJY))
            e_ujy = f_ujy * MAG_TO_FLUX_ERR * float(p['e_magnitude'])
            q = {k: v for k, v in p.items()
                 if k not in ('magnitude', 'e_magnitude', 'system')}
            q.update(fluxdensity=f_ujy / scale, e_fluxdensity=e_ujy / scale,
                     u_fluxdensity=unit)
            phot_list.append(q)
        first = min(float(p['time']) for p in dets)
        for j in range(n_negative):
            q = copy.deepcopy(phot_list[0])
            q.update(time=str(first - 40.0 - j),
                     fluxdensity=-0.5 / scale, e_fluxdensity=2.0 / scale)
            phot_list.append(q)
        return _event(data, name, phot_list)

    # Flux densities in Jy and µJy, with no magnitudes.
    jy_model, _ = load(flux_density_event('Jy', 1.0e6))
    check(jy_model is not None, 'flux-density-only (Jy) event loads')
    out_j, obs_j = run(jy_model, x)
    otypes = np.asarray(out_j['observation_types'])[obs_j]
    check(np.all(otypes == 'magcount'), 'Jy points are fit in flux')
    want_ujy = 10.0 ** (-0.4 * (model_mag_ab - AB_ZP_UJY))
    got_ujy = np.asarray(out_j['model_observations'], dtype=float)[obs_j]
    check(np.allclose(got_ujy, want_ujy, rtol=1e-9),
          'model flux (µJy) is the AB flux of the predicted Vega magnitude')
    data_ujy = np.array([
        10.0 ** (-0.4 * (float(p['magnitude']) + offset_of[p['band']] -
                         AB_ZP_UJY)) for p in dets])
    e_ujy = data_ujy * MAG_TO_FLUX_ERR * np.array(
        [float(p['e_magnitude']) for p in dets])
    check(np.allclose(np.asarray(out_j['countrates'], dtype=float),
                      data_ujy, rtol=1e-9), 'Jy data are rescaled to µJy')
    check(np.allclose(np.asarray(out_j['kdiagonal'], dtype=float),
                      e_ujy ** 2, rtol=1e-9),
          'error diagonal is the given flux errors squared, in µJy')
    check(np.allclose(np.asarray(out_j['kresiduals'], dtype=float),
                      np.abs(want_ujy - data_ujy), rtol=1e-9),
          'residuals are model minus data in µJy')

    ujy_model, _ = load(flux_density_event('µJy', 1.0))
    check(np.isclose(jy_model.ln_likelihood(x), ujy_model.ln_likelihood(x),
                     rtol=1e-12),
          'Jy and µJy input give the same likelihood')

    # Negative fluxes (forced photometry before explosion) are kept.
    neg_model, _ = load(flux_density_event('mJy', 1.0e3, n_negative=3))
    out_n, obs_n = run(neg_model, x)
    check(int(np.sum(obs_n)) == len(dets) + 3,
          'negative flux points are kept')
    check(np.isfinite(neg_model.ln_likelihood(x)),
          'likelihood with negative fluxes is finite')

    # Count rates in the band's (Vega) system: the zeropoint's value is
    # irrelevant once rescaled.
    def countrate_event(zp, with_mag=False):
        phot_list = []
        for p in dets:
            m = float(p['magnitude'])
            ct = 10.0 ** (-0.4 * (m - zp))
            q = {k: v for k, v in p.items()
                 if k not in ('magnitude', 'e_magnitude')}
            q.update(countrate=ct,
                     e_countrate=ct * MAG_TO_FLUX_ERR * float(
                         p['e_magnitude']), zeropoint=zp)
            if with_mag:
                q.update(magnitude=p['magnitude'],
                         e_magnitude=p['e_magnitude'])
            phot_list.append(q)
        return _event(data, name, phot_list)

    ct25, _ = load(countrate_event(25.0))
    ct31, _ = load(countrate_event(31.4))
    check(ct25 is not None, 'count-rate-only event (no instrument) loads')
    check(np.isclose(ct25.ln_likelihood(x), ct31.ln_likelihood(x),
                     rtol=1e-12),
          'likelihood does not depend on the count-rate zeropoint')
    out_c, obs_c = run(ct25, x)
    want_vega = 10.0 ** (-0.4 * (model_mag - AB_ZP_UJY))
    check(np.allclose(np.asarray(out_c['model_observations'],
                                 dtype=float)[obs_c], want_vega, rtol=1e-9),
          'band-system zeropoint gets no AB offset')

    # Magnitude + count rate: magnitudes unless --flux.
    both = countrate_event(25.0, with_mag=True)
    bm, _ = load(both)
    ob, oo = run(bm, x)
    check(np.all(np.asarray(ob['observation_types'])[oo] == 'magnitude'),
          'magnitude + count rate is fit in magnitudes by default')
    bf, _ = load(both, prefer_fluxes=True)
    ob, oo = run(bf, x)
    check(np.all(np.asarray(ob['observation_types'])[oo] == 'magcount'),
          'magnitude + count rate is fit in flux with --flux')

    # A count rate with no zeropoint is not an optical flux.
    check(not _is_band_flux({'countrate': '1', 'band': 'V'}),
          'count rate without zeropoint is not a band flux')

    # The example events in examples/ load, are fit in flux, and the Jy and
    # count-rate versions of the same light curve agree.
    lls = []
    for fn in ('ztf_flux_jy.json', 'ztf_flux_counts.json'):
        ex = json.load(open(os.path.join('examples', fn)))
        ex_name = list(ex.keys())[0]
        fitter = Fitter(test=True, quiet=True, exit_on_prompt=True,
                        prefer_cache=True)
        exm = Model(model='ia_physics', data=ex, test=True,
                    printer=fitter._printer, fitter=fitter, pool=pool)
        ok = exm.load_data(ex, event_name=ex_name, pool=pool)
        check(bool(ok), 'examples/{} loads'.format(fn))
        if not ok:
            continue
        xe = np.full(exm._num_free_parameters, 0.5)
        oe = exm.run_stack(xe, root='objective')
        oo = np.asarray(oe['observed'], dtype=bool)
        check(np.all(np.asarray(oe['observation_types'])[oo] == 'magcount')
              and int(np.sum(oo)) == len(ex[ex_name]['photometry']),
              'examples/{}: every point is fit in flux'.format(fn))
        lls.append(exm.ln_likelihood(xe))
    check(len(lls) == 2 and np.isclose(lls[0], lls[1], rtol=1e-3),
          'Jy and count-rate examples give the same likelihood')

    bad = flux_density_event('erg', 1.0)
    try:
        load(bad)
        check(False, 'unknown u_fluxdensity raises')
    except ValueError:
        check(True, 'unknown u_fluxdensity raises')

    if failures:
        print('{} check(s) failed'.format(len(failures)), file=sys.stderr)
        sys.exit(1)
    print('all band-flux checks passed')


if __name__ == '__main__':
    main()
