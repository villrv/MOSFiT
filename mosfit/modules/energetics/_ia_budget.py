"""Thermonuclear (Type Ia) energy budget: E_k = E_nuc - E_bind, used by the `ia_energetics` module.

Plain functions (the leading underscore keeps MOSFiT's module loader from registering them as modules).
"""
import numpy as np


# --- Energy conservation for thermonuclear ejecta (Maeda & Iwamoto 2009; as used by Scalzo et al. 2014a)
E_NI, E_FE, E_IME = 1.56e51, 1.74e51, 1.24e51   # erg per Msun burned from C/O

# Cold C/O white-dwarf binding energies from csp/wd_binding_table.py (sn-modal-lc): Chandrasekhar's degenerate-electron
# equation of state, mu_e = 2, hydrostatic equilibrium, E_bind = |W| - U; no Coulomb or thermal corrections. Agrees with
# Yoon & Langer (2005) to 2-5% over their fitted range (1.16-1.41 Msun). Rows: log10 rho_c = 5.5 ... 10.5 g/cm^3.
_WD_M = np.array([
    0.25010, 0.27611, 0.30417, 0.33431, 0.36651, 0.40072, 0.43684, 0.47475, 0.51428, 0.55520,
    0.59728, 0.64025, 0.68381, 0.72765, 0.77147, 0.81495, 0.85781, 0.89978, 0.94059, 0.98005,
    1.01796, 1.05418, 1.08858, 1.12109, 1.15165, 1.18024, 1.20686, 1.23153, 1.25431, 1.27525,
    1.29442, 1.31191, 1.32781, 1.34221, 1.35521, 1.36692, 1.37742, 1.38681, 1.39519, 1.40265,
    1.40927, 1.41513, 1.42031, 1.42487, 1.42889, 1.43242, 1.43551, 1.43821, 1.44058, 1.44264,
    1.44443])                                         # Msun
_WD_EB = np.array([
    0.00518, 0.00655, 0.00825, 0.01035, 0.01290, 0.01600, 0.01973, 0.02416, 0.02939, 0.03550,
    0.04257, 0.05067, 0.05985, 0.07014, 0.08157, 0.09414, 0.10781, 0.12255, 0.13830, 0.15496,
    0.17244, 0.19063, 0.20942, 0.22867, 0.24827, 0.26808, 0.28798, 0.30785, 0.32759, 0.34709,
    0.36626, 0.38503, 0.40331, 0.42105, 0.43820, 0.45472, 0.47058, 0.48576, 0.50025, 0.51404,
    0.52713, 0.53954, 0.55126, 0.56232, 0.57273, 0.58251, 0.59170, 0.60030, 0.60836, 0.61589,
    0.62292])                                         # 1e51 erg

# Near and above the Chandrasekhar mass: Yoon & Langer (2005) Eqs. 22, 32-34 for rotating white dwarfs at
# rho_c = 3e9 g/cm^3, as in Maeda & Iwamoto (2009) and Scalzo et al. (2010); used beyond their fitted M < 2.0 Msun as
# those papers do. In 1e51 erg.
_YL_X = np.log10(3e9)
_YL_MNR = 1.436 * (1 - np.exp(-0.01316 * _YL_X ** 2.706 + 0.2493 * _YL_X))     # non-rotating mass, 1.389 Msun
_YL_BENR = (-32.759747 + 6.7179802 * _YL_X - 0.28717609 * _YL_X ** 2) / 10
_YL_C5 = (-370.73052 + 132.97204 * _YL_X - 16.117031 * _YL_X ** 2 + 0.66986678 * _YL_X ** 3) / 10


def _yl_rot(M):
    return _YL_BENR + _YL_C5 * np.clip(M - _YL_MNR, 0, None) ** 1.03


_m = np.linspace(_YL_MNR, _WD_M[-1], 2001)
_M_JOIN = _m[np.argmax(np.interp(_m, _WD_M, _WD_EB) >= _yl_rot(_m))]     # where the two curves cross, ~1.41 Msun


def wd_binding(M):
    """C/O white-dwarf binding energy (erg) at mass M (Msun): the cold white-dwarf structure table above up to _M_JOIN
    (~1.41 Msun), then Yoon & Langer's rotating fit at rho_c = 3e9 g/cm^3, which is continuous with it there."""
    M = np.asarray(M, dtype=float)
    eb = np.where(M <= _M_JOIN, np.interp(M, _WD_M, _WD_EB), _yl_rot(M)) * 1e51
    return float(eb) if eb.ndim == 0 else eb

def kinetic_energy(M_ej, M_ni, M_fe=0.1, M_co=0.02):
    """E_k = E_nuc - E_bind, with the remainder burned to IMEs."""
    M_ime = M_ej - M_ni - M_fe - M_co
    if M_ime < 0:
        return -1.0
    return E_NI * M_ni + E_FE * M_fe + E_IME * M_ime - wd_binding(M_ej)
