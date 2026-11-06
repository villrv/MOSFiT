"""Definitions for the `IaEnergetics` class."""
from mosfit.modules.energetics._ia_budget import kinetic_energy
from mosfit.modules.energetics.energetic import Energetic


# Important: Only define one ``Module`` class per file.


class IaEnergetics(Energetic):
    """Kinetic energy of thermonuclear ejecta from energy conservation.

    E_k = E_nuc - E_bind (Maeda & Iwamoto 2009): burning to nickel, stable iron and intermediate-mass elements,
    minus the white-dwarf binding energy (cold C/O structure table; Yoon & Langer 2005 rotating fit near and above
    the Chandrasekhar mass). Returns `ekinetic` in erg, -1 for impossible masses (more burned mass than ejecta).
    """

    _REFERENCES = [
        {'bibcode': '2009MNRAS.400.1225M'},
        {'bibcode': '2005A&A...435..967Y'}
    ]

    def process(self, **kwargs):
        """Process module."""
        m_ej = kwargs[self.key('mejecta')]
        m_ni = kwargs[self.key('fnickel')] * m_ej
        m_fe = kwargs.get(self.key('mfe'), 0.1)
        return {self.key('ekinetic'): kinetic_energy(m_ej, m_ni, M_fe=m_fe)}
