"""Definitions for the `IaPhysicsConstraints` class."""
from mosfit.modules.constraints.constraint import Constraint
from mosfit.physics_lc import kinetic_energy


# Important: Only define one ``Module`` class per file.


class IaPhysicsConstraints(Constraint):
    """Constraints for the `ia_physics` models (energy budget of `physicsni`).

    1. The burned masses fit inside the ejecta: M_IME = M_ej - M_Ni - M_Fe - M_CO >= 0
       (kinetic_energy returns -1 otherwise).
    2. The explosion unbinds the white dwarf: E_k = E_nuc - E_bind > 0.
    3. Nickel below 95% of the ejecta (the engine's limit).

    Hard constraints: a violation sets the score to -inf, which the likelihood maps to its floor.
    `physicsni` keeps the same checks so it never builds a light curve from impossible masses.
    """

    _REFERENCES = []

    def process(self, **kwargs):
        """Process module."""
        m_ej = kwargs[self.key('mejecta')]
        m_ni = kwargs[self.key('fnickel')] * m_ej
        m_fe = kwargs.get(self.key('mfe'), 0.1)
        ok = kinetic_energy(m_ej, m_ni, M_fe=m_fe) > 0 and m_ni < 0.95 * m_ej
        return {self.key('score_modifier'): 0.0 if ok else -float('inf')}
