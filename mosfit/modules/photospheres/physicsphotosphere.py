"""Photosphere from the physics model's tau = 2/3 radius, with a temperature floor
(once T would drop below the floor, the radius recedes so that T = floor; as temperature_floor)."""
import numpy as np
from astropy import constants as c
from mosfit.constants import FOUR_PI
from mosfit.modules.photospheres.photosphere import Photosphere


# Important: Only define one ``Module`` class per file.


class PhysicsPhotosphere(Photosphere):
    """Physics-model photosphere with a minimum temperature."""

    STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        kwargs = self.prepare_input(self.key('physradius'), **kwargs)
        lum = np.asarray(kwargs[self.key('luminosities')], dtype=float)
        rad = np.asarray(kwargs[self.key('physradius')], dtype=float)
        t_floor = kwargs[self.key('temperature')]
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = (lum / (self.STEF_CONST * rad ** 2)) ** 0.25
        low = ~(temp > t_floor)
        temp[low] = t_floor
        rad = rad.copy()
        rad[low] = np.sqrt(lum[low] / (self.STEF_CONST * t_floor ** 4))
        temp[lum <= 0] = 0.0
        rad[lum <= 0] = 0.0
        return {self.key('radiusphot'): rad, self.key('temperaturephot'): temp}
