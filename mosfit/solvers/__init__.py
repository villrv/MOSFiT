"""Light-curve solvers: physics that turns ejecta + heating into a bolometric light curve.

Solvers are plain libraries, not MOSFiT modules: engine modules import them, map fit parameters onto their inputs
and cache anything that depends only on shapes.
"""
