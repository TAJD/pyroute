""" Craft performance.

Thomas Dickson
thomas.dickson@soton.ac.uk
24/04/2018
"""
import numpy as np
from scipy.interpolate import interp2d
from numba import jit


class polar(object):
    """Store and return information on sailing craft polars."""

    def __init__(self, tws_range, twa_range, perf, unc=1.0, apf=1.0,
                 failure=None):
        """Initialise sailing craft performance data.

        twa_range, numpy array containing true wind angle values
        tws_range, numpy array containing true wind speed values
        perf, scalar changing performance deterministically
        unc, scalar associating uncertainty with craft
        apf, scalar between 0.0 and 1.0 returning the acceptable
        probability of failure of the craft.
        """
        self.twa_range = twa_range
        self.tws_range = tws_range
        self.perf = perf
        self.unc = unc
        self.apf = apf
        self.failure = failure

    @jit(cache=True)
    def return_perf(self, tws, twa):
        """Return sailing craft performance."""
        p = interp2d(self.twa_range, self.tws_range, self.perf,
                     kind='linear')
        return p(twa, tws)*self.unc
