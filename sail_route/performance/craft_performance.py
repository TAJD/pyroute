""" Craft performanceself.

Thomas Dickson
thomas.dickson@soton.ac.uk
24/04/2018
"""
import numpy as np
from scipy.interpolate import interp2d
from numba import jit, jitclass, float64


def get_perf():
    path = "/Users/thomasdickson/Documents/sail_routing/routing/data/first_40_farr.csv"
    perf = np.genfromtxt(path, delimiter=";", skip_header=1)
    return perf[:, 1:]

# spec = [
#     ('tws_range', float64[:]),
#     ('twa_range', float64[:]),
#     ('perf', float64[:]),
#     ('unc', float64),
#     ('tws', float64),
#     ('twa', float64),
# ]
#
# @jitclass(spec)
class polar(object):
    """Store and return information on sailing craft polars."""

    def __init__(self, tws_range, twa_range, perf, unc=0.0):
        """Initialise sailing craft performance data."""
        self.twa_range = twa_range
        self.tws_range = tws_range
        self.perf = perf
        self.unc = unc

    @jit(cache=True)
    def return_perf(self, tws, twa):
        """Return sailing craft performance."""
        p = interp2d(self.twa_range, self.tws_range, self.perf,
                     kind='linear', bounds_error=False, fill_value=0.0)
        return p(twa, tws)


def return_boat_perf():
    perf = get_perf()
    twa = np.array([30.0, 36.0, 42.0, 50.0, 70.0, 90.0,
                    120.0, 130.0, 150.0, 160.0, 180.0])
    tws = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                    16.0, 20.0, 25.0, 30.0, 35.0])
    first_40 = polar(twa, tws, perf)
    return first_40


if __name__ == '__main__':
    boat = return_boat_perf()
    print(boat.return_perf(30.0, 4.0))
