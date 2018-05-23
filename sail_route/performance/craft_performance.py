""" Craft performanceself.

Thomas Dickson
thomas.dickson@soton.ac.uk
24/04/2018
"""
import numpy as np
from scipy.interpolate import interp2d
from numba import jit


def get_perf():
    path = "/Users/thomasdickson/Documents/sail_routing/routing/data/first_40_farr.csv"
    perf = np.genfromtxt(path, delimiter=";", skip_header=1)
    return perf[:, 1:]


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
