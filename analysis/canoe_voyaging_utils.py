"""Utilities used in Canoe voyaging simulations.

Thomas Dickson
thomas.dickson@soton.ac.uk
15/05/2018
"""
import numpy as np
from datetime import timedelta
from context import sail_route
from sail_route.performance.craft_performance import polar

pyroute_path = "/mainfs/home/td7g11/pyroute"


def datetime_range(start, end, delta):
    """Generate range of dates."""
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


def tong_uncertain(unc, apf, fm):
    """Load predicted Tongiaki performance."""
    perf = np.genfromtxt(pyroute_path+"/analysis/poly_data/data_dir/tongiaki_vpp.csv",
                  delimiter=",")
    tws = np.array([0, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
    twa = np.array([0, 60, 70, 80, 90, 100, 110, 120])
    return polar(twa, tws, perf, unc, apf, fm)
