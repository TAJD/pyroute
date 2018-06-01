"""ASV Utilities file

File to assist analysis of autonomous sailing craft transatlantic voyage modelling.

1. Load ASV performance data
2. Check that the weather data loads correctly.

Thomas Dickson
31/05/2018
thomas.dickson@soton.ac.uk
"""

import numpy as np
from context import sail_route
from sail_route.performance.craft_performance import polar


weather_path = "/mainfs/home/td7g11/weather_data/transat_weather/"
pyroute_path = "/mainfs/home/td7g11/pyroute/"


def asv_uncertain(unc, apf, fm):
    """Load Maribot Vane performance data."""
    perf = np.genfromtxt(pyroute_path+"...")
    tws = np.array([4, 8, 12, 16, 20])
    twa = np.array([25, 40, 55, 70, 85, 100, 115, 130, 145, 160])
    return polar(twa, tws, unc, apf, fm)
