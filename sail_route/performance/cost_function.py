""" Cost function for routing.

Describing the cost function which calculates the earliest time to be taken
between two points in the offshore sailing craft domain.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""

import numpy as np
import datetime
from numba import njit, jit


@jit
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km * 0.5399565


@jit
def craft_failure_model(time, tws, twd):
    """Return probability of failure as a function of time and environment."""
    time = time/datetime.timedelta(hours=1)
    return 1 - np.exp(0.001*time)


@jit
def cost_function(x1, y1, x2, y2, tws, twd, craft, lifetime=None):
    """Calculate the time taken to transit between two locations."""
    dist = haversine(x1, y1, x2, y2) + np.random.rand(1)
    # convert from twd to twa
    dlon = x2 - x1
    dlat = y2 - y1
    bearing = 360.0 - (np.rad2deg(np.arctan2(dlon, dlat)) + 180.0)
    twa = bearing - twd
    twa = (twa + 180) % 360 - 180
    speed = craft.return_perf(np.abs(twa), tws)
    if lifetime is not None:
        pf = craft_failure_model(lifetime, tws, twa)
    else:
        pf = 0.0
    if speed == 0.0:
        return datetime.timedelta(hours=10), pf
    else:
        return datetime.timedelta(hours=np.float64(dist/speed)), pf
