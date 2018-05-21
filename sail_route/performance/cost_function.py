""" Cost function for routing.

Describing the cost function which calculates the earliest time to be taken
between two points in the offshore sailing craft domain.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""

import numpy as np
import datetime
from numba import jit, njit


@njit(fastmath=True, nogil=True)
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points.

    Return the value in km."""
    lon1, lat1, lon2, lat2 = np.radians(np.array([lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin((dlat)/2)**2 + np.cos(lat1) * np.cos(lat2) * \
        np.sin((dlon)/2)**2
    dist = 6371 * 2 * np.arcsin(np.sqrt(a)) * 0.5399565
    bearing = np.rad2deg(np.arctan2(dlat, dlon))
    return dist, bearing


@njit(fastmath=True, nogil=True)
def wind_strength_step(x):
    """Wind strength failure function."""
    return 1 * (x > 20.0)


@jit(cache=True)
def craft_failure_model(time, tws, twa):
    """Return probability of failure as a function of time and environment."""
    time = time/datetime.timedelta(hours=1)
    pf_wind = wind_strength_step(tws)
    pf_time = np.exp(0.001*time) - 1
    return np.min((np.max((pf_time, pf_wind)), 1.0))


@njit(fastmath=True, nogil=True)
def dir_to_relative(x, y):
    """Calculate relative angle to bearing."""
    return np.absolute((x - y + 180) % 360 - 180)


@jit(fastmath=True, nogil=True, cache=True)
def failure_criteria(craft, time, speed, tws, twa, wd, wh, wp):
    """Craft failure model. Returns an array of booleans."""
    # if (wd < 30.0):
    #     return True
    # elif (wh > 5.0):
    #     return True
    # elif (speed < 0.3):
    #     return True
    # if (twa < 30):
    #     return True
    # else:
    return False


@jit(cache=True, nogil=True)
def cost_function(x1, y1, x2, y2, tws, twd, i_wd, i_wh, i_wp,
                  craft, lifetime=None):
    """Calculate the time taken to transit between two locations."""
    dist, bearing = haversine(x1, y1, x2, y2)
    twa = dir_to_relative(bearing, twd)
    wave_dir = dir_to_relative(bearing, i_wd)
    speed = craft.return_perf(np.abs(twa), tws)
    # fc = failure_criteria(craft, lifetime, speed, tws, twa,
    #                       wave_dir, i_wh, i_wp)
    # if fc is True:
    #     return np.inf
    # else:
    if speed < 0.3:
        return np.inf
    else:
        return datetime.timedelta(hours=np.float64(dist/speed))
