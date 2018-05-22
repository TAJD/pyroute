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
# from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, \
#                        State, BayesianNetwork


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

#
# @jit(nopython=True, cache=True)
# def wind_speed(tws):
#     if tws > 20.0:
#         return 1
#     else:
#         return 0
#
#
# @jit(nopython=True, cache=True)
# def wind_dir(twa):
#     if twa < 30.0:
#         return 1
#     else:
#         return 0
#
#
# @jit(nopython=True, cache=True)
# def wave_height(h):
#     if h > 2.0:
#         return 1
#     else:
#         return 0
#
#
# @jit(nopython=True, cache=True)
# def wave_dir(theta):
#     if theta < 30.0:
#         return 1
#     else:
#         return 0
#
#
# @jit(cache=True)
# def setup_model():
#     wind_speed = DiscreteDistribution({1: 0.8, 0: 0.2})
#     wind_dir = DiscreteDistribution({1: 0.8, 0: 0.2})
#     wind = ConditionalProbabilityTable([[1, 1, 0, 0.01],
#                                         [1, 1, 1, 0.99],
#                                         [1, 0, 0, 0.2],
#                                         [1, 0, 1, 0.8],
#                                         [0, 1, 0, 0.2],
#                                         [0, 1, 1, 0.8],
#                                         [0, 0, 0, 0.99],
#                                         [0, 0, 1, 0.01]],
#                                        [wind_speed, wind_dir])
#     wave_height = DiscreteDistribution({1: 0.8, 0: 0.2})
#     wave_dir = DiscreteDistribution({1: 0.8, 0: 0.2})
#     wave = ConditionalProbabilityTable([[1, 1, 0, 0.01],
#                                         [1, 1, 1, 0.99],
#                                         [1, 0, 0, 0.2],
#                                         [1, 0, 1, 0.8],
#                                         [0, 1, 0, 0.2],
#                                         [0, 1, 1, 0.8],
#                                         [0, 0, 0, 0.99],
#                                         [0, 0, 1, 0.01]],
#                                        [wave_height, wave_dir])
#     failure = ConditionalProbabilityTable([[1, 1, 1, 1.0],
#                                            [1, 1, 0, 0.0],
#                                            [1, 0, 1, 1.0],
#                                            [1, 0, 0, 0.0],
#                                            [0, 1, 1, 1.0],
#                                            [0, 1, 0, 0.0],
#                                            [0, 0, 1, 0.0],
#                                            [0, 0, 0, 0.0]],
#                                           [wave, wind])
#     s1 = State(wind_speed, name="TWS")
#     s2 = State(wind_dir, name="TWA")
#     s3 = State(wind, name="Wind conditions")
#     s4 = State(wave_height, name="WH")
#     s5 = State(wave_dir, name="WD")
#     s6 = State(wave, name="Wave conditions")
#     s7 = State(failure, name="Craft failure")
#     model = BayesianNetwork("Sailing craft failure")
#     model.add_states(s1, s2, s3, s4, s5, s6, s7)
#     model.add_transition(s1, s3)
#     model.add_transition(s2, s3)
#     model.add_transition(s4, s6)
#     model.add_transition(s5, s6)
#     model.add_transition(s3, s7)
#     model.add_transition(s6, s7)
#     model.bake()
#     model.predict_proba({})
#     return model
#
#
# @jit(cache=True)
# def test_model():
#     model = setup_model()
#     # print(model)
#     # model.plot()
#     plt.figure(figsize=(14, 10))
#     model.plot()
#     plt.savefig("/home/thomas/Documents/pyroute/analysis/asv/"+"simp_bbn.png")
#     val = model.predict_proba({'TWS': wind_speed(10.0),
#                                'TWA': wind_dir(50.0),
#                                'Wind conditions': 1,
#                                'WH': wave_height(2.0),
#                                'WD': wave_dir(20.0),
#                                'Wave conditions': 1})
#     np.testing.assert_almost_equal(val[-1].probability(1), 1.0)
#
#
# def failure_model(tws, twa, wh, wd):
#     """Failure model using bbn."""
#     model = setup_model()
#     result = model.predict_proba({'TWS': wind_speed(tws),
#                                   'TWA': wind_dir(twa),
#                                   'Wind conditions': 1,
#                                   'WH': wave_height(wh),
#                                   'WD': wave_dir(wd),
#                                   'Wave conditions': 1})
#     return result[-1].probability(1)


# @jit(fastmath=True, nogil=True, cache=True)
# def failure_criteria(craft, time, speed, tws, twa, wd, wh, wp):
#     """Craft failure model. Returns an array of booleans."""
#     # if (wd < 30.0):
#     #     return True
#     # elif (wh > 5.0):
#     #     return True
#     # elif (speed < 0.3):
#     #     return True
#     # if (twa < 30):
#     #     return True
#     # else:
#     return False


@jit(cache=True, nogil=True)
def cost_function(x1, y1, x2, y2, tws, twd, i_wd, i_wh, i_wp,
                  craft, lifetime=None):
    """Calculate the time taken to transit between two locations."""
    dist, bearing = haversine(x1, y1, x2, y2)
    twa = dir_to_relative(bearing, twd)
    # wave_dir = dir_to_relative(bearing, i_wd)
    speed = craft.return_perf(np.abs(twa), tws)
    # fc = failure_model(tws, twa, i_wh, wave_dir)
    # if fc > craft.apf:
    #     return np.inf
    if speed < 0.3:
        return np.inf
    else:
        return datetime.timedelta(hours=np.float64(dist/speed))
