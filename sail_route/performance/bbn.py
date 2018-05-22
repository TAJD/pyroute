"""Modelling craft failure using Bayesian Belief Network.

Thomas Dickson
thomas.dickson@soton.ac.uk
22/05/2018
"""

from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, \
                       State, BayesianNetwork
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit


plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = """\\usepackage{subdepth},
                                         \\usepackage{type1cm}"""


@jit(nopython=True, cache=True)
def wind_speed(tws):
    if tws > 20.0:
        return 1
    else:
        return 0


@jit(nopython=True, cache=True)
def wind_dir(twa):
    if twa < 30.0:
        return 1
    else:
        return 0


@jit(nopython=True, cache=True)
def wave_height(h):
    if h > 2.0:
        return 1
    else:
        return 0


@jit(nopython=True, cache=True)
def wave_dir(theta):
    if theta < 30.0:
        return 1
    else:
        return 0


@jit(cache=True)
def setup_model():
    wind_speed = DiscreteDistribution({1: 0.8, 0: 0.2})
    wind_dir = DiscreteDistribution({1: 0.8, 0: 0.2})
    wind = ConditionalProbabilityTable([[1, 1, 0, 0.01],
                                        [1, 1, 1, 0.99],
                                        [1, 0, 0, 0.2],
                                        [1, 0, 1, 0.8],
                                        [0, 1, 0, 0.2],
                                        [0, 1, 1, 0.8],
                                        [0, 0, 0, 0.99],
                                        [0, 0, 1, 0.01]],
                                       [wind_speed, wind_dir])
    wave_height = DiscreteDistribution({1: 0.8, 0: 0.2})
    wave_dir = DiscreteDistribution({1: 0.8, 0: 0.2})
    wave = ConditionalProbabilityTable([[1, 1, 0, 0.01],
                                        [1, 1, 1, 0.99],
                                        [1, 0, 0, 0.2],
                                        [1, 0, 1, 0.8],
                                        [0, 1, 0, 0.2],
                                        [0, 1, 1, 0.8],
                                        [0, 0, 0, 0.99],
                                        [0, 0, 1, 0.01]],
                                       [wave_height, wave_dir])
    failure = ConditionalProbabilityTable([[1, 1, 1, 1.0],
                                           [1, 1, 0, 0.0],
                                           [1, 0, 1, 1.0],
                                           [1, 0, 0, 0.0],
                                           [0, 1, 1, 1.0],
                                           [0, 1, 0, 0.0],
                                           [0, 0, 1, 0.0],
                                           [0, 0, 0, 0.0]],
                                          [wave, wind])
    s1 = State(wind_speed, name="TWS")
    s2 = State(wind_dir, name="TWA")
    s3 = State(wind, name="Wind conditions")
    s4 = State(wave_height, name="WH")
    s5 = State(wave_dir, name="WD")
    s6 = State(wave, name="Wave conditions")
    s7 = State(failure, name="Craft failure")
    model = BayesianNetwork("Sailing craft failure")
    model.add_states(s1, s2, s3, s4, s5, s6, s7)
    model.add_transition(s1, s3)
    model.add_transition(s2, s3)
    model.add_transition(s4, s6)
    model.add_transition(s5, s6)
    model.add_transition(s3, s7)
    model.add_transition(s6, s7)
    model.bake()
    model.predict_proba({})
    return model


@jit(cache=True)
def test_model():
    model = setup_model()
    # print(model)
    # model.plot()
    plt.figure(figsize=(14, 10))
    model.plot()
    plt.savefig("/home/thomas/Documents/pyroute/analysis/asv/"+"simp_bbn.png")
    val = model.predict_proba({'TWS': wind_speed(10.0),
                               'TWA': wind_dir(50.0),
                               'Wind conditions': 1,
                               'WH': wave_height(2.0),
                               'WD': wave_dir(20.0),
                               'Wave conditions': 1})
    np.testing.assert_almost_equal(val[-1].probability(1), 1.0)


def failure_model(tws, twa, wh, wd):
    """Failure model using bbn."""
    model = setup_model()
    result = model.predict_proba({'TWS': wind_speed(tws),
                                  'TWA': wind_dir(twa),
                                  'Wind conditions': 1,
                                  'WH': wave_height(wh),
                                  'WD': wave_dir(wd),
                                  'Wave conditions': 1})
    return result[-1].probability(1)


if __name__ == '__main__':
    test_model()
