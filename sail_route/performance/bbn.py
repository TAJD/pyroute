"""Modelling craft failure using Bayesian Belief Network.

Thomas Dickson
thomas.dickson@soton.ac.uk
22/05/2018
"""

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def wind_speed(tws):
    """Wind speed failure function."""
    if tws > 25:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wind_dir(twa):
    """Wind direction failure function."""
    if twa < 0.0:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wave_height(h):
    """Wave height failure function."""
    if h > 3:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wave_dir(theta):
    """Wave direction failure function."""
    if theta < 60.0:
        return 1
    else:
        return 0


@jit(nogil=True, fastmath=True)
def gen_env_model():
    """Specify BBN."""
    cpd_tws = TabularCPD('TWS', 2, values=[[0.8, 0.2]])
    cpd_twa = TabularCPD('TWA', 2, values=[[0.8, 0.2]])
    cpd_wind = TabularCPD('Wind', 2,
                          values=[[1, 0.1, 0.1, 0.0],
                                  [0.0, 0.9, 0.9, 1.0]],
                           # values = [[1, 0.999, 0.999, 0.998],
                           #           [0.0, 0.001, 0.001, 0.002]], # min
                          evidence=['TWA', 'TWS'],
                          evidence_card=[2, 2])
    cpd_wh = TabularCPD('WH', 2, values=[[0.8, 0.2]])
    cpd_wd = TabularCPD('WD', 2, values=[[0.8, 0.2]])
    cpd_waves = TabularCPD('Waves', 2,
                           values=[[1, 0.1, 0.1, 0.0],  # normal vals
                                   [0.0, 0.9, 0.9, 1.0]],
                           # values = [[1, 0.999, 0.999, 0.998],
                           #           [0.0, 0.001, 0.001, 0.002]], # min failure
                           evidence=['WH', 'WD'],
                           evidence_card=[2, 2])
    cpd_fail = TabularCPD('Craft failure', 2,
                          values=[[1.0, 0.1, 0.1, 0.0],
                                  [0.0, 0.9, 0.9, 1.0]],
                          evidence=['Waves', 'Wind'],
                          evidence_card=[2, 2])
    model = BayesianModel([('TWS', 'Wind'), ('TWA', 'Wind'),
                           ('WH', 'Waves'), ('WD', 'Waves'),
                           ('Waves', 'Craft failure'),
                           ('Wind', 'Craft failure')])
    model.add_cpds(cpd_tws, cpd_twa, cpd_wind,
                   cpd_wh, cpd_wd, cpd_waves, cpd_fail)
    belief_propagation = BeliefPropagation(model)
    return belief_propagation


@jit(cache=True, nogil=True, fastmath=True)
def env_bbn_interrogate(bp, tws, twa, h, theta):
    """
    Interrogate BBN for failure probability.

    Modelling failure as a function of environmental conditions.
    """
    q = bp.query(variables=['Craft failure'],
                 evidence={'TWS': wind_speed(tws),
                           'TWA': wind_dir(twa),
                           'WH': wave_height(h),
                           'WD': wave_dir(theta)})
    return q['Craft failure'].values[-1]


if __name__ == '__main__':
    model = gen_env_model()
    print("No failure: ", env_bbn_interrogate(model, 10, 60, 0, 40))
    print("Wave direction condition: ", env_bbn_interrogate(model, 10, 60, 0, 10))
    print("Full wave failure: ", env_bbn_interrogate(model, 10, 60, 4, 10))
    print("Wind speed failure: ", env_bbn_interrogate(model, 40, 60, 4, 10))
    print("Wind cond failure: ", env_bbn_interrogate(model, 40, 10, 4, 10))
