"""Modelling craft failure using Bayesian Belief Network.

Thomas Dickson
thomas.dickson@soton.ac.uk
22/05/2018
"""

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from numba import jit
import numpy.testing as npt


@jit(fastmath=True, nopython=True, cache=True)
def wind_speed(tws):
    if tws > 20.0:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wind_dir(twa):
    if twa < 30.0:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wave_height(h):
    if h > 2.0:
        return 1
    else:
        return 0


@jit(fastmath=True, nopython=True, cache=True)
def wave_dir(theta):
    if theta < 30.0:
        return 1
    else:
        return 0


@jit(cache=True, nogil=True, fastmath=True)
def gen_env_model():
    cpd_tws = TabularCPD('TWS', 2, values=[[0.8, 0.2]])
    cpd_twa = TabularCPD('TWA', 2, values=[[0.8, 0.2]])
    cpd_wind = TabularCPD('Wind', 2,
                          values=[[1, 0.5, 0.5, 0.0],
                                  [0.0, 0.5, 0.5, 1.0]],
                          evidence=['TWA', 'TWS'],
                          evidence_card=[2, 2])
    cpd_wh = TabularCPD('WH', 2, values=[[0.8, 0.2]])
    cpd_wd = TabularCPD('WD', 2, values=[[0.8, 0.2]])
    cpd_waves = TabularCPD('Waves', 2,
                           values=[[1, 0.2, 0.2, 0.0],
                                   [0.0, 0.8, 0.8, 1.0]],
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
def env_bbn_interrogate(craft, tws, twa, h, theta):
    bp = craft.failure
    """Modelling failure as a function of environmental conditions."""
    q = bp.query(variables=['Craft failure'],
                 evidence={'TWS': wind_speed(tws),
                           'TWA': wind_dir(twa),
                           'WH': wave_height(h),
                           'WD': wave_dir(theta)})
    return q['Craft failure'].values[-1]
