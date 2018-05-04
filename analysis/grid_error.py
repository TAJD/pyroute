"""Calculating grid convergence error.

Based on: Procedure for Estimation and Reporting of Uncertainty Due to Discretization
in CFD Applications

Thomas Dickson
thomas.dickson@soton.ac.uk
04/05/2018
"""

import numpy as np


def calc_h(N, Delta_A):
    """Calculate h."""
    area = np.array([Delta_A for i in range(N)])
    return np.sqrt((1/N * np.sum(area)))
