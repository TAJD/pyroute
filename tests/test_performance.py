"""
Functions testing implementation of performance functionsself.

Thomas Dickson
thomas.dickson@soton.ac.uk
"""
from context import *
from sail_route.performance.cost_function import haversine, dir_to_relative
from sail_route.performance.craft_performance import polar
# import pytest
import numpy as np
import numpy.testing as npt

pp = "/Users/thomasdickson/Documents/python_routing/"

def test_haversine():
    """
    Test haversine formula.

    Sources used:
    https://www.fcc.gov/media/radio/distance-and-azimuths
    https://rosettacode.org/wiki/Haversine_formula#Python
    """
    dist, bearing = haversine(-88.67, 36.12, -118.40, 33.94)
    npt.assert_allclose(dist, 1462.22, rtol=0.01)
    npt.assert_allclose(bearing, 276.33, rtol=0.01)


def test_dir_to_relative():
    """Test relative angle calculation function."""
    npt.assert_almost_equal(dir_to_relative(-20, 20), 40)
    npt.assert_almost_equal(dir_to_relative(20, -20), 40)
    npt.assert_almost_equal(dir_to_relative(40, 60), 20)
    npt.assert_almost_equal(dir_to_relative(-20, -40), 20)


def test_performance_interpolation():
    """Test the interpolation of performance polars."""
    path = pp + "tests/test_data/first_40_farr.csv"
    perf = np.genfromtxt(path, delimiter=";", skip_header=1)
    twa = np.array([30.0, 36.0, 42.0, 50.0, 70.0, 90.0,
                    120.0, 130.0, 150.0, 160.0, 180.0])
    tws = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                    16.0, 20.0, 25.0, 30.0, 35.0])
    first_40 = polar(twa, tws, perf[:, 1:])
    npt.assert_almost_equal(first_40.return_perf(30.0, 4.0), 2.16)
