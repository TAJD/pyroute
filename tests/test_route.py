"""
Functions testing implementation of offshore sailing craft routing algorithm.

Thomas Dickson
thomas.dickson@soton.ac.uk
24/04/18

"""

from context import *
from sail_route.route.grid_locations import return_co_ords
import pytest
import numpy as np
import numpy.testing as np


def test_func_fast():
    pass


@pytest.mark.slow
def test_func_slow():
    pass

if __name__ == '__main__':
    print("run")
    start_long = -14.0
    finish_long = -6.0
    start_lat = 47.0
    finish_lat = 47.0
    longs, lats, land = return_co_ords(start_long, finish_long,
                                       start_lat, finish_lat)
