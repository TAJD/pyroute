"""Utilities used in Canoe voyaging simulations.

Thomas Dickson
thomas.dickson@soton.ac.uk
15/05/2018
"""
import numpy as np
from datetime import timedelta
from context import sail_route
from sail_route.weather.weather_assistance import plot_wind_data, generate_gif
from sail_route.performance.craft_performance import polar


def datetime_range(start, end, delta):
    """Generate range of dates."""
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


def plot_wind():
    """Plot representative wind scenario."""
    path = "/home/thomas/Documents/pyroute/analysis/poly_data/data_dir/"
    plot_wind_data(path, path+"/wind_forecast.nc")
    generate_gif(path, "Polynesian_July")


def load_tongiaki_perf():
    """Load predicted Tongiaki voyaging canoe performance."""
    pyroute_path = "/home/thomas/Documents/pyroute/"
    perf = np.genfromtxt(pyroute_path+"/analysis/poly_data/data_dir/tongiaki_vpp.csv",
                  delimiter=",")
    tws = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
    twa = np.array([0, 60, 70, 80, 90, 100, 110, 120])
    return polar(twa, tws, perf, 0.0, 0.0)