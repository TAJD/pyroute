"""Basic simulations of Finney voyages.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""

from context import sail_route
import numpy as np
from datetime import datetime, timedelta
from sail_route.time_func import timefunc, do_cprofile
from sail_route.weather.weather_assistance import download_wind, plot_wind_data, generate_gif
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route, \
                                   plot_reliability_route
from sail_route.performance.craft_performance import polar
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords


def datetime_range(start, end, delta):
    """Generate range of dates."""
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


def download_wind_poly():
    """Download representative wind scenario."""
    path = "/Users/thomasdickson/Documents/python_routing/analysis/poly_data"
    download_wind(path, -10, -150.0, -18.0, -135.0)


def plot_wind():
    """Plot representative wind scenario."""
    path = "/Users/thomasdickson/Documents/python_routing/analysis/poly_data/data_dir/"
    plot_wind_data(path, path+"/wind_forecast.nc")
    generate_gif(path, "Polynesian_July")


def load_tongiaki_perf():
    """Load predicted Tongiaki voyaging canoe performance."""
    perf = np.genfromtxt("/Users/thomasdickson/Documents/python_routing/analysis/poly_data/data_dir/tongiaki_vpp.csv", delimiter=",")
    tws = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
    twa = np.array([60, 70, 80, 90, 100, 110, 120])
    return polar(twa, tws, perf, 0.0)


def run_simulation():
    tahiti = Location(-149.426, -17.651)
    marquesas = Location(-139.33, -9)
    craft = load_tongiaki_perf()
    no_nodes = 35
    dist, bearing = haversine(tahiti.long, tahiti.lat, marquesas.long,
                               marquesas.lat)
    node_distance = (dist/0.5399565)/no_nodes
    r = Route(tahiti, marquesas, no_nodes, no_nodes,
              node_distance*1000.0, craft)
    wind_fname = "/Users/thomasdickson/Documents/python_routing/analysis/poly_data/data_dir/wind_forecast.nc"
    diagram_path = "/Users/thomasdickson/Documents/python_routing/analysis/poly_data"
    sd = datetime(2014, 7, 1, 0, 0)
    ed = datetime(2014, 7, 2, 0, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    for t in dt:
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        jt, et, pf_vals = min_time_calculate(r, wind_fname, t, craft)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        plot_mt_route(t, r, x, y, et, jt, diagram_path+"/"+str(t))
        plot_reliability_route(t, r, x, y, pf_vals, jt, diagram_path+"/"+str(t))

if __name__ == '__main__':
    # download_wind_poly()
    # plot_wind()
    run_simulation()
