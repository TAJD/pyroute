"""Basic simulations of Finney voyages.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""

from context import sail_route
import numpy as np
from datetime import datetime, timedelta
from sail_route.weather.weather_assistance import plot_wind_data, \
                                                  generate_gif, \
                                                  return_domain
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route, \
                                   plot_reliability_route
from sail_route.performance.craft_performance import polar
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords
from sail_route.route.solve_route import shortest_path
from grid_error import calc_h


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
    path = pyroute_path + "analysis/poly_data"
    perf = np.genfromtxt(pyroute_path+"/analysis/poly_data/data_dir/tongiaki_vpp.csv", delimiter=",")
    tws = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
    twa = np.array([60, 70, 80, 90, 100, 110, 120])
    return polar(twa, tws, perf, 0.0)


def run_simulation_over_days():
    tahiti = Location(-149.426, -17.651)
    marquesas = Location(-139.33, -9)
    craft = load_tongiaki_perf()
    no_nodes = 3
    dist, bearing = haversine(tahiti.long, tahiti.lat, marquesas.long,
                              marquesas.lat)
    node_distance = (dist/0.5399565)/no_nodes
    r = Route(tahiti, marquesas, no_nodes, no_nodes,
              node_distance*1000.0, craft)
    pyroute_path = "/home/thomas/Documents/pyroute/"
    wind_fname = pyroute_path + "analysis/poly_data/data_dir/wind_forecast.nc"
    waves_fname = pyroute_path + "analysis/poly_data/data_dir/wave_data.nc"
    diagram_path = pyroute_path + "analysis/poly_data"
    sd = datetime(2000, 7, 1, 0, 0)
    ed = datetime(2000, 7, 2, 0, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    for t in dt:
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = return_domain(wind_fname, waves_fname)
        jt, et, pf_vals = min_time_calculate(r, t, craft,
                           x, y, land,
                           tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        plot_mt_route(t, r, x, y, et, jt, diagram_path+"/"+str(t))
        plot_reliability_route(t, r, x, y, pf_vals, jt, diagram_path+"/"+str(t))


def grid_error():
    """Perform grid error study for routing given real weather.

    Return a plot of the difference between results as a function
    of grid error."""
    tahiti = Location(-149.426, -17.651)
    marquesas = Location(-139.33, -9)
    craft = load_tongiaki_perf()
    pyroute_path = "/home/thomas/Documents/pyroute/"
    wind_fname = pyroute_path + "analysis/poly_data/data_dir/wind_forecast.nc"
    waves_fname = pyroute_path + "analysis/poly_data/data_dir/wave_data.nc"
    diagram_path = pyroute_path + "analysis/poly_data"
    sd = datetime(2014, 7, 1, 0, 0)
    dist, bearing = haversine(tahiti.long, tahiti.lat, marquesas.long,
                              marquesas.lat)
    tws, twd, wd, wh, wp = return_domain(wind_fname, waves_fname)
    nodes = np.array([i**2 for i in range(7, 9)])
    times = []
    h_vals = []
    for count, node in enumerate(nodes):
        node_distance = (dist/0.5399565)/node
        r = Route(tahiti, marquesas, node, node,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        jt, et, pf_vals = min_time_calculate(r, sd, craft,
                                             x, y, land,
                                             tws, twd, wd, wh, wp)
        h_vals.append(calc_h(node, node_distance**2))
        vt = datetime.fromtimestamp(jt) - sd
        print(h_vals[-1], "  ", vt)
        times.append(vt.total_seconds())
    h_vals, times = np.array(h_vals), np.array(times)
    with open(diagram_path+"grid_output_large.txt", 'wb') as f:
        np.savetxt(f, np.c_[h_vals, times], delimiter='\t')


if __name__ == '__main__':
    run_simulation_over_days()
    # grid_error()
