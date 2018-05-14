"""Using developed sail routing methodology.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""
from context import sail_route
from datetime import timedelta
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num


from sail_route.sail_routing import Location, Route, min_time_calculate, \
                                   plot_route
from sail_route.performance.craft_performance import return_boat_perf
from sail_route.route.grid_locations import return_co_ords

home_path = "/home/td7g11/pyroute/"


def run_simulation():
    start = Location(-14.0, 47.0)
    finish = Location(-6.0, 47.0)
    craft = return_boat_perf()
    no_nodes = 10
    r = Route(start, finish, no_nodes, no_nodes, 30000.0, craft)
    wind_fname = "/Users/thomasdickson/Documents/sail_routing/routing/domain_application/data_dir/wind_forecast.nc"
    diagram_path = "/Users/thomasdickson/Documents/python_routing/analysis/output/multiple"
    t = datetime(2014, 7, 1, 0, 0)
    x, y, land = return_co_ords(r.start.long, r.finish.long,
                                r.start.lat, r.finish.lat,
                                r.n_ranks, r.n_width, r.d_node)
    jt, et, pf_vals = min_time_calculate(r, wind_fname, time, craft)
    vt = datetime.fromtimestamp(jt) - t
    print("Journey time is: ", vt)
    plot_route(time, r, x, y, et, jt, pf_vals, diagram_path+"/single_route.png")


def run_single(n, d=30000):
    """Run multiple simulations for example scenario."""
    start = Location(-14.0, 47.0)
    finish = Location(-6.0, 47.0)
    craft = return_boat_perf()
    no_nodes = n
    r = Route(start, finish, no_nodes, no_nodes, d, craft)
    wind_fname = "/Users/thomasdickson/Documents/sail_routing/routing/domain_application/data_dir/wind_forecast.nc"
    diagram_path = "/Users/thomasdickson/Documents/python_routing/analysis/output/multiple"
    time = datetime(2014, 7, 1, 0, 0)
    x, y, land = return_co_ords(r.start.long, r.finish.long,
                                r.start.lat, r.finish.lat,
                                r.n_ranks, r.n_width, r.d_node)
    jt, et, pf_vals = min_time_calculate(r, wind_fname, time, craft)
    plot_route(time, r, x, y, et, jt, pf_vals, diagram_path+"/single_route_%.0f_%.0f.png" % (n, d))
    return jt, et, pf_vals


def simulate_multiple(n_numbers):
    """Multiple node number and distance simulations."""
    # distance = np.linspace(10000, 50000, 3)
    t = datetime(2014, 7, 1, 0, 0)
    jt_results = []
    et_results = []
    pf_results = []
    for i in n_numbers:
        start_time = time.time()
        jt, et, pf_vals = run_single(i, d=30000)
        jt_results.append(jt)
        et_results.append(et)
        pf_results.append(pf_vals)
        print("Route simulation")
        print("No nodes: %.0f" % i)
        print("Journey time is: ", datetime.fromtimestamp(jt) - t)
        print("%s seconds to run" % (time.time() - start_time))
    return jt_results, et_results, pf_results


def plot_route_times(n_numbers, jt_results, start_time, diagram_path):
    # vt = datetime.fromtimestamp(jt) - start
    conv_st = date2num(start_time)
    conv_ft = [start_time + timedelta(seconds=i) for i in jt_results]
    conv_ft = [date2num(i) for i in conv_ft]
    conv_delta = [i - conv_st for i in conv_ft]
    #
    plt.figure()
    plt.plot(n_numbers, conv_delta)
    plt.xlabel("Number of nodes in domain")
    plt.ylabel("Routing time (hours)")
    plt.gcf().autofmt_xdate()
    plt.savefig(diagram_path+"/grid_convergence")


if __name__ == '__main__':
    # run_single(8, 30000)
    n_numbers = np.arange(5, 18)
    jt_results, et_results, pf_results = simulate_multiple(n_numbers)
    t = datetime(2014, 7, 1, 0, 0)
    diagram_path = "/Users/thomasdickson/Documents/python_routing/analysis/output/multiple"
    plot_route_times(n_numbers, jt_results, t, diagram_path)
