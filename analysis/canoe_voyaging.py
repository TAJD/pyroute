"""Basic simulations of Finney voyages.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""
from context import sail_route
import numpy as np
from datetime import datetime
from canoe_voyaging_utils import datetime_range, load_tongiaki_perf
from sail_route.weather.weather_assistance import return_domain
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords
from grid_error import calc_h


def run_simulation_over_days():
    """Run routing simulations between tahiti and marquesas."""
    tahiti = Location(-149.426, -17.651)
    marquesas = Location(-139.33, -9)
    craft = load_tongiaki_perf()
    n_nodes = 160
    n_width = n_nodes
    print("Nodes in rank: ", n_nodes)
    print("Nodes in width: ", n_width)
    dist, bearing = haversine(tahiti.long, tahiti.lat, marquesas.long,
                              marquesas.lat)
    node_distance = 1000*dist/n_width
    print("Node height distance is ", dist/n_nodes*1000, " m")
    print("Node width distance is ", node_distance, " m")
    area = node_distance*dist/n_nodes*1000
    total_area = n_nodes * n_width * area
    h = (1/(n_nodes * n_width) * total_area)**0.5
    print("h = {0} ".format(h))
    r = Route(tahiti, marquesas, n_nodes, n_width,
              node_distance, craft)
    pyroute_path = "/home/thomas/Documents/pyroute/"
    wind_fname = pyroute_path + "analysis/poly_data/data_dir/wind_forecast.nc"
    waves_fname = pyroute_path + "analysis/poly_data/data_dir/wave_data.nc"
    dia_path = pyroute_path + "analysis/poly_data"
    sd = datetime(2000, 7, 2, 0, 0)
    ed = datetime(2000, 7, 3, 0, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    for t in dt:
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = return_domain(wind_fname, waves_fname)
        jt, et, x_r, y_r = min_time_calculate(r, t, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        plot_mt_route(t, r, x, y, x_r, y_r, et, jt, dia_path+"/"+str(t))
        # plot_reliability_route(t, r, x, y, pf_vals, jt, dia_path+"/"+str(t))


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
    sd = datetime(2000, 7, 1, 0, 0)
    dist, bearing = haversine(tahiti.long, tahiti.lat, marquesas.long,
                              marquesas.lat)
    tws, twd, wd, wh, wp = return_domain(wind_fname, waves_fname)
    nodes = np.array([i**2 for i in range(7, 15, 2)])
    times = []
    h_vals = []
    for count, node in enumerate(nodes):
        node_distance = dist/node
        r = Route(tahiti, marquesas, node, node,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        jt, x_route, y_route = min_time_calculate(r, sd, craft,
                                                  x, y, land,
                                                  tws, twd, wd, wh, wp,
                                                  False)
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
