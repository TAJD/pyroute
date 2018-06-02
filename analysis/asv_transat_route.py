"""Simulating ASV Transat voyaging

Thomas Dickson
thomas.dickson@soton.ac.uk
01/06/2018
"""
from context import sail_route
import numpy as np
from time import gmtime, strftime
from datetime import datetime
from canoe_voyaging_utils import datetime_range
from asv_utils import asv_uncertain
from sail_route.performance.bbn import gen_env_model
from sail_route.weather.load_weather import process_era5_weather
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route, \
                                   plot_isochrones
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords
from grid_error import calc_h


pp = "/home/td7g11/pyroute/"


def run_simulation_over_days():
    """Run transat routing simulations."""
    start = Location(-2.3700, 50.256)
    finish = Location(-61.777, 17.038)
    fm = gen_env_model()
    craft = asv_uncertain(1.0, 1.0, fm)
    n_nodes = 100
    n_width = n_nodes
    print("Nodes in rank: ", n_nodes)
    print("Nodes in width: ", n_width)
    dist, bearing = haversine(start.long, start.lat,
                              finish.long, finish.lat)
    node_distance = 4000*dist/n_width
    print("Node height distance is ", dist/n_nodes*1000, " m")
    print("Node width distance is ", node_distance, " m")
    area = node_distance*dist/n_nodes*1000
    total_area = n_nodes * n_width * area
    h = (1/(n_nodes * n_width) * total_area)**0.5
    print("h = {0} ".format(h))
    r = Route(start, finish, n_nodes, n_width,
              node_distance, craft)
    weather_path = pp + "analysis/asv_transat/2016_jan_march.nc"
    dia_path = pp + "analysis/asv_transat/results/"
    sd = datetime(2016, 1, 2, 6, 0)
    ed = datetime(2016, 1, 3, 6, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    for t in dt:
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
        jt, et, x_r, y_r = min_time_calculate(r, t, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        fill = 10
        plot_isochrones(t, r, x, y, et, fill,
                        dia_path+str(t)+"_"+str(craft.apf)+"_"+str(n_nodes)+"_")
        plot_mt_route(t, r, x, y, x_r, y_r,
                      et, jt, fill,
                      dia_path+str(t)+"_"+str(craft.apf)+"_"+str(n_nodes)+"_")


def asv_grid_error():
    """
    Check convergence of solution.

    Using Mar
    """
    start = Location(-149.426, -17.651)
    finish = Location(-157.92, 21.83)
    fm = gen_env_model()
    craft = asv_uncertain(1.0, 1.0, fm)
    weather_path = pp + "analysis/asv_transat/2016_jan_march.nc"
    diagram_path = pp + "analysis/asv_transat/results/"
    sd = datetime(2016, 1, 2, 6, 0)
    dist, bearing = haversine(start.long, start.lat,
                              finish.long, finish.lat)
    nodes = np.array([1280])
    times = []
    h_vals = []
    for count, node in enumerate(nodes):
        node_distance = dist/node
        r = Route(start, finish, node, node,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
        jt, et, x_r, y_r = min_time_calculate(r, sd, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - sd
        print("Journey time is: ", vt)
        h_vals.append(calc_h(node, node_distance**2))
        vt = datetime.fromtimestamp(jt) - sd
        print(h_vals[-1], "  ", vt)
        times.append(vt.total_seconds())
    h_vals, times = np.array(h_vals), np.array(times)
    with open(diagram_path+"asv_convergence_"+strftime("%Y-%m-%d %H:%M:%S",
                                                       gmtime())+".txt", 'wb') as f:
        np.savetxt(f, np.c_[nodes, h_vals, times], delimiter='\t')


if __name__ == '__main__':
    # run_simulation_over_days()
    asv_grid_error()
