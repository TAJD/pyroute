"""Basic simulations of Finney voyages.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""
from context import sail_route
import numpy as np
from datetime import datetime
from canoe_voyaging_utils import datetime_range, tong_uncertain
from sail_route.performance.bbn import gen_env_model
from sail_route.weather.load_weather import process_wind, process_waves
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords


# pp = "/home/td7g11/pyroute/"
pp = "/Users/thomasdickson/Documents/python_routing/"
# dia_path = pp + "analysis/poly_data/finney_sims"


def run_simulation_over_days():
    """Run routing simulations between tahiti and marquesas."""
    tahiti = Location(-149.426, -17.651)
    hawaii = Location(-157.92, 21.83)
    fm = gen_env_model()
    craft = tong_uncertain(1.0, 1.0, fm)
    n_nodes = 40
    n_width = n_nodes
    print("Nodes in rank: ", n_nodes)
    print("Nodes in width: ", n_width)
    dist, bearing = haversine(tahiti.long, tahiti.lat, hawaii.long,
                              hawaii.lat)
    node_distance = 4000*dist/n_width
    print("Node height distance is ", dist/n_nodes*1000, " m")
    print("Node width distance is ", node_distance, " m")
    area = node_distance*dist/n_nodes*1000
    total_area = n_nodes * n_width * area
    h = (1/(n_nodes * n_width) * total_area)**0.5
    print("h = {0} ".format(h))
    r = Route(tahiti, hawaii, n_nodes, n_width,
              node_distance, craft)
    wind_fname = pp + "analysis/poly_data/data_dir/finney_wind_forecast.nc"
    waves_fname = pp + "analysis/poly_data/data_dir/finney_wave_data.nc"
    dia_path = pp + "analysis/poly_data/finney_sims/"
    sd = datetime(1976, 5, 1, 0, 0)
    ed = datetime(1976, 5, 2, 0, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    for t in dt:
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd = process_wind(wind_fname, x, y)
        wd, wh, wp = process_waves(waves_fname, x, y)
        jt, et, x_r, y_r = min_time_calculate(r, t, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        fill = 10
        string = str(t)+"_"+str(craft.apf)+"_"+str(craft.unc)+"_"+str(n_nodes)
        plot_mt_route(t, r, x, y, x_r, y_r,
                      et, jt, fill,
                      dia_path+string+"_")


def grid_error():
    """
    Perform grid error study for routing given real weather.

    Return a plot of the difference between results as a function
    of grid error.
    """
    tahiti = Location(-149.426, -17.651)
    hawaii = Location(-157.92, 21.83)
    fm = gen_env_model()
    craft = tong_uncertain(1.0, 1.0, fm)
    wind_fname = pp + "analysis/poly_data/data_dir/wind_forecast.nc"
    waves_fname = pp + "analysis/poly_data/data_dir/wave_data.nc"
    diagram_path = pp + "analysis/poly_data"
    sd = datetime(1976, 5, 1, 0, 0)
    dist, bearing = haversine(tahiti.long, tahiti.lat, hawaii.long,
                              hawaii.lat)
    nodes = np.array([10])
    times = []
    h_vals = []
    for count, node in enumerate(nodes):
        node_distance = dist/node
        r = Route(tahiti, hawaii, node, node,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd = process_wind(wind_fname, x, y)
        wd, wh, wp = process_waves(waves_fname, x, y)
        # jt, et, x_r, y_r = min_time_calculate(r, sd, craft,
        #                                       x, y, land,
        #                                       tws, twd, wd, wh, wp)
    #     vt = datetime.fromtimestamp(jt) - sd
    #     print("Journey time is: ", vt)
    #     h_vals.append(calc_h(node, node_distance**2))
    #     vt = datetime.fromtimestamp(jt) - sd
    #     print(h_vals[-1], "  ", vt)
    #     times.append(vt.total_seconds())
    # h_vals, times = np.array(h_vals), np.array(times)
    # with open(diagram_path+"grid_output_50_100_200_400_800.txt", 'wb') as f:
    #     np.savetxt(f, np.c_[h_vals, times], delimiter='\t')


if __name__ == '__main__':
    run_simulation_over_days()
    # grid_error()
