"""Demonstrating the influence of the failure model.

Tasks achieved in this file;

1. Generation of control weather scenario.
2. Simulation of different levels of failure criteria over the control weather scenario.

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
from sail_route.weather.load_weather import process_era5_weather, change_area_values
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, plot_mt_route, \
                                   plot_isochrones
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords


pp = "/home/td7g11/pyroute/"


def failure_controlled_weather():
    """Load weather file for simulations."""
    rel_levels = np.array([0.81, 0.95, 1.0])
    unc_levels = np.array([1.0])
    test_matrix = np.array(np.meshgrid(rel_levels,
                                       unc_levels)).T.reshape(-1, 2)
    fts = []
    start = Location(-2.3700, 50.256)
    finish = Location(-61.777, 17.038)
    fm = gen_env_model()
    craft = asv_uncertain(1.0, 1.0, fm)
    weather_path = pp + "analysis/asv_transat/2016_jan_march.nc"
    diagram_path = pp + "analysis/failure_model/"
    sd = datetime(2008, 1, 2, 6, 0)
    dist, bearing = haversine(start.long, start.lat,
                              finish.long, finish.lat)
    nodes = 20
    node_distance = 4000*dist/nodes
    r = Route(start, finish, nodes, nodes,
              node_distance*1000.0, craft)
    x, y, land = return_co_ords(r.start.long, r.finish.long,
                                r.start.lat, r.finish.lat,
                                r.n_ranks, r.n_width, r.d_node)
    tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
    # Alter the weather for the entire domain
    lon1 = -59.9
    lat1 = -1.77
    lon2 = -6.4
    lat2 = 61.5
    tws = change_area_values(tws, 15.0, lon1, lat1, lon2, lat2)
    twd = change_area_values(twd, 270.0, lon1, lat1, lon2, lat2)
    #  Alter the wave conditions
    wd = change_area_values(wd, 90.0, lon1, lat1, lon2, lat2)
    wh = change_area_values(wh, 0.0, lon1, lat1, lon2, lat2)

    # # Alter the weather for the smaller area
    a1 = 3.0
    lon1_area1 = -40.0-a1
    lon2_area1 = -40.0+a1
    lat1_area1 = 33.0-a1
    lat2_area1 = 33.0+a1
    wd = change_area_values(wd, 180.0, lon1_area1, lat1_area1,
                            lon2_area1, lat2_area1)
    a2 = 5.0
    lon1_area2 = -40.0-a2
    lon2_area2 = -40.0+a2
    lat1_area2 = 33.0-a2
    lat2_area2 = 33.0+a2
    wh = change_area_values(wh, 4.0, lon1_area2, lat1_area2, lon2_area2,
                            lat2_area2)

    for i in range(test_matrix.shape[0]):
        craft = asv_uncertain(test_matrix[i, 1], test_matrix[i, 0], fm)

        jt, et, x_r, y_r = min_time_calculate(r, sd, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - sd
        fts.append(vt.total_seconds())
    results = np.array(fts)
    print(results)
    save_array = np.hstack((test_matrix, results[..., None]))
    print(save_array)
    with open(diagram_path+"control_"+strftime("""%Y-%m-%d %H:%M:%S""",
                                               gmtime())+".txt",
              'wb') as f:
        np.savetxt(f, save_array, delimiter='\t', fmt='%1.3f')


if __name__ == '__main__':
    failure_controlled_weather()
