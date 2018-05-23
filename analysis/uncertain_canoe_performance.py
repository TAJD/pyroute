"""Simulating uncertain canoe performance.

Thomas Dickson
thomas.dickson@soton.ac.uk
21/05/18
"""
from context import sail_route
import numpy as np
from datetime import datetime
from canoe_voyaging_utils import tong_uncertain
from sail_route.weather.weather_assistance import return_domain
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords
from sail_route.performance.bbn import gen_env_model
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_uncertainty(perfs, times):
    """Plot output from uncertainty plot."""
    plt.figure()
    plt.scatter(perfs, times, label='Estimated')
    plt.scatter(1.0, 34*24, label='Hokolua')
    plt.xlabel("Performance variation")
    plt.ylabel("Voyage time (hours)")
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    return plt


def plot_uncertain_routes(start, route, x, y, unc, x_r, y_r, results):
    """Plot uncertain voyaging routes."""
    add_param = 2
    res = 'i'
    plt.figure(figsize=(6, 10))
    map = Basemap(projection='tmerc',
                  ellps='WGS84',
                  lat_0=(y.min() + y.max())/2,
                  lon_0=(x.min() + x.max())/2,
                  llcrnrlon=x.min()-add_param,
                  llcrnrlat=y.min()-add_param,
                  urcrnrlon=x.max()+add_param,
                  urcrnrlat=y.max()+add_param,
                  resolution=res)  # f = fine resolution
    map.drawcoastlines()
    r_s_x, r_s_y = map(route.start.long, route.start.lat)
    map.scatter(r_s_x, r_s_y, color='red', s=50, label='Start')
    r_f_x, r_f_y = map(route.finish.long, route.finish.lat)
    map.scatter(r_f_x, r_f_y, color='blue', s=50, label='Finish')
    parallels = np.arange(-90.0, 90.0, 5.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.arange(180., 360., 5.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1])
    for n, i in enumerate(unc):
        route_x, route_y = map(x_r[n], y_r[n])
        map.plot(route_x, route_y, label="""Perf = {:.2f}\%,
                    Time = {:.2f} hours""".format(i, results[n]))
    plt.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True,
               framealpha=0.5)
    plt.tight_layout()
    return plt


def plot_uncertain_routes_hex(start, route, x, y, unc, x_r, y_r, results):
    """Plot uncertain voyaging routes."""
    add_param = 2
    res = 'i'
    plt.figure(figsize=(6, 10))
    map = Basemap(projection='tmerc',
                  ellps='WGS84',
                  lat_0=(y.min() + y.max())/2,
                  lon_0=(x.min() + x.max())/2,
                  llcrnrlon=x.min()-add_param,
                  llcrnrlat=y.min()-add_param,
                  urcrnrlon=x.max()+add_param,
                  urcrnrlat=y.max()+add_param,
                  resolution=res)  # f = fine resolution
    map.drawcoastlines()
    r_s_x, r_s_y = map(route.start.long, route.start.lat)
    map.scatter(r_s_x, r_s_y, color='red', s=50, label='Start')
    r_f_x, r_f_y = map(route.finish.long, route.finish.lat)
    map.scatter(r_f_x, r_f_y, color='blue', s=50, label='Finish')
    parallels = np.arange(-90.0, 90.0, 5.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.arange(180., 360., 5.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1])
    map.hexbin(np.array(x_r).flatten(), np.array(y_r).flatten(),
               gridsize=400)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    return plt


def run_uncertain_performance_simulation():
    """Run routing simulations for a given day with uncertain performance."""
    start = Location(-149.426, -17.651)
    finish = Location(-157.92, 21.83)
    start_date = datetime(1976, 5, 1, 0, 0)
    n_nodes = 20
    n_width = n_nodes
    print("Nodes in rank: ", n_nodes)
    print("Nodes in width: ", n_width)
    dist, bearing = haversine(start.long, start.lat, finish.long,
                              finish.lat)
    node_distance = 2000*dist/n_width
    print("Node height distance is ", dist/n_nodes*1000, " m")
    print("Node width distance is ", node_distance, " m")
    area = node_distance*dist/n_nodes*1000
    total_area = n_nodes * n_width * area
    h = (1/(n_nodes * n_width) * total_area)**0.5
    print("h = {0} ".format(h))
    pyroute_path = "/home/thomas/Documents/pyroute/"
    wind_fname = pyroute_path + "analysis/poly_data/data_dir/finney_wind_forecast.nc"
    waves_fname = pyroute_path + "analysis/poly_data/data_dir/finney_wave_data.nc"
    dia_path = pyroute_path + "analysis/poly_data/finney_sims"
    unc = np.linspace(0.95, 1.05, 3)
    results = np.zeros_like(unc)
    route_x = []
    route_y = []
    fm = gen_env_model()
    for n, i in enumerate(unc):
        craft = tong_uncertain(i, 1.0, fm)
        r = Route(start, finish, n_nodes, n_width, node_distance,
                  craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = return_domain(wind_fname, waves_fname)
        jt, et, x_r, y_r = min_time_calculate(r, start_date, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - start_date
        results[n] = vt.total_seconds()/(60.0*60.0)
        print(results[n]/24)
        route_x.append(x_r)
        route_y.append(y_r)
    plot_uncertainty(unc, results)
    plt.savefig(dia_path+"/unc_vt.png")
    plot_uncertain_routes(start, r, x, y, unc, route_x, route_y, results)
    plt.savefig(dia_path+"/unc_vt_routes.png")
    # plot_uncertain_routes_hex(start, r, x, y, unc, x_r, y_r, results)
    # plt.show()


if __name__ == '__main__':
    run_uncertain_performance_simulation()
