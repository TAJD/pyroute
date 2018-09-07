"""Simulating ASV Transat voyaging

Thomas Dickson
thomas.dickson@soton.ac.uk
01/06/2018
"""
from context import sail_route
import numpy as np
import textwrap
from time import gmtime, strftime
from datetime import datetime
from canoe_voyaging_utils import datetime_range
from asv_utils import asv_uncertain
from sail_route.performance.bbn import gen_env_model
from sail_route.weather.load_weather import process_era5_weather, change_area_values
from sail_route.sail_routing import Location, Route, \
                                   min_time_calculate, \
                                   plot_mt_route
from sail_route.performance.cost_function import haversine
from sail_route.route.grid_locations import return_co_ords
from grid_error import calc_h


import matplotlib # removing this causes a segmentation fault
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = """\\usepackage{subdepth},
                                         \\usepackage{type1cm}"""


pp = "/home/td7g11/pyroute/"

a1 = 3.0
lon1_area1 = -40.0-a1
lon2_area1 = -40.0+a1
lat1_area1 = 33.0-a1
lat2_area1 = 33.0+a1

a2 = 5.0
lon1_area2 = -40.0-a2
lon2_area2 = -40.0+a2
lat1_area2 = 33.0-a2
lat2_area2 = 33.0+a2


def run_simulation_over_days():
    """Run transat routing simulations."""
    # Specify the bounds of the whole region
    lon1 = -59.9
    lat1 = -1.77
    lon2 = -6.4
    lat2 = 61.5
    start = Location(-12.0, 45.0)
    finish = Location(-60.0, 17.5)
    fm = gen_env_model()
    craft = asv_uncertain(1.0, 1.0, fm)
    n_nodes = 480
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
    sd = datetime(2008, 1, 2, 6, 0)
    ed = datetime(2008, 1, 3, 6, 0)
    dt = [d for d in datetime_range(sd, ed, {'days': 1, 'hours': 0})]
    x, y, land = return_co_ords(r.start.long, r.finish.long,
                                r.start.lat, r.finish.lat,
                                r.n_ranks, r.n_width, r.d_node)
    tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
    tws = change_area_values(tws, 15.0, lon1, lat1, lon2, lat2)
    twd = change_area_values(twd, 0.0, lon1, lat1, lon2, lat2)
    wd = change_area_values(wd, 0.0, lon1, lat1, lon2, lat2)
    wh = change_area_values(wh, 0.0, lon1, lat1, lon2, lat2)
    wh = change_area_values(wh, 4.0, lon1_area2, lat1_area2, lon2_area2,
                            lat2_area2)
    wd = change_area_values(wd, 240.0, lon1_area1, lat1_area1,
                            lon2_area1, lat2_area1)
    for t in dt:
        jt, et, x_r, y_r = min_time_calculate(r, t, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - t
        print("Journey time is: ", vt)
        fill = 10
        string = str(t)+"_"+str(craft.apf)+"_"+str(craft.unc)+"_"+str(n_nodes)+"_weather"
        plot_failure_route(t, r, x, y, x_r, y_r,
                           et, jt, fill,
                           dia_path+string+"_")
        plot_mt_route(t, r, x, y, x_r, y_r,
                      et, jt, fill,
                      dia_path+string+"_")


def asv_grid_error():
    """
    Check convergence of solution.

    Using Maribot Vane performance estimates for the transatlantic voyage.
    """
    lon1 = -59.9
    lat1 = -1.77
    lon2 = -6.4
    lat2 = 61.5
    start = Location(-12.0, 45.0)
    finish = Location(-60.0, 17.5)
    fm = gen_env_model()
    craft = asv_uncertain(1.0, 1.0, fm)
    weather_path = pp + "analysis/asv_transat/2016_jan_march.nc"
    diagram_path = pp + "analysis/asv_transat/results/"
    sd = datetime(2016, 1, 1, 6, 0)
    dist, bearing = haversine(start.long, start.lat,
                              finish.long, finish.lat)
    nodes = np.array([640])
    times = []
    h_vals = []
    for count, node in enumerate(nodes):
        node_distance = 4000*dist/node
        r = Route(start, finish, node, node,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
        tws = change_area_values(tws, 15.0, lon1, lat1, lon2, lat2)
        twd = change_area_values(twd, 0.0, lon1, lat1, lon2, lat2)
        wd = change_area_values(wd, 0.0, lon1, lat1, lon2, lat2)
        wh = change_area_values(wh, 0.0, lon1, lat1, lon2, lat2)
        wh = change_area_values(wh, 4.0, lon1_area2, lat1_area2, lon2_area2,
                                lat2_area2)
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


def reliability_uncertainty_routing():
    """Routing for a range of uncertainty and reliability levels."""
    rel_levels = np.array([0.7, 0.85, 1.0])
    unc_levels = np.array([1.0])
    test_matrix = np.array(np.meshgrid(rel_levels,
                                       unc_levels)).T.reshape(-1, 2)
    fts = []
    start = Location(-2.3700, 50.256)
    finish = Location(-61.777, 17.038)
    fm = gen_env_model()
    weather_path = pp + "analysis/asv_transat/2016_jan_march.nc"
    diagram_path = pp + "analysis/asv_transat/results/"
    sd = datetime(2016, 1, 2, 6, 0)
    dist, bearing = haversine(start.long, start.lat,
                              finish.long, finish.lat)
    nodes = 60
    node_distance = 4000*dist/nodes
    for i in range(test_matrix.shape[0]):
        craft = asv_uncertain(test_matrix[i, 1], test_matrix[i, 0], fm)
        r = Route(start, finish, nodes, nodes,
                  node_distance*1000.0, craft)
        x, y, land = return_co_ords(r.start.long, r.finish.long,
                                    r.start.lat, r.finish.lat,
                                    r.n_ranks, r.n_width, r.d_node)
        tws, twd, wd, wh, wp = process_era5_weather(weather_path, x, y)
        jt, et, x_r, y_r = min_time_calculate(r, sd, craft,
                                              x, y, land,
                                              tws, twd, wd, wh, wp)
        vt = datetime.fromtimestamp(jt) - sd
        fts.append(vt.total_seconds())
    results = np.array(fts)
    print(results)
    save_array = np.hstack((test_matrix, results[..., None]))
    print(save_array)
    with open(diagram_path+"unc_reliability_routing_"+strftime("""%Y-%m-%d
                                                               %H:%M:%S""",
                                                               gmtime())+".txt",
              'wb') as f:
        np.savetxt(f, save_array, delimiter='\t', fmt='%1.3f')


def plot_failure_route(start, route, x, y, x_r, y_r, et, jt, fill, fname):
    """Plot minimum time output from routing simulations."""
    vt = datetime.fromtimestamp(jt) - start
    # ul = jt + vt.total_seconds()/6
    add_param = fill
    res = 'i'
    plt.figure(figsize=(6, 10))
    map = Basemap(projection='merc',
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
    parallels = np.arange(-90.0, 90.0, 20.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.arange(180., 360., 20.)
    map.fillcontinents(color='black')
    map.drawmeridians(meridians, labels=[0, 0, 0, 1])
    map.scatter(r_f_x, r_f_y, color='blue', s=50, label='Finish')

    x1, y1 = map(lon1_area2, lat1_area2)
    x2, y2 = map(lon1_area2, lat2_area2)
    x3, y3 = map(lon2_area2, lat2_area2)
    x4, y4 = map(lon2_area2, lat1_area2)
    poly1 = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                    fill=False, hatch='\\', label="Area 2")
    plt.gca().add_patch(poly1)
    x1, y1 = map(lon1_area1, lat1_area1)
    x2, y2 = map(lon1_area1, lat2_area1)
    x3, y3 = map(lon2_area1, lat2_area1)
    x4, y4 = map(lon2_area1, lat1_area1)
    poly2 = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                    fill=False, hatch='/', label="Area 1")
    plt.gca().add_patch(poly2)

    if vt.total_seconds() < 10000000:
        x_r, y_r = map(x_r, y_r)
        map.plot(x_r, y_r, color='green', label='Minimum time path')
        tit = "\n".join(textwrap.wrap("Journey time: " + str(vt), 80))
        plt.title(tit)
    else:
        plt.title("Voyage failed")
        try:
            map.scatter(x[et == np.inf], y[et == np.inf], color='red',
                        s=1, label='No go')
        except ValueError:
            pass
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)
    plt.savefig(fname+"min_time"+".png")
    plt.clf()


if __name__ == '__main__':
    # run_simulation_over_days()
    # asv_grid_error()
    reliability_uncertainty_routing()
