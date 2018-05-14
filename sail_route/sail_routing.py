""" Sail routing module

Thomas Dickson
thomas.dickson@soton.ac.uk
"""

import sys
import inspect
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
import warnings

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sail_route.time_func import timefunc
from sail_route.route.grid_locations import gen_indx
from sail_route.route.solve_route import shortest_path, get_locs
from sail_route.performance.cost_function import cost_function
from sail_route.weather.weather_assistance import return_domain, \
                                       setup_interpolator
warnings.filterwarnings("ignore")

def recompile_nb_code():
    this_module = sys.modules[__name__]
    module_members = inspect.getmembers(this_module)

    for member_name, member in module_members:
        if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
            member.recompile()
recompile_nb_code()


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = """\\usepackage{subdepth},
                                         \\usepackage{type1cm}"""


class Location(object):
    "Location"

    def __init__(self, long, lat):
        """Return a location object."""
        self.long = long
        self.lat = lat


class Route:
    """Route object."""

    def __init__(self, start, finish, n_ranks, n_width, d_node, craft):
        """Initialise route object."""
        self.start = start
        self.finish = finish
        self.n_ranks = n_ranks
        self.n_width = n_width
        self.d_node = d_node
        self.craft = craft


@timefunc
def min_time_calculate(route, time, craft, x, y, land, tws, twd, wd, wh, wp):
    """Calculate the earliest arrival time across co-ordinates."""
    pf_vals = np.zeros_like(x)  # array to hold failure probability
    earl_time = np.full_like(x, np.inf)  # array to hold earliest times
    indxs, pindxs = gen_indx(x)  # array to hold indexes of each location
    tws_interp = setup_interpolator(tws)
    twd_interp = setup_interpolator(twd)
    wd_interp = setup_interpolator(wd)
    wh_interp = setup_interpolator(wh)
    wp_interp = setup_interpolator(wp)
    journey_time = 10**10

    for i in range(route.n_width):
        if land[i, 0] is True:
            earl_time[i, 0] == np.inf
        i_tws = tws_interp([x[i, 0], y[i, 0], time]).data
        i_twd = twd_interp([x[i, 0], y[i, 0], time]).data
        i_wd = wd_interp([x[i, 0], y[i, 0], time]).data
        i_wh = wh_interp([x[i, 0], y[i, 0], time]).data
        i_wp = wp_interp([x[i, 0], y[i, 0], time]).data
        travel_time, pf = cost_function(route.start.long,
                                        route.start.lat,
                                        x[i, 0], y[i, 0],
                                        i_tws, i_twd,
                                        i_wd, i_wh, i_wp,
                                        craft)
        total_time = time + travel_time
        pf_vals[i, :] = pf
        earl_time[i, 0] = total_time.timestamp()

    for j in range(route.n_ranks-1):
        for i in range(route.n_width):
            utime = datetime.fromtimestamp(earl_time[i, j])
            i_wd = wd_interp([x[i, j], y[i, j], time]).data
            i_wh = wh_interp([x[i, j], y[i, j], time]).data
            i_wp = wp_interp([x[i, j], y[i, j], time]).data
            i_tws = tws_interp([x[i, j], y[i, j], time]).data
            i_twd = twd_interp([x[i, j], y[i, j], time]).data
            lifetime = utime - time
            for k in range(route.n_width):
                    if land[k, j+1] is True:
                        earl_time[k, j+1] == np.inf
                    else:
                        travel_time, pf = cost_function(x[i, j],
                                                        y[i, j],
                                                        x[k, j+1],
                                                        y[k, j+1],
                                                        i_tws, i_twd,
                                                        i_wd, i_wh, i_wp,
                                                        craft,
                                                        lifetime)
                    jt = utime + travel_time
                    if jt.timestamp() < earl_time[k, j+1]:
                        earl_time[k, j+1] = jt.timestamp()
                        pf_vals[k, j+1] = pf
                        pindxs[k, j+1] = indxs[i, j]

    for i in range(route.n_width):
        time = datetime.fromtimestamp(earl_time[i, -1])
        i_tws = tws_interp([x[i, -1],
                           y[i, -1], time]).data
        i_twd = twd_interp([x[i, -1],
                           y[i, -1], time]).data
        travel_time, pf = cost_function(x[i, -1],
                                        y[i, -1],
                                        route.finish.long,
                                        route.finish.lat,
                                        i_tws, i_twd,
                                        i_wd, i_wh, i_wp,
                                        craft,
                                        lifetime)
        et = datetime.fromtimestamp(earl_time[i, -1]) + travel_time
        if datetime.fromtimestamp(journey_time) > et:
            journey_time = et.timestamp()
            pf_vals[i, -1] = pf
            end_node = indxs[i, -1]
    sp = shortest_path(indxs, pindxs, [end_node])
    x_route, y_route = get_locs(indxs, sp, x, y)
    return journey_time, earl_time, pf_vals, x_route, y_route


def min_vals(x, y, et):
    """Identify the locations along the route identified with minimum vals."""
    x_ind, y_ind = np.where(et == et.min(axis=0))
    x_locs = x[x_ind, y_ind]
    y_locs = y[x_ind, y_ind]
    earliest_times = et[x_ind, y_ind]
    return x_locs, y_locs, earliest_times


def round_timedelta(td, period):
    """Round the given timedelta by the given timedelta period.

    :param td: `timedelta` to round
    :param period: `timedelta` period to round by.
    """
    period_seconds = period.total_seconds()
    half_period_seconds = period_seconds / 2
    remainder = td.total_seconds() % period_seconds
    if remainder >= half_period_seconds:
        return timedelta(seconds=td.total_seconds() + (period_seconds - remainder))
    else:
        return timedelta(seconds=td.total_seconds() - remainder)


def timestamp_to_delta_time(start, x):
    """Get voyage time from start and finish times."""
    delta = datetime.fromtimestamp(x) - start
    return round_timedelta(delta, timedelta(minutes=1))


def plot_mt_route(start, route, x, y, x_r, y_r, et, jt, fname):
    """Plot minimum time output from routing simulations."""
    vt = datetime.fromtimestamp(jt) - start
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
    x_r, y_r = map(x_r, y_r)
    map.plot(x_r, y_r, color='green')
    x, y = map(x, y)
    ctf = map.contourf(x, y, et, cmap='gray')
    cbar = plt.colorbar(ctf, orientation='horizontal')
    y_tick_labs = [timestamp_to_delta_time(start, x) for x in
                   np.linspace(et.min(), et.max(), 9)]
    cbar.ax.set_xticklabels(y_tick_labs, rotation=25)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
               borderaxespad=0)
    plt.tight_layout()
    plt.title("Minimum journey time: " + str(vt))
    plt.savefig(fname+"min_time"+".png")
    plt.clf()


def plot_reliability_route(start, route, x, y, pf_vals, jt, fname):
    """Plot reliability predictions from routing."""
    plt.figure(figsize=(6, 10))
    res = 'i'
    add_param = 2
    map = Basemap(projection='tmerc',
                  ellps='WGS84',
                  lat_0=(y.min() + y.max())/2,
                  lon_0=(x.min() + x.max())/2,
                  llcrnrlon=x.min()-add_param,
                  llcrnrlat=y.min()-add_param,
                  urcrnrlon=x.max()+add_param,
                  urcrnrlat=y.max()+add_param,
                  # lat_ts=(y.min() + y.max())/2,
                  resolution=res)  # f = fine resolution
    map.drawcoastlines()
    x, y = map(x, y)
    ctf = map.contourf(x, y, pf_vals, cmap='Reds', vmin=0, vmax=1)
    r_s_x, r_s_y = map(route.start.long, route.start.lat)
    map.scatter(r_s_x, r_s_y, color='red', s=50, label='Start')
    r_f_x, r_f_y = map(route.finish.long, route.finish.lat)
    map.scatter(r_f_x, r_f_y, color='blue', s=50, label='Finish')
    x_locs_pf, y_locs_pf, pf_min = min_vals(x, y, pf_vals)
    cbar = plt.colorbar(ctf, orientation='horizontal',
                        boundaries=np.linspace(0, 1, 10))
    cbar.set_label(r'$p_f$')
    plt.savefig(fname+"reliability"+".png")
    plt.clf()
