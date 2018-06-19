"""Sailing craft routing.

Thomas Dickson
thomas.dickson@soton.ac.uk
"""

import sys
import inspect
import numpy as np
import datetime
import textwrap
from datetime import datetime
from datetime import timedelta
import warnings

import matplotlib # removing this causes a segmentation fault
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sail_route.time_func import timefunc
from sail_route.route.grid_locations import gen_indx
from sail_route.route.solve_route import shortest_path, get_locs
from sail_route.performance.cost_function import cost_function
warnings.filterwarnings("ignore")


def recompile_nb_code():
    """Recompile numba generated code."""
    this_module = sys.modules[__name__]
    module_members = inspect.getmembers(this_module)

    for member_name, member in module_members:
        if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
            member.recompile()


recompile_nb_code()


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


class Location(object):
    """Location."""

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
def min_time_calculate(route, time, craft, x, y,
                       land, tws, twd, wd, wh, wp, verb=True):
    """Calculate the earliest arrival time across co-ordinates."""
    earl_time = np.full_like(x, np.inf)
    indxs, pindxs = gen_indx(x)
    end_node = 0
    journey_time = 10**10
    for i in range(route.n_width):
        wind_speed = tws.sel(lon_b=x[0, i], lat_b=y[0, i],
                             time=time, method='nearest').data
        wind_dir = twd.sel(lon_b=x[0, i], lat_b=y[0, i],
                           time=time, method='nearest').data
        i_wd = wd.sel(lon_b=x[0, i], lat_b=y[0, i],
                      time=time, method='nearest').data
        i_wh = wh.sel(lon_b=x[0, i], lat_b=y[0, i],
                      time=time, method='nearest').data
        i_wp = wp.sel(lon_b=x[0, i], lat_b=y[0, i],
                      time=time, method='nearest').data
        travel_time = cost_function(route.start.long,
                                    route.start.lat,
                                    x[0, i], y[0, i],
                                    wind_speed, wind_dir,
                                    i_wd, i_wh, i_wp,
                                    craft)
        if (travel_time == np.inf) | (land[0, i] is True):
            pass
        else:
            total_time = time + travel_time
            earl_time[0, i] = total_time.timestamp()
    for i in range(route.n_ranks-1):
        for j in range(route.n_width):
            if earl_time[i, j] == np.inf:
                pass
            else:
                utime = datetime.fromtimestamp(earl_time[i, j])
                i_wd = wd.sel(lon_b=x[i, j], lat_b=y[i, j],
                              time=utime, method='nearest').data
                i_wh = wh.sel(lon_b=x[i, j], lat_b=y[i, j],
                              time=utime, method='nearest').data
                i_wp = wp.sel(lon_b=x[i, j], lat_b=y[i, j],
                              time=utime, method='nearest').data
                i_tws = tws.sel(lon_b=x[i, j], lat_b=y[i, j],
                                time=utime, method='nearest').data
                i_twd = twd.sel(lon_b=x[i, j], lat_b=y[i, j],
                                time=utime, method='nearest').data
                lifetime = utime - time
                for k in range(route.n_width):
                        if land[i+1, k] is True:
                            earl_time[i+1, k] == np.inf
                        else:
                            travel_time = cost_function(x[i, j],
                                                        y[i, j],
                                                        x[i+1, k],
                                                        y[i+1, k],
                                                        i_tws, i_twd,
                                                        i_wd, i_wh, i_wp,
                                                        craft,
                                                        lifetime)
                        if (travel_time == np.inf):
                            pass
                        else:
                            jt = utime + travel_time
                            if jt.timestamp() < earl_time[i+1, k]:
                                earl_time[i+1, k] = jt.timestamp()
                                pindxs[i+1, k] = indxs[i, j]
            if np.isfinite(earl_time[i+1, :]) is not True:
                pass
    for i in range(route.n_width):
        if earl_time[-1, i] == np.inf:
            pass
        else:
            time = datetime.fromtimestamp(earl_time[-1, i])
            i_wd = wd.sel(lon_b=x[-1, j], lat_b=y[-1, j],
                          time=utime, method='nearest').data
            i_wh = wh.sel(lon_b=x[-1, j], lat_b=y[-1, j],
                          time=utime, method='nearest').data
            i_wp = wp.sel(lon_b=x[-1, j], lat_b=y[-1, j],
                          time=utime, method='nearest').data
            i_tws = tws.sel(lon_b=x[-1, j], lat_b=y[-1, j],
                            time=utime, method='nearest').data
            i_twd = twd.sel(lon_b=x[-1, j], lat_b=y[-1, j],
                            time=utime, method='nearest').data
            travel_time = cost_function(x[-1, i],
                                        y[-1, i],
                                        route.finish.long,
                                        route.finish.lat,
                                        i_tws, i_twd,
                                        i_wd, i_wh, i_wp,
                                        craft,
                                        lifetime)
            if travel_time == np.inf:
                pass
            else:
                et = datetime.fromtimestamp(earl_time[-1, i]) + travel_time
                if datetime.fromtimestamp(journey_time) > et:
                    journey_time = et.timestamp()
                    end_node = indxs[-1, i]
    sp = shortest_path(indxs, pindxs, [end_node])
    x_route, y_route = get_locs(indxs, sp, x, y)
    x_route = np.hstack(([route.finish.long], x_route,
                        [route.start.long]))
    y_route = np.hstack(([route.finish.lat], y_route,
                        [route.start.lat]))
    if verb is True:
        return journey_time, earl_time, x_route, y_route
    else:
        return journey_time, x_route, y_route


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


def plot_mt_route(start, route, x, y, x_r, y_r, et, jt, fill, fname):
    """Plot minimum time output from routing simulations."""
    vt = datetime.fromtimestamp(jt) - start
    # ul = jt + vt.total_seconds()/6
    add_param = fill
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
    parallels = np.arange(-90.0, 90.0, 20.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.arange(180., 360., 20.)
    map.fillcontinents(color='black')
    map.drawmeridians(meridians, labels=[0, 0, 0, 1])
    map.scatter(r_f_x, r_f_y, color='blue', s=50, label='Finish')
    if vt.total_seconds() < 10000000:
        x_r, y_r = map(x_r, y_r)
        map.plot(x_r, y_r, color='green', label='Minimum time path')
        x, y = map(x, y)
        ctf = map.contourf(x, y, et, cmap='bwr')
        y_tick_labs = [timestamp_to_delta_time(start, x) for x in
                       np.linspace(et[et < 1e307].min(),
                                   et[et < 1e307].max(), 9)]
        # y_tick_labs = [timestamp_to_delta_time(start, x) for x in
        #                np.linspace(et.min(),
        #                            et.max(), 9)]
        cbar = plt.colorbar(ctf, orientation='horizontal')
        cbar.ax.set_xticklabels(y_tick_labs, rotation=25)
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
    # plt.tight_layout()
    plt.savefig(fname+"min_time"+".png")
    plt.clf()


def plot_isochrones(start, route, x, y, et, fill, fname):
    """Plot isochrones for shortest path."""
    plt.figure(figsize=(6, 10))
    map = Basemap(projection='merc',
                  ellps='WGS84',
                  lat_0=(y.min() + y.max())/2,
                  lon_0=(x.min() + x.max())/2,
                  llcrnrlon=x.min()-fill,
                  llcrnrlat=y.min()-fill,
                  urcrnrlon=x.max()+fill,
                  urcrnrlat=y.max()+fill,
                  resolution='i')  # f = fine resolution
    map.drawcoastlines()
    start1_lon = -16.0
    start1_lat = 51.0
    start2_lon = -8.0
    start2_lat = 45.0
    fin_lon = -60.0
    fin1_lat = 25.0
    fin2_lat = 10.0

    s1lon, s1lat = map(start1_lon, start1_lat)
    s2lon, s2lat = map(start2_lon, start2_lat)
    f1lon, f1lat = map(fin_lon, fin1_lat)
    f2lon, f2lat = map(fin_lon, fin2_lat)
    plt.plot([s1lon, s2lon], [s1lat, s2lat], label="Start line")
    plt.plot([f1lon, f2lon], [f1lat, f2lat], label="Finish line")
    x, y = map(x, y)
    try:
        ctf = map.contourf(x, y, et, cmap='bwr')
        y_tick_labs = [timestamp_to_delta_time(start, x) for x in
                       np.linspace(et[et < 10000000].min(),
                                   et[et < 10000000].max(), 9)]
        cbar = plt.colorbar(ctf, orientation='horizontal')
        cbar.ax.set_xticklabels(y_tick_labs, rotation=25)
    except ValueError:
        pass
    map.scatter(x[et > 9000000], y[et > 9000000], color='red',
                s=1, label='No go')
    map.fillcontinents(color='black')
    plt.savefig(fname+"isochrones"+".png")
