"""
Generate grid locations on real world domain.

Thomas Dickson

28/03/2018


Function to:

Generate a grid of points between a start and finish location.
Return a true/false matrix which determines whether the point
lies on land or not.

"""

import os
import numpy as np
from mpl_toolkits.basemap import Basemap
import pyproj
from shapely.geometry import Point


def line_points(x, y, n_nodes, dist):
    """Calculate the locations of the points along a rank."""
    p_ll = pyproj.Proj(init='epsg:4326')
    p_mt = pyproj.Proj(init='epsg:3857')
    g = pyproj.Geod(ellps='clrk66')
    tran = pyproj.transform(p_ll, p_mt, x, y)
    upper = Point((tran[0], tran[1]+dist*n_nodes/2))
    lower = Point((tran[0], tran[1]-dist*n_nodes/2))
    tran_upper = pyproj.transform(p_mt, p_ll, upper.x, upper.y)
    tran_lower = pyproj.transform(p_mt, p_ll, lower.x, lower.y)
    points = g.npts(tran_upper[0], tran_upper[1],
                    tran_lower[0], tran_lower[1], n_nodes-2)
    return np.vstack((np.array(tran_upper), np.array(points),
                     np.array(tran_lower)))


def gen_grid(start_long, finish_long, start_lat, finish_lat,
             n_ranks=10, n_nodes=10, dist=5000):
    """Function to return grid between start and finish."""
    g = pyproj.Geod(ellps='clrk66')
    azimuths = g.inv(start_long, start_lat, finish_long, finish_lat)
    rot = azimuths[0]-90.0
    height = dist * np.sin(rot) + dist*np.cos(rot)
    great_circle = g.npts(start_long, start_lat, finish_long, finish_lat,
                          n_ranks)
    grid = [line_points(g[0], g[1], n_nodes, height) for g in great_circle]
    return grid


def check_land(grid):
    """Check co-ordinates."""
    bm = Basemap()
    points = []
    for g in grid:
        g_land = [bm.is_land(p[0], p[1]) for p in g]
        points.append(g_land)
    return points


def return_co_ords(start_long, finish_long, start_lat, finish_lat,
                   n_ranks=10, n_nodes=10, dist=5000):
    grid = gen_grid(start_long, finish_long, start_lat, finish_lat,
                    n_ranks, n_nodes, dist)
    land = check_land(grid)
    g = np.reshape(np.hstack(grid), (1, -1))
    x = np.reshape(g[0][::2], (n_ranks, n_nodes))
    y = np.reshape(g[0][1::2], (n_ranks, n_nodes))
    return x, y, np.array(land)


if __name__ == '__main__':
    start_long = -14.0
    finish_long = -6.0
    start_lat = 47.0
    finish_lat = 47.0
    longs, lats, land = return_co_ords(start_long, finish_long,
                                       start_lat, finish_lat)
