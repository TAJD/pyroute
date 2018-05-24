"""Load weather files.

Reduce the amount of interpolating associated with large weather files.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/05/2018
"""

import os
import sys
import numpy as np
import xarray as xr
import xesmf as xe


def look_in_netcdf(path):
    """Load netcdf file and return xrray."""
    with xr.open_dataset(path) as ds:
        print(ds.keys())


def load_dataset(path_nc, var):
    """Load netcdf file and return a specific variable."""
    with xr.open_dataset(path_nc) as ds:
        ds.coords['lat'] = ('latitude', ds['latitude'].values)
        ds.coords['lon'] = ('longitude', ds['longitude'].values)
        ds.swap_dims({'longitude': 'lon', 'latitude': 'lat'})
        return ds[var]


def regrid_data(ds, longs, lats):
    """Regrid dataset to new longs and lats."""
    ds_out = xr.Dataset({'lat': (['lat_b'], lats),
                         'lon': (['lon_b'], longs), })
    regridder = xe.Regridder(ds, ds_out, 'bilinear', reuse_weights=True)
    ds0 = regridder(ds)
    ds0.coords['lat_b'] = ('lat_b', ds0['lat'].values)
    ds0.coords['lon_b'] = ('lon_b', ds0['lon'].values)
    return ds0


def process_wind(path_nc, longs, lats):
    """Return wind speed and direction data regridded to the
    location of each node."""
    ds_u10 = load_dataset(path_nc, 'u10')
    regrid_ds_u10 = regrid_data(ds_u10, longs[:, 0], lats[0, :])
    ds_v10 = load_dataset(path_nc, 'v10')
    regrid_ds_v10 = regrid_data(ds_v10, longs[:, 0], lats[0, :])
    ws = 1.943844 * (regrid_ds_u10**2 + regrid_ds_v10**2)**0.5
    wind_dir = np.rad2deg(np.arctan2(regrid_ds_u10, regrid_ds_v10)) + 180.0
    return ws, wind_dir


if __name__ == '__main__':
    sys.path.append(os.path.abspath("/home/thomas/Documents/pyroute/sail_route/route"))
    from grid_locations import return_co_ords
    def haversine(lon1, lat1, lon2, lat2):
        """Calculate the great circle distance between two points.

        Return the value in km."""
        lon1, lat1, lon2, lat2 = np.radians(np.array([lon1, lat1, lon2, lat2]))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin((dlat)/2)**2 + np.cos(lat1) * np.cos(lat2) * \
            np.sin((dlon)/2)**2
        dist = 6371 * 2 * np.arcsin(np.sqrt(a)) * 0.5399565
        bearing = np.rad2deg(np.arctan2(dlat, dlon))
        return dist, bearing

    start_long = -149.426
    start_lat = -17.651
    finish_long = -157.92
    finish_lat = 21.83
    n_ranks = 10
    n_nodes = 10
    dist, bearing = haversine(start_long, start_lat, finish_long,
                              finish_lat)
    node_distance = 2000*dist/n_nodes
    longs, lats, land = return_co_ords(start_long, finish_long,
                                       start_lat, finish_lat,
                                       n_ranks, n_nodes, dist)
    path = "/home/thomas/Documents/pyroute/analysis/poly_data/data_dir/finney_wind_forecast.nc"
    ws, wd = process_wind(path, longs, lats)
    # print(ws[[0], [0]])
    ind = xr.DataArray([[0], [0]], dims=['lon', 'lat'])
    print(ws.sel(lon_b=-150.2, lat_b=-14.19))
