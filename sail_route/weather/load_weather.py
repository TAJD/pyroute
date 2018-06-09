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
    regridder = xe.Regridder(ds, ds_out, 'patch', reuse_weights=True)
    ds0 = regridder(ds)
    ds0.coords['lat_b'] = ('lat_b', ds0['lat'].values)
    ds0.coords['lon_b'] = ('lon_b', ds0['lon'].values)
    return ds0


def process_wind(path_nc, longs, lats):
    """
    Return wind speed and direction data.

    Data is regridded to the location of each node.
    """
    ds_u10 = load_dataset(path_nc, 'u10')
    regrid_ds_u10 = regrid_data(ds_u10, longs[:, 0], lats[0, :])
    ds_v10 = load_dataset(path_nc, 'v10')
    regrid_ds_v10 = regrid_data(ds_v10, longs[:, 0], lats[0, :])
    ws = 1.943844 * (regrid_ds_u10**2 + regrid_ds_v10**2)**0.5
    wind_dir = np.rad2deg(np.arctan2(regrid_ds_u10, regrid_ds_v10)) + 180.0
    return ws, wind_dir


def process_waves(path_nc, longs, lats):
    """Return wave data."""
    wh = load_dataset(path_nc, 'swh')
    wd = load_dataset(path_nc, 'mwd')
    wp = load_dataset(path_nc, 'mwp')
    regrid_wh = regrid_data(wh, longs[:, 0], lats[0, :])
    regrid_wd = regrid_data(wd, longs[:, 0], lats[0, :])
    regrid_wp = regrid_data(wp, longs[:, 0], lats[0, :])
    return regrid_wh, regrid_wd, regrid_wp


def process_era5_weather(path_nc, longs, lats):
    """Return era5 weather data."""
    wisp = load_dataset(path_nc, 'wind')
    widi = load_dataset(path_nc, 'dwi')
    wh = load_dataset(path_nc, 'shts')
    wd = load_dataset(path_nc, 'mdts')
    wp = load_dataset(path_nc, 'mpts')
    rg_wisp = regrid_data(wisp, longs[:, 0], lats[0, :])
    rg_widi = regrid_data(widi, longs[:, 0], lats[0, :])
    rg_wh = regrid_data(wh, longs[:, 0], lats[0, :])
    rg_wd = regrid_data(wd, longs[:, 0], lats[0, :])
    rg_wp = regrid_data(wp, longs[:, 0], lats[0, :])
    return rg_wisp, rg_widi, rg_wh, rg_wd, rg_wp


def change_area_values(array, value, lon1, lat1, lon2, lat2):
    """
    Change the weather values in a given rectangular area.

    array is an xarray DataArray
    value is the new value
    lon1 and lat1 are the coordinates of the bottom left corner of the area
    lon2 and lat2 are the coordinates of the top right of the area
    """
    lc = array.coords['lon']
    la = array.coords['lat']
    array.loc[dict(lon_b=lc[(lc > lon1) & (lc < lon2)],
                   lat_b=la[(la > lat1) & (la < lat2)])] = value
    return array


if __name__ == '__main__':
    sys.path.append(os.path.abspath("/home/td7g11/pyroute/sail_route/route/"))
    from grid_locations import return_co_ords

    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points.

        Return the value in km.
        """
        lon1, lat1, lon2, lat2 = np.radians(np.array([lon1, lat1, lon2, lat2]))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin((dlat)/2)**2 + np.cos(lat1) * np.cos(lat2) * \
            np.sin((dlon)/2)**2
        dist = 6371 * 2 * np.arcsin(np.sqrt(a)) * 0.5399565
        bearing = np.rad2deg(np.arctan2(dlat, dlon))
        return dist, bearing
    # start = Location(-2.3700, 50.256)
    # finish = Location(-61.777, 17.038)
    start_long = -2.37
    start_lat = 50.256
    finish_long = -61.777
    finish_lat = 17.083
    n_ranks = 10
    n_nodes = 10
    dist, bearing = haversine(start_long, start_lat, finish_long,
                              finish_lat)
    node_distance = 2000*dist/n_nodes
    longs, lats, land = return_co_ords(start_long, finish_long,
                                       start_lat, finish_lat,
                                       n_ranks, n_nodes, dist)
    weather_path = "/home/td7g11/pyroute/analysis/asv_transat/2016_jan_march.nc"
    # look_in_netcdf(weather_path)
    # rg_wisp, rg_widi, rg_wh, rg_wd, rg_wp = process_era5_weather(weather_path, longs, lats)
    wisp = load_dataset(weather_path, 'wind')
    print(wisp.mean())
    wisp = change_area_values(wisp, 10.0, 300, 20, 350, 60)
    print(wisp.mean())
