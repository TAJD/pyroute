import datetime
import subprocess
import os
import iris
import numpy as np
from mpl_toolkits import basemap
from ecmwfapi import ECMWFDataServer
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import iris.coord_categorisation
import cartopy.crs as ccrs
from numba import jit


def download_wind(path, N, W, S, E):
    """Download wind data."""
    server = ECMWFDataServer()
    server.retrieve({
        'stream': "oper",
        'levtype': "sfc",
        'param': "165.128/166.128/167.128",
        'dataset': "interim",
        'step': "0",
        'grid': "0.75/0.75",
        'time': "00/06/12/18",
        'date': "2014-07-01/to/2014-07-14",
        'type': "an",
        'class': "ei",
        #area:  N/W/S/E
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/wind_forecast.nc"
    })


def download_wave_date(path, N, W, S, E):
    """Download wave data."""
    server = ECMWFDataServer()

    server.retrieve({
        "class": "e4",
        "dataset": "era40",
        "date": "2002-07-01/to/2002-07-31",
        "levtype": "sfc",
        "param": "229.140/230.140/232.140",
        "step": "0",
        "stream": "wave",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/wave_data.nc"
    })


def plot_wind_data(folder, fname):
    """Use iris to manipulate data."""
    vwind = iris.load_cube(fname, '10 metre V wind component')
    uwind = iris.load_cube(fname, '10 metre U wind component')

    windspeed = 1.943844 * (uwind ** 2 + vwind ** 2) ** 0.5
    windspeed.rename('windspeed')

    ulon = uwind.coord('longitude')
    x = ulon.points
    y = uwind.coord('latitude').points
    X, Y = np.meshgrid(x, y)
    u = uwind.data
    v = vwind.data
    u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
    v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)
    t = [datetime.timedelta(hours=np.float64(x)) for x in
         np.array(windspeed.coord('time').points)]
    timestamps = np.array(t) + datetime.datetime(1900, 1, 1)
    for i, yx_slice in enumerate(windspeed.slices(['latitude',
                                                   'longitude'])):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        cf0 = plt.contourf(X, Y, windspeed[i].data, np.arange(0.0, 25.0, 3.0))
        cb = plt.colorbar(cf0)
        cb.set_label('Wind speed (knots)')
        plt.quiver(x, y, u_norm[i], v_norm[i], pivot='middle')
        plt.title(str(timestamps[i]))
        plt.savefig(folder+"file"+str(i)+".png")
        plt.close()


def generate_gif(folder, name):
    """Generate gif from pngs in folder. Delete pngs afterwards."""
    os.chdir(folder)
    subprocess.call(['ffmpeg', '-framerate', '0.5', '-i', 'file%d.png',
                     '-r', '30', 'output.avi'])
    subprocess.call(['ffmpeg', '-i', 'output.avi', name+'.gif'])
    test = os.listdir(folder)
    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(folder, item))
        if item.endswith(".avi"):
            os.remove(os.path.join(folder, item))


@jit(cache=True)
def prepare_wind_data(fname):
    """Return wind direction and speed from V and U components."""
    vwind = iris.load_cube(fname, '10 metre V wind component')
    uwind = iris.load_cube(fname, '10 metre U wind component')
    windspeed = 1.943844 * (uwind ** 2 + vwind ** 2) ** 0.5
    windspeed.rename('windspeed')
    u = uwind.data
    v = vwind.data
    unorm = uwind/np.sqrt(u ** 2.0 + v ** 2.0)
    vnorm = vwind/np.sqrt(u ** 2.0 + v ** 2.0)
    wind_dir = np.rad2deg(np.arctan2(unorm.data, vnorm.data)) + 180.0
    uwind.data = wind_dir
    uwind.rename('winddir')
    return windspeed, uwind


@jit(cache=True)
def prepare_wave_data(fname):
    """Return cubes of wave height, wave direction and wave period from cubes."""
    wh = iris.load_cube(fname, 'Significant height of combined wind waves and swell')
    wd = iris.load_cube(fname, 'Mean wave direction')
    wp = iris.load_cube(fname, 'Mean wave period')
    return wd, wh, wp


@jit(cache=True)
def setup_interpolator(cube):
    """Return interpolator for lat, long and time for cube."""
    if cube is 0.0:
        return 0.0
    interp = iris.analysis.Linear().interpolator(cube, ['longitude',
                                                        'latitude',
                                                        'time'])
    return interp
