""" Download wave data.

Thomas Dickson
thomas.dickson@soton.ac.uk
30/04/2018
"""

from ecmwfapi import ECMWFDataServer
from sail_route.weather.weather_assistance import plot_wind_data


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
        'date': "2000-07-01/to/2000-07-14",
        'type': "an",
        'class': "ei",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/wind_forecast.nc"
    })


def download_wave(path, N, W, S, E):
    """Download wave data."""
    server = ECMWFDataServer()

    server.retrieve({
        "class": "e4",
        "dataset": "era40",
        "date": "2000-07-01/to/2000-07-31",
        "levtype": "sfc",
        "param": "229.140/230.140/232.140",
        "step": "0",
        "stream": "wave",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/wave_data.nc"
    })


def download_wind_poly():
    """Download wind for Polynesian routing scenario."""
    pyroute_path = "/home/thomas/Documents/pyroute/"
    path = pyroute_path + "analysis/poly_data"
    download_wind(path, -10, -150.0, -18.0, -135.0)


if __name__ == '__main__':
    N = -10
    S = -18.0
    W = -150.0
    E = -135.0
    path = "/home/thomas/Documents/pyroute/analysis/poly_data"
    download_wind(path, N, W, S, E)
    download_wave(path, N, W, S, E)
