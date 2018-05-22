""" Download wave data.

Thomas Dickson
thomas.dickson@soton.ac.uk
30/04/2018
"""

from ecmwfapi import ECMWFDataServer
# from sail_route.weather.weather_assistance import plot_wind_data, generate_gif


def download_weather(path, N, W, S, E):
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
        'date': "2016-05-01/to/2016-09-01",
        'type': "an",
        'class': "ei",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/transat_wind.nc"
    })


def download_wave_interim(path, N, W, S, E):
    """Download wave data."""
    server = ECMWFDataServer()

    server.retrieve({
        "class": "e4",
        "dataset": "interim",
        "date": "2016-05-01/to/2016-08-01",
        "levtype": "sfc",
        "param": "229.140/230.140/232.140",
        "step": "0",
        "stream": "wave",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/summer2016_wave_data.nc"
    })



def download_wind_era40(path, N, W, S, E):
    """Download wind data."""
    server = ECMWFDataServer()
    server.retrieve({
        'stream': "oper",
        'levtype': "sfc",
        'param': "165.128/166.128/167.128",
        'dataset': "era40",
        'step': "0",
        'grid': "0.75/0.75",
        'time': "00/06/12/18",
        'date': "1976-05-01/to/1976-08-01",
        'type': "an",
        'class': "ei",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/finney_wind_forecast.nc"
    })


def download_wave_era40(path, N, W, S, E):
    """Download wave data."""
    server = ECMWFDataServer()

    server.retrieve({
        "class": "e4",
        "dataset": "era40",
        "date": "2016-05-01/to/2016-08-01",
        "levtype": "sfc",
        "param": "229.140/230.140/232.140",
        "step": "0",
        "stream": "wave",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
        'format': "netcdf",
        'target': path+"/data_dir/finney_wave_data.nc"
    })


def download_wind_poly():
    """Download wind for Polynesian routing scenario."""
    pyroute_path = "/home/thomas/Documents/pyroute/"
    path = pyroute_path + "analysis/poly_data"
    download_wind(path, -10, -150.0, -18.0, -135.0)


def download_weather_ERA5(path, N, W, S, E):
    "Download weather from era5 model. Downloads wind and wave data."
    server = ECMWFDataServer()
    server.retrieve({
    "class": "ea",
    "dataset": "era5",
    "date": "2016-05-01/to/2016-05-31",
    "domain": "g",
    "expver": "1",
    "number": "0/1/2/3/4/5/6/7/8/9",
    "param": "229.140/238.140/239.140/245.140/249.140",
    "stream": "ewda",
    "time": "00:00:00/04:00:00/08:00:00/12:00:00/16:00:00/20:00:00",
    "type": "an",
    # 'area': str(N) + "/" + str(W) + "/" + str(S) + "/" + str(E),
    'format': "netcdf",
    'target': path+"/data_dir/2016_summer_data.nc"
    })



if __name__ == '__main__':
    N = 55.0
    S = 0.0
    W = -60.0
    E = -10.0
    path = "/home/thomas/Documents/pyroute/analysis/asv"
    # download_wind_era40(path, N, W, S, E)
    # download_wave_era40(path, N, W, S, E)
    # download_wind(path, N, W, S, E)
    # download_wave_interim(path, N, W, S, E)
    download_weather_ERA5(path, N, W, S, E)
    # plot_wind_data(path+"/data_dir/", "wind_forecast.nc")
    # generate_gif(path+"/data_dir", "2000 July")
