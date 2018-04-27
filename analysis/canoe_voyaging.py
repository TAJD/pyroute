"""Basic simulations of Finney voyages.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""

from context import sail_route
from sail_route.weather.weather_assistance import download_wind, plot_wind_data, generate_gif
from sail_route.sail_routing import Location, return_boat_perf, Route, \
                                   min_time_calculate, plot_route

def download_wind_poly():
    path = "/Users/thomasdickson/Documents/sail_routing/python_routing/analysis/poly_data/"
    download_wind(path, 25.0, -165.0, -18.0, -135.0)


def plot_wind():
    path = "/Users/thomasdickson/Documents/sail_routing/python_routing/analysis/poly_data"
    plot_wind_data(path, "data_dir/wind_forecast.nc")
    generate_gif(path, "Polynesian_July")

if __name__ == '__main__':
    download_wind_poly()
    plot_wind()
