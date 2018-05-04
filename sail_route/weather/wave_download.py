""" Download wave data.

Thomas Dickson
thomas.dickson@soton.ac.uk
30/04/2018
"""
import iris

# from ecmwfapi import ECMWFDataServer


def prepare_wave_data(fname):
    """Return cubes of wave height, wave direction and wave period from cubes."""
    wh = iris.load_cube(fname, 'Significant height of combined wind waves and swell')
    wd = iris.load_cube(fname, 'Mean wave direction')
    wp = iris.load_cube(fname, 'Mean wave period')
    return wd, wh, wp



if __name__ == '__main__':
    fname = "/home/thomas/Documents/pyroute/analysis/poly_data/data_dir/wave_data.nc"
    print(prepare_wave_data(fname))
