""" Download wave data.

Thomas Dickson
thomas.dickson@soton.ac.uk
30/04/2018
"""


from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

path = "/Users/thomasdickson/Documents/python_routing/analysis/poly_data"

server.retrieve({
    "class": "e4",
    "dataset": "era40",
    "date": "2002-07-01/to/2002-07-31",
    "levtype": "sfc",
    "param": "229.140/230.140/232.140",
    "step": "0",
    "stream": "wave",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    'format': "netcdf",
    'target': path+"/data_dir/wave_data.nc"
})
