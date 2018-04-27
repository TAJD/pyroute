"""Using developed sail routing methodology.

Thomas Dickson
thomas.dickson@soton.ac.uk
25/04/18
"""
from context import sail_route
from datetime import datetime


from sail_route.sail_routing import Location, Route, min_time_calculate, \
                                   plot_route
from sail_route.performance.craft_performance import return_boat_perf
from sail_route.route.grid_locations import return_co_ords


def run_simulation():
    start = Location(-14.0, 47.0)
    finish = Location(-6.0, 47.0)
    craft = return_boat_perf()
    no_nodes = 6
    r = Route(start, finish, no_nodes, no_nodes, 30000.0, craft)
    wind_fname = "/Users/thomasdickson/Documents/sail_routing/routing/domain_application/data_dir/wind_forecast.nc"
    diagram_path = "/Users/thomasdickson/Documents/sail_routing/python_routing/analysis/output/multiple"
    time = datetime(2014, 7, 1, 0, 0)
    x, y, land = return_co_ords(r.start.long, r.finish.long,
                                r.start.lat, r.finish.lat,
                                r.n_ranks, r.n_width, r.d_node)
    jt, et, pf_vals = min_time_calculate(r, wind_fname, time, craft)
    vt = datetime.fromtimestamp(jt) - time
    print("Journey time is: ", vt)
    plot_route(time, r, x, y, et, jt, pf_vals, diagram_path+"/test.png")


if __name__ == '__main__':
    run_simulation()
