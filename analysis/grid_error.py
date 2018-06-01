"""Calculating grid convergence error.

Based on: Procedure for Estimation and Reporting of Uncertainty Due to
Discretization
in CFD Applications

Thomas Dickson
thomas.dickson@soton.ac.uk
04/05/2018
"""

from context import sail_route
import numpy as np
import matplotlib.pyplot as plt
from sail_route.performance.cost_function import haversine
from sail_route.sail_routing import Location


def calc_h(N, Delta_A):
    """Calculate h."""
    area = np.array([Delta_A for i in range(N)])
    return np.sqrt((1/N * np.sum(area)))


def return_h(no_nodes, p1, p2):
    """
    Return h for a specific route.

    no_nodes is the number of nodes along one side of the grid.
    """
    dist, bearing = haversine(p1.long, p1.lat, p2.long, p2.lat)
    no_nodes = no_nodes**2
    node_distance = (dist/0.5399565)/no_nodes
    return calc_h(no_nodes, node_distance**2)


if __name__ == '__main__':
    p1 = Location(-149.426, -17.651)
    p2 = Location(-139.33, -9)
    N = np.arange(10, 50, 2)
    H = [return_h(n, p1, p2) for n in N]
    plt.figure()
    plt.plot(N, H)
    plt.title("Tahiti to Marquesas")
    plt.ylabel("h")
    plt.xlabel("No nodes along edge")
    plt.savefig("h_vals.png")
