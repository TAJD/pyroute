""" Shortest path calculation.

Functions to assist with returning the shortest path for a given route.

Thomas Dickson
14/05/2018
thomas.dickson@soton.ac.uk
"""

import numpy as np
from numba import jit


@jit
def shortest_path(indx, pindx, sp):
    """Create a list of the nodes visited on the shortest path."""
    ix = np.argwhere(indx == sp[-1])
    pix = pindx[ix[0][0], ix[0][1]]
    sp.append(pix)
    if pix == -1:
        return np.array(sp)
    else:
        return shortest_path(indx, pindx, sp)


@jit
def get_locs(indx, sp, x_locs, y_locs):
    """Get the locations of the points on the shortest path."""
    X = []
    Y = []
    for k in sp[:-1]:
        i, j = np.where(indx == k)
        X.append(x_locs[i, j].view())
        Y.append(y_locs[i, j].view())
    return X, Y
