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
        return sp
    else:
        return shortest_path(indx, pindx, sp)
