"""Developing shortest path identification algorithm"""

import numpy as np


def gen_indx(x_locs):
    """Return the indexes for each node to be iterated over.

    Uses the x_locs array as an array to mimic."""
    n_elem = x_locs.shape[0] * x_locs.shape[1]
    indx = np.arange(n_elem).reshape(x_locs.shape[0], x_locs.shape[1])
    pindx = np.zeros_like(indx)
    pindx[:] = -1
    return indx, pindx


def gen_sample_data():
    return np.array([[-1, 3, 1], [-1, 3, 1], [-1, 6, 7]])


def shortest_path(indx, pindx, sp):
    """Create a list of the nodes visited on the shortest path."""
    ix = np.argwhere(indx == sp[-1])
    pix = pindx[ix[0][0], ix[0][1]]
    sp.append(pix)
    if pix == -1:
        return sp
    else:
        return shortest_path(indx, pindx, sp)


if __name__ == '__main__':
    sd = gen_sample_data()
    indxs, pindxs = gen_indx(sd)
    # print("Inputs to shortest path routine")
    # print(sd)
    # print(indxs)
    # print(pindxs)
    spr = shortest_path(indxs, sd, [5])
    print("Outputs")
    print(spr)
