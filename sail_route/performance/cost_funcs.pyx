"""Cython functions for cost functions."""

import numpy as np
cimport numpy as np

cpdef float haversine(float lon1, float lat1, float lon2, float lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371.0 * 0.5399565
