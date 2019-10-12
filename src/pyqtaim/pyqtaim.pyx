from libc.math cimport sqrt
import numpy as np
# cimport numpy as np



cdef class WatershedWeights:
    cdef double[:] indices
    cdef dict basin_weights

    def __cint__(self, double[:] indices):
        self.indices = indices  # watershed pts index in original grid
        self.basins_weights = {}  # {basin_index: points_wts}


def sort_points(density_array):
    args = np.argsort(density_array)[::-1]
    return args

def search_watershed_pts(grid, target_fun, grad_fun):
    cdef int[:] basins, sorted_args
    cdef int i, j, counter=0

    basins = np.zeros(grid.shape, dtype=float)
    sorted_args = sort_points(target_fun(grid))
    for i in sorted_args:
        ...

