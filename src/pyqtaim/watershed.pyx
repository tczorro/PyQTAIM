from libc.math import sqrt
from pyqtaim.uniformgrid cimport UniformGrid
from scipy.sparse import lil_matrix

import numpy as np
cimport numpy as np


cdef class Watershed:
    cdef public long[:] water_indices, basins
    cdef dict water_pt_basins
    cdef public int n_basins
    cdef UniformGrid grid
    cdef object water_basin_wts

    def __cinit__(self, UniformGrid grid):
        # self.water_indices = indices  # watershed pts index in original grid
        self.n_basins = 0
        self.basins = np.ones(grid.size, dtype=int) * -1  # basins number of all pts
        self.grid = grid
        self.water_basin_wts = None  # {basin_index: points_wts}
        self.water_pt_basins = {}

    cpdef long[:] sort_points(self, density_array):
        cdef long[:] args
        args = np.argsort(density_array)[::-1]
        return args

    cpdef void search_watershed_pts(self, double[:] target_array):
        cdef np.ndarray[np.int_t, ndim=1] basins, nbh_indices
        cdef long[:] sorted_args
        cdef int i, j, counter = 0
        cdef list water_pt_indices = []

        # basins = np.zeros(grid.shape, dtype=int)
        sorted_args = self.sort_points(target_array)
        for i in sorted_args:
            # print(i)
            # nbh_indices = self.grid.neighbours_indices_of_grid_point(i, 6)
            # nbh_basins = np.array(self.basins)[nbh_indices]
            # # if check unique neighbour basins
            # print('nbh', nbh_basins)
            # unique_basins = np.unique(nbh_basins[nbh_basins >= 0])
            # print('uni', unique_basins)
            # if len(unique_basins) == 1:
            #     self.basins[i] = unique_basins[0]
            # elif len(unique_basins) >= 2:
            #     # add to watershed list
            #     self.water_pt_basins[i] = unique_basins
            #     water_pt_indices.append(i)
            # elif len(unique_basins) == 0:
            nbh_indices = self.grid.neighbours_indices_of_grid_point(i, 26)
            nbh_basins = np.array(self.basins)[nbh_indices]
            unique_basins = np.unique(nbh_basins[nbh_basins >= 0])
            if len(unique_basins) == 1:
                self.basins[i] = unique_basins[0]
            elif len(unique_basins) >= 2:
                # add to watershed list
                self.water_pt_basins[i] = unique_basins
                water_pt_indices.append(i)
            elif len(unique_basins) == 0:
                # new maximum
                counter += 1
                self.basins[i] = counter
        self.n_basins = counter
        self.water_indices = np.array(water_pt_indices, dtype=int)
        self.water_basin_wts = lil_matrix((len(self.water_pt_basins), self.n_basins))
        # print(self.water_basin_wts.toarray())

    cpdef double[:] compute_weights_for_all_watershed_pts(self, double[:] target_array):
        cdef int i, j, n_wt_pts, shape_1, shape_2, shape_3
        cdef double center_value
        cdef double[:] dist, values
        cdef long[:] nhbs_1, nhbs_2, nhbs_3, all_nhbs, basins
        n_wt_pts = self.water_indices.shape[0]
        for i in range(n_wt_pts):
            nhbs_1 = self.grid.neighbours_indices_of_distance(self.water_indices[i], 1)
            nhbs_2 = self.grid.neighbours_indices_of_distance(self.water_indices[i], 2)
            nhbs_3 = self.grid.neighbours_indices_of_distance(self.water_indices[i], 3)
            shape_1 = nhbs_1.shape[0]
            shape_2 = nhbs_2.shape[0]
            shape_3 = nhbs_3.shape[0]
            dist = np.zeros(shape_1 + shape_2 + shape_3, dtype=float)
            all_nhbs = np.zeros(dist.shape[0], dtype=int)
            # get all nearby neighbours
            for j in range(shape_1):
                dist[j] = 1.
                all_nhbs[j] = nhbs_1[j]
            for j in range(shape_2):
                dist[j + shape_1] = sqrt(2.)
                all_nhbs[j + shape_1] = nhbs_2[j]
            for j in range(shape_3):
                dist[j + shape_1 + shape_2] = sqrt(3.)
                all_nhbs[j + shape_1 + shape_2] = nhbs_3[j]
            # get value for each neighbours
            center_value = target_array[self.water_indices[i]]
            values = np.zeros(all_nhbs.shape[0], dtype=float)
            for j in range(all_nhbs.shape[0]):
                values[j] = target_array[all_nhbs[j]]
                basins[j] = self.basins[all_nhbs[j]]
            weights = self.water_basin_wts[np.asarray(self.water_indices), :].toarray()
            wts = compute_watershed_weights(center_value, values, dist, basins, weights)
            return wts

cpdef double[:] compute_watershed_weights(double x_rho,
                                          double[:] rho_prime,
                                          double[:] dists,
                                          long [:] basins,
                                          double[:, :] weights):
    cdef int n_basins, n_nbhs, i, j
    cdef double tmp, sum_diff=0
    cdef double[:] diffs, Js, wts
    # wts for given point for each basins
    n_nbhs = rho_prime.shape[0]
    wts = np.zeros(weights.shape[1], dtype=float)
    diffs = np.zeros(n_nbhs, dtype=float)
    n_nbhs = rho_prime.shape[0]
    n_basins = weights.shape[1]
    Js = np.zeros(n_nbhs, dtype=float)
    for i in range(n_nbhs):
        tmp = (rho_prime[i] - x_rho) / dists[i]
        if tmp > 0.:
            diffs[i] = tmp
            sum_diff += tmp
    for i in range(n_nbhs):
        Js[i] = diffs[i] / sum_diff
        basin_val = basins[i]
        # not a watershed point
        if basin_val >= 0:
            wts[basin_val] += Js[i]
        # nb is a watershed pt
        elif basin_val == -1:
            for j in range(weights.shape[1]):
                wts[j] = weights[i][j] * Js[i]
    return wts
