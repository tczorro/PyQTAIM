from libc.math cimport sqrt
from pyqtaim.uniformgrid cimport UniformGrid

import numpy as np
cimport numpy as np


cdef class Watershed:
    cdef public long[:] water_indices, basins
    cdef public int n_basins
    cdef public dict water_pt_basins, water_pt_wts
    cdef UniformGrid grid

    def __cinit__(self, UniformGrid grid):
        # self.water_indices = indices  # watershed pts index in original grid
        self.n_basins = 0
        self.basins = np.ones(grid.size, dtype=int) * -2  # basins number of all pts
        self.grid = grid
        self.water_pt_basins = {}
        self.water_pt_wts = {}

    cpdef long[:] sort_points(self, density_array):
        cdef long[:] args
        args = np.argsort(density_array)[::-1]
        return args

    cpdef void search_watershed_pts(self, double[:] target_array):
        # cdef np.ndarray[np.int_t, ndim=1] , nbh_basins
        cdef long[:] sorted_args, unique_basins, nbh_indices, wt_inds
        cdef int i, j, k, counter = 0, basin_ind
        cdef list nbh_basins, water_pt_indices = []
        cdef double[:] wts

        # index >= 0, basin index
        # index = -1, watershed pt
        # index = -2, not assigned

        sorted_args = self.sort_points(target_array)
        for i in sorted_args:
            nbh_basins = []
            nbh_indices = self.grid.neighbours_indices_of_grid_point(i, 26)
            for j in nbh_indices:
                basin_ind = self.basins[j]
                if basin_ind == -1:  # watershed point
                    wt_inds = self.water_pt_basins[j]
                    for k in wt_inds:
                        if k not in nbh_basins:
                            nbh_basins.append(k)
                elif basin_ind >= 0:
                    if basin_ind not in nbh_basins:
                        nbh_basins.append(basin_ind)
            unique_basins = np.array(nbh_basins, dtype=int)
            if len(unique_basins) == 1:
                self.basins[i] = unique_basins[0]
            elif len(unique_basins) >= 2:
                # compute fraction value of watershed pts
                wts = self.compute_weights_for_watershed_pts(i, unique_basins, target_array)
                self.basins[i] = -1  # set watershed index to -1
                self.water_pt_basins[i] = unique_basins
                self.water_pt_wts[i] = wts
                water_pt_indices.append(i)
            elif len(unique_basins) == 0:
                # new maximum
                self.basins[i] = counter
                counter += 1
        self.n_basins = counter

    cpdef double[:] compute_weights_for_watershed_pts(self,
                                                      int index,
                                                      long [:] unique_basins,
                                                      double[:] target_array
                                                      ):
        cdef int i, j, k, n_wt_pts, shape_1, shape_2, shape_3, n_nbhs, basin_val, n_basins
        # cdef long shape_1, shape_2, shape_3
        cdef double tmp, sum_diff=0
        cdef double[:] dist, values, diffs, Js, wts
        cdef long[:] nhbs_1, nhbs_2, nhbs_3, all_nhbs
        # cdef dict wts = {}, basin_wts

        # distance 1 nhbs
        nhbs_1 = self.grid.neighbours_indices_of_distance(index, 1)
        # distance 2 nhbs
        nhbs_2 = self.grid.neighbours_indices_of_distance(index, 2)
        # distance 3 nhbs
        nhbs_3 = self.grid.neighbours_indices_of_distance(index, 3)
        shape_1 = nhbs_1.shape[0]
        shape_2 = nhbs_2.shape[0]
        shape_3 = nhbs_3.shape[0]
        dist = np.zeros(shape_1 + shape_2 + shape_3, dtype=float)
        # all neighbour indices
        all_nhbs = np.zeros(dist.shape[0], dtype=int)
        # get all nearby neighbours
        for i in range(shape_1):
            dist[i] = 1.
            all_nhbs[i] = nhbs_1[i]
        for i in range(shape_2):
            dist[i + shape_1] = sqrt(2.)
            all_nhbs[i + shape_1] = nhbs_2[i]
        for i in range(shape_3):
            dist[i + shape_1 + shape_2] = sqrt(3.)
            all_nhbs[i + shape_1 + shape_2] = nhbs_3[i]

        # wts for given point for each basins
        n_nbhs = shape_1 + shape_2 + shape_3
        # wts = np.zeros(weights.shape[1], dtype=float)
        diffs = np.zeros(n_nbhs, dtype=float)
        Js = np.zeros(n_nbhs, dtype=float)
        for i in range(n_nbhs):
            pt_ind = all_nhbs[i]
            tmp = (target_array[pt_ind] - target_array[index]) / dist[i]
            if tmp > 0.:
                diffs[i] = tmp
                sum_diff += tmp

        wts = np.zeros(unique_basins.shape[0], dtype=float)

        for i in range(n_nbhs):
            # i: seq of all nbh pts
            # pt_ind: point index in the grid
            pt_ind = all_nhbs[i]
            Js[i] = diffs[i] / sum_diff
            basin_val = self.basins[pt_ind]
            # not a watershed point
            if basin_val >= 0:
                # j: seq of unique basins
                for j in range(unique_basins.shape[0]):
                    if basin_val == unique_basins[j]:
                        wts[j] += Js[i]
                        break
            # nb is a watershed pt
            elif basin_val == -1:
                its_basin = self.water_pt_basins[pt_ind]
                its_wts = self.water_pt_wts[pt_ind]
                # n_basins = len(wt_basin)
                for j in range(unique_basins.shape[0]):
                    for k in range(its_basin.shape[0]):
                        if its_basin[k] == unique_basins[j]:
                            wts[j] += Js[i] * its_wts[k]
                        # wts[j] = basin_wts[j] * Js[i]
        return wts



    cdef double[:] basin_wts(self, int basin_index):
        cdef double[:] basin_wts_value, wts
        cdef long[:] basins
        cdef int n_total_pt, i, j
        if basin_index >= self.n_basins:
            raise ValueError("index value is not valid")
        n_total_pt = self.grid.size
        basin_wts_value = np.zeros(n_total_pt, dtype=float)
        # loop over all points
        for i in range(n_total_pt):
            if self.basins[i] == basin_index:
                basin_wts_value[i] += 1.
            elif self.basins[i] == -1:
                basins = self.water_pt_basins[i]
                wts = self.water_pt_wts[i]
                for j in range(basins.shape[0]):
                    if basins[j] == basin_index:
                        basin_wts_value[i] += wts[j]
        return basin_wts_value

    def compute_basin_wts(self, basin_index):
        return np.asarray(self.basin_wts(basin_index))
