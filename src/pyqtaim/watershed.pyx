from pyqtaim.uniformgrid cimport UniformGrid
from scipy.sparse import lil_matrix

import numpy as np
cimport numpy as np



cdef class Watershed:
    cdef long[:] water_indices, basins
    cdef dict water_pt_basins
    cdef int n_basins
    cdef UniformGrid gridUniformGrid
    cdef object water_basin_wts

    def __cinit__(self, long[:] indices, long[:] basins, int n_basins, UniformGrid grid):
        # self.water_indices = indices  # watershed pts index in original grid
        self.n_basins = n_basins
        self.basins = np.ones(grid.size, dtype=int) * -1  # basins number of all pts
        self.grid = grid
        self.water_basin_wts = None  # {basin_index: points_wts}
        self.water_pt_basins = {}

    cpdef np.ndarray[np.int_t, ndim=1] sort_points(self, density_array):
        cdef np.ndarray[np.int_t, ndim=1] args
        args = np.argsort(density_array)[::-1]
        return args

    cpdef dict search_watershed_pts(self, UniformGrid grid, double[:] target_array):
        cdef np.ndarray[np.int_t, ndim=1] basins, nbh_indices, sorted_args
        # cdef long[:] sorted_args
        cdef int i, j, basin_counter = 0
        cdef list water_pt_indices = []

        # basins = np.zeros(grid.shape, dtype=int)
        sorted_args = self.sort_points(target_array)
        for i in sorted_args:
            nbh_indices = grid.neighbours_indices_of_grid_point(i, 6)
            nbh_basins = self.basins[nbh_indices]
            # if check unique neighbour basins
            unique_basins = np.unique(nbh_basins[nbh_basins >= 0])
            if unique_basins == 1:
                self.basins[i] = unique_basins[0]
            elif unique_basins >= 2:
                # add to watershed list
                self.water_pt_basins[i] = unique_basins
                water_pt_indices.append(i)
            elif unique_basins == 0:
                nbh_indices = grid.neighbours_indices_of_grid_point(i, 26)
                nbh_basins = self.basins[nbh_indices]
                unique_basins = np.unique(nbh_basins[nbh_basins > 0])
                if unique_basins == 1:
                    self.basins[i] = unique_basins[0]
                elif unique_basins >= 2:
                    # add to watershed list
                    self.water_pt_basins[i] = unique_basins
                    water_pt_indices.append(i)
                elif unique_basins == 0:
                    # new maximum
                    self.basins[i] = basin_counter
                    basin_counter += 1
        self.water_indices = np.array(water_pt_indices, dtype=int)
        self.water_basin_wts = lil_matrix((len(self.water_pt_basins), basin_counter))


def compute_watershed_weights(x_rho, rho_prime, weights):
    n_basins = weights.shape[1]
    diff = (rho_prime - x_rho).clip(min=0)
    J = diff / np.sum(diff)
    basin_wts = np.sum(J[:, None] * weights, axis=0)
    return basin_wts
