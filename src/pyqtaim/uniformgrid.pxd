cimport numpy as np

cdef class UniformGrid:
    cdef double[:, ::1] points
    cdef double[:] xs, ys, zs
    cdef double[:] strip
    cdef readonly double weights
    cdef readonly long size
    cdef long[:] xyz_shape

    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_grid_point(self, int, int)
    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_distance(self, int, int)
