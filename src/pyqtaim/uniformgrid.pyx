import numpy as np
cimport numpy as np

cdef class UniformGrid:
    cdef double[:, ::1] points
    cdef double[:] xs, ys, zs
    cdef double weights
    cdef long[:] xyz_shape
    cdef long size

    def __cinit__(self, double[:] xs, double[:] ys, double[:] zs):
        self.size = xs.shape[0] * ys.shape[0] * zs.shape[0]
        self.xyz_shape = np.array([xs.shape[0], ys.shape[0], zs.shape[0]])
        self.weights = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.points = np.zeros((self.size, 3), dtype=float)
        # this part can be done by numpy meshgrid
        cdef int index=0
        cdef double i, j, k
        for i in xs:
            for j in ys:
                for k in zs:
                    self.points[index][0] += i
                    self.points[index][1] += j
                    self.points[index][2] += k
                    index += 1

    @property
    def points(self):
        return np.asarray(self.points)

    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_grid_point(
            self,
            int index,
            int neighbours=6
        ):
        cdef int x_shape, y_shape, z_shape, i, j, k
        cdef int flag_x=0, flag_y=0, flag_z=0 # 0 not boundary
        cdef list nhb_ind_list=[]
        cdef np.ndarray[np.int_t,ndim=1] nhb_ind_array
        # 1 reach the upper boundary, -1 reach the lower boundary
        x_shape = self.xyz_shape[0]
        y_shape = self.xyz_shape[1]
        z_shape = self.xyz_shape[2]
        x = index // (z_shape * y_shape)
        y = (index % (z_shape * y_shape))// (z_shape)
        z = index % self.zs.shape[0]
        # check boundary
        if neighbours == 6:
            # add x - 1
            if x != 0:
                nhb_ind_list.append(index - (z_shape * y_shape))
            # add x + 1
            if x != x_shape -1:
                nhb_ind_list.append(index + (z_shape * y_shape))

            # add y - 1
            if y != 0:
                nhb_ind_list.append(index - z_shape)
            # add y + 1
            if y != y_shape -1:
                nhb_ind_list.append(index + z_shape)

            # add z - 1
            if z != 0:
                nhb_ind_list.append(index - 1)
            # add z + 1
            if z != z_shape -1:
                nhb_ind_list.append(index + 1)

        # 26 surrounding neighbours
        elif neighbours == 26:
            xlist = []
            ylist = []
            zlist = []
            # add x - 1
            if x != 0:
                xlist.append(-1)
            # add x + 1
            if x != x_shape - 1:
                xlist.append(1)
            xlist.append(0)

            # add y - 1
            if y != 0:
                ylist.append(-1)
            # add y + 1
            if y != y_shape -1:
                ylist.append(1)
            ylist.append(0)

            # add z - 1
            if z != 0:
                zlist.append(-1)
            # add z + 1
            if z != z_shape -1:
                zlist.append(1)
            zlist.append(0)
            for i in xlist:
                for j in ylist:
                    for k in zlist:
                        add_ind = (z_shape * y_shape) * i + z_shape * j + k
                        if add_ind != 0:
                            nhb_ind_list.append(index + add_ind)
        else:
            raise ValueError(f"Input neighbours is not valid, got {neighbours}")

        nhb_ind_array = np.asarray(nhb_ind_list)
        return nhb_ind_array
