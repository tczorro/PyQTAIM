from libc.math cimport sqrt
import numpy as np
cimport numpy as np

cdef class UniformGrid3D(UniformGrid):
    # cdef double[:, ::1] points
    # cdef double[:] xs, ys, zs
    # cdef double weights
    # cdef long[:] xyz_shape
    # cdef long size

    def __cinit__(self, double[:] xs, double[:] ys, double[:] zs=None):
        self.size = xs.shape[0] * ys.shape[0] * zs.shape[0]
        self.xyz_shape = np.array([xs.shape[0], ys.shape[0], zs.shape[0]])
        self.strip = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
        self.weights = self.strip[0] * self.strip[1] * self.strip[2]
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.points = np.zeros((self.size, 3), dtype=float)
        # this part can be done by numpy meshgrid
        cdef int i, j, k, index=0
        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                for k in range(zs.shape[0]):
                    self.points[index][0] += xs[i]
                    self.points[index][1] += ys[j]
                    self.points[index][2] += zs[k]
                    index += 1


    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_distance(self, int index, int num_pt):
        cdef long[:] nhb_pts_ind
        cdef long nhb_ind
        cdef double tmp, x, y, z

        if num_pt == 1:
            return self.neighbours_indices_of_grid_point(index, 6)
        nhb_pts_ind = self.neighbours_indices_of_grid_point(index, 26)

        cdef list tmp_ind = []

        if num_pt == 2:
            for nhb_ind in nhb_pts_ind:
                x = self.points[nhb_ind][0] - self.points[index][0]
                y = self.points[nhb_ind][1] - self.points[index][1]
                z = self.points[nhb_ind][2] - self.points[index][2]
                dis = sqrt(x**2 + y**2 + z**2)
                if 1.4 < dis < 1.42:
                    tmp_ind.append(nhb_ind)
        elif num_pt == 3:
            for nhb_ind in nhb_pts_ind:
                x = self.points[nhb_ind][0] - self.points[index][0]
                y = self.points[nhb_ind][1] - self.points[index][1]
                z = self.points[nhb_ind][2] - self.points[index][2]
                dis = sqrt(x**2 + y**2 + z**2)
                if 1.7 < dis:
                    tmp_ind.append(nhb_ind)
        return np.array(tmp_ind)


    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_grid_point(self, int index, int neighbours):
        cdef int x_shape, y_shape, z_shape, i, j, k
        # cdef int flag_x=0, flag_y=0, flag_z=0 # 0 not boundary
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

cdef class UniformGrid:
    def __cinit__(self, double[:] xs, double[:] ys, double[:] zs=None):
        self.size = xs.shape[0] * ys.shape[0]
        self.xyz_shape = np.array([xs.shape[0], ys.shape[0]])
        self.strip = np.array([xs[1] - xs[0], ys[1] - ys[0]])
        self.weights = self.strip[0] * self.strip[1]
        self.xs = xs
        self.ys = ys
        self.points = np.zeros((self.size, 2), dtype=float)
        # this part can be done by numpy meshgrid
        cdef int i, j, index=0
        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                self.points[index][0] += xs[i]
                self.points[index][1] += ys[j]
                index += 1

    @property
    def points(self):
        return np.asarray(self.points)

    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_distance(self, int index, int num_pt):
        cdef long[:] nhb_pts_ind
        cdef long nhb_ind
        cdef double tmp, x, y

        if num_pt == 1:
            return self.neighbours_indices_of_grid_point(index, 4)
        nhb_pts_ind = self.neighbours_indices_of_grid_point(index, 8)

        cdef list tmp_ind = []

        if num_pt == 2:
            for nhb_ind in nhb_pts_ind:
                x = self.points[nhb_ind][0] - self.points[index][0]
                y = self.points[nhb_ind][1] - self.points[index][1]
                dis = sqrt(x**2 + y**2)
                if 1.4 < dis < 1.42:
                    tmp_ind.append(nhb_ind)
        return np.array(tmp_ind)


    cpdef np.ndarray[np.int_t, ndim=1] neighbours_indices_of_grid_point(self, int index, int neighbours):
        cdef int x_shape, y_shape, i, j
        cdef list nhb_ind_list=[]
        cdef np.ndarray[np.int_t,ndim=1] nhb_ind_array
        # 1 reach the upper boundary, -1 reach the lower boundary
        x_shape = self.xyz_shape[0]
        y_shape = self.xyz_shape[1]
        x = index // y_shape
        y = index % y_shape
        # check boundary
        if neighbours == 6:
            # add x - 1
            if x != 0:
                nhb_ind_list.append(index - y_shape)
            # add x + 1
            if x != x_shape -1:
                nhb_ind_list.append(index + y_shape)

            # add y - 1
            if y != 0:
                nhb_ind_list.append(index - 1)
            # add y + 1
            if y != y_shape -1:
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
            for i in xlist:
                for j in ylist:
                    add_ind = (y_shape) * i + j
                    if add_ind != 0:
                        nhb_ind_list.append(index + add_ind)
        else:
            raise ValueError(f"Input neighbours is not valid, got {neighbours}")

        nhb_ind_array = np.asarray(nhb_ind_list)
        return nhb_ind_array
