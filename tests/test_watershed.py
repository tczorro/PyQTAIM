from unittest import TestCase

from pyqtaim.uniformgrid import UniformGrid, UniformGrid3D
from pyqtaim.watershed import Watershed
import numpy as np
from numpy.testing import assert_array_equal


class TestWatershed(TestCase):

    def test_Watershed_init(self):
        x = np.arange(10, dtype=float)
        grid = UniformGrid3D(x, x, x)
        # initialize watershed obj
        Watershed(grid)

    def test_sort_points(self):
        x = np.arange(5, dtype=float)
        grid = UniformGrid3D(x, x, x)
        wt_obj = Watershed(grid)
        # f = x - y + z
        value = grid.points[:, 0] - grid.points[:, 1] + grid.points[:, 2]
        indices = wt_obj.sort_points(value)
        # sort in descending sequence
        for i in range(len(indices) - 1):
            value[i] > value[i + 1]
        ref_ind = np.argsort(value)[::-1]
        assert_array_equal(indices, ref_ind)

    def test_watershed_pts(self):
        x = np.arange(10, dtype=float)
        # generate 2D grid
        grid = UniformGrid(x, x)

        def gauss(coors_xyz, center=[0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2 / 20, axis=-1)))
        value_array = gauss(grid.points, [0, 0]) + gauss(grid.points, [8, 9])

        disp_array = np.around(value_array, 2)
        for i in range(10):
            print(disp_array[i * 10: i * 10 + 10])
        wt_obj = Watershed(grid)
        wt_obj.search_watershed_pts(value_array)
        assert wt_obj.n_basins == 2
        # print(np.array(wt_obj.water_indices))

    def test_water_shed_weights(self):
        ...
