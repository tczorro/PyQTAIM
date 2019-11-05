from unittest import TestCase

from pyqtaim.uniformgrid import UniformGrid, UniformGrid3D
from pyqtaim.watershed import Watershed
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal


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
        indices = np.asarray(wt_obj.sort_points(value))
        # sort in descending sequence
        for i in range(len(indices) - 1):
            value[i] > value[i + 1]
        ref_ind = np.argsort(value)[::-1]
        assert_array_equal(indices, ref_ind)

    def test_watershed_pts_3d(self):
        x = np.arange(10, dtype=float)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2 / 30, axis=-1)))

        value_array = gauss(grid.points, [0, 0, 0]) + gauss(grid.points, [7, 8, 9])
        wt_obj = Watershed(grid)
        wt_obj.search_watershed_pts(value_array)
        assert wt_obj.n_basins == 2

    def test_compute_weights_for_watershed_pts(self):
        x = np.arange(10, dtype=float)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2 / 25, axis=-1)))

        value_array = gauss(grid.points, [0, 0, 0]) + gauss(grid.points, [7, 8, 9])
        wt_obj = Watershed(grid)
        wt_obj.search_watershed_pts(value_array)
        assert wt_obj.n_basins == 2
        # wt_obj.compute_weights_for_all_watershed_pts(value_array)
        # print(wt_obj.water_basin_wts.toarray())
        for i in range(grid.size):
            basin = wt_obj.basins[i]
            assert basin != -2
            if basin == -1:
                bs_vals = wt_obj.water_pt_basins[i]
                wts = wt_obj.water_pt_wts[i]
                assert len(np.asarray(bs_vals)) == len(np.asarray(wts))
                assert_almost_equal(np.sum(np.asarray(bs_vals)), 1)

    def test_basin_wts(self):
        x = np.arange(10, dtype=float)
        n_x = len(x)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2 / 10, axis=-1)))

        for _ in range(5):
            center = np.random.rand(3) * 3
            value_array = gauss(grid.points, center) + gauss(grid.points, [9., 9., 9.] - center)
            wt_obj = Watershed(grid)
            wt_obj.search_watershed_pts(value_array)
            assert wt_obj.n_basins == 2
            basin_wt1 = wt_obj.compute_basin_wts(0)
            basin_wt2 = wt_obj.compute_basin_wts(1)
            assert_allclose(basin_wt1 + basin_wt2, np.ones(grid.size))
            sum1 = np.sum(basin_wt1 * value_array)
            sum2 = np.sum(basin_wt2 * value_array)
            assert_almost_equal(sum1, sum2)

    def test_basin_wts2(self):
        x = np.linspace(0., 2., 11)
        n_x = len(x)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2, axis=-1)))

        for _ in range(5):
            center = np.random.rand(3) * 0.5
            value_array = gauss(grid.points, center) + gauss(grid.points, [2., 2., 2.] - center)
            wt_obj = Watershed(grid)
            wt_obj.search_watershed_pts(value_array)
            assert wt_obj.n_basins == 2
            basin_wt1 = wt_obj.compute_basin_wts(0)
            basin_wt2 = wt_obj.compute_basin_wts(1)
            assert_allclose(basin_wt1 + basin_wt2, np.ones(grid.size))
            sum1 = np.sum(basin_wt1 * value_array)
            sum2 = np.sum(basin_wt2 * value_array)
            assert_almost_equal(sum1, sum2)


    def test_basin_wts_near(self):
        x = np.arange(10, dtype=float)
        n_x = len(x)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2 / 20, axis=-1)))

        for _ in range(5):
            center = np.random.rand(3) * 3
            value_array = gauss(grid.points, center) + gauss(grid.points, [9., 9., 9.] - center)
            wt_obj = Watershed(grid)
            wt_obj.search_watershed_pts_near(value_array)
            assert wt_obj.n_basins == 2
            basin_wt1 = wt_obj.compute_basin_wts(0)
            basin_wt2 = wt_obj.compute_basin_wts(1)
            assert_allclose(basin_wt1 + basin_wt2, np.ones(grid.size))
            sum1 = np.sum(basin_wt1 * value_array)
            sum2 = np.sum(basin_wt2 * value_array)
            assert_almost_equal(sum1, sum2)

    def test_basin_wts_near2(self):
        x = np.linspace(0., 3., 10)
        # x = np.arange(10, dtype=float)
        n_x = len(x)
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2, axis=-1)))

        for _ in range(5):
            center = np.random.rand(3) * 0.4
            value_array = gauss(grid.points, center) + gauss(grid.points, [3., 3., 3.] - center)
            wt_obj = Watershed(grid)
            wt_obj.search_watershed_pts_near(value_array)
            assert wt_obj.n_basins == 2
            basin_wt1 = wt_obj.compute_basin_wts(0)
            basin_wt2 = wt_obj.compute_basin_wts(1)
            # assert_allclose(basin_wt1 + basin_wt2, np.ones(grid.size))
            sum1 = np.sum(basin_wt1 * value_array)
            sum2 = np.sum(basin_wt2 * value_array)
            assert_almost_equal(sum1, sum2)
        # print(np.where(basin_wt1 + basin_wt2 == 0))
        # print(np.round(basin_wt1.reshape(n_x, n_x, n_x), 3))
        # print('=======')
        # print(np.round(basin_wt2.reshape(n_x, n_x, n_x), 3))
        # print('=======')
        # print(np.round(value_array.reshape(n_x, n_x, n_x), 3))
        # # assert_almost_equal(sum1, sum2)
        # print('=======')
        # index_seq = np.array(wt_obj.sort_points(value_array))
        # tot = n_x ** 3
        # seq_check = np.zeros(tot)
        # for i in range(tot):
        #     seq_check[index_seq[i]] = i
        # print(seq_check.reshape(n_x, n_x, n_x))
        # assert False
    '''
    def test_water_shed_weights(self):
        rho_center = 0.24
        rhos_prime = np.array([0.21, 0.21, 0.29, 0.29])
        dis = np.array([1.0, 1.0, 1.0, 1.0])
        basins = np.array([0, 1, 0, 1])
        water_weights = np.zeros((4, 2))
        # grid and watershed object
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis, basins, water_weights
        )
        result = np.asarray(wts)
        assert_allclose(result, [0.5, 0.5])

        # only one type basins
        basins2 = np.array([-1, -1, 0, 0])
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis, basins2, water_weights
        )
        result = np.asarray(wts)
        assert_allclose(result, [1, 0])

        # only one type basins
        basins3 = np.array([-1, -1, 1, 1])
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis, basins3, water_weights
        )
        result = np.asarray(wts)
        assert_allclose(result, [0, 1])

        # only two type basins
        basins4 = np.array([-1, 0, 0, 1])
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis, basins4, water_weights
        )
        result = np.asarray(wts)
        assert_allclose(result, [0.5, 0.5])

    def test_water_shed_weights_2(self):
        rho_center = 0.32
        rhos_prime = np.array([0.35, 0.42, 0.39, 0.38])
        dis = np.array([1., 1., 1., 1.])
        basins = np.array([-1, 0, 0, 1])
        water_weights = np.zeros((4, 2))
        water_weights[0] = np.array([0.3, 0.7])
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis, basins, water_weights
        )
        ref_wts = np.array([17. / 26, 6. / 26])
        ref_wts += np.array([3 / 26 * 0.3, 3 / 26 * .7])
        assert_allclose(np.asarray(wts), ref_wts)

        dis_2 = np.array([1., 1.414, 1., 1.414])
        wts = compute_watershed_weights(
            rho_center, rhos_prime, dis_2, basins, water_weights
        )
        flows = (rhos_prime - rho_center) / dis_2
        ratio = flows / np.sum(flows)
        ref_wts = np.array([ratio[1] + ratio[2], ratio[3]])
        ref_wts += np.array([ratio[0] * 0.3, ratio[0] * 0.7])
        assert_allclose(np.asarray(wts), ref_wts)

    def test_basin_wts(self):
        x = np.arange(0, 2.0, 0.05, dtype=float)
        n_x = len(x)
        tot = n_x ** 3
        # generate 3D grid
        grid = UniformGrid3D(x, x, x)

        def gauss(coors_xyz, center=[0, 0, 0]):
            center = np.array(center)
            return np.exp(-(np.sum((coors_xyz - center) ** 2, axis=-1)))
        value_array = gauss(grid.points, [0, 0, 0]) + gauss(grid.points, [1.95, 1.95, 1.95])

        wt_obj = Watershed(grid)
        wt_obj.search_watershed_pts(value_array)
        wt_obj.compute_weights_for_all_watershed_pts(value_array)
        basin_wt1 = wt_obj.compute_basin_wts(0)
        basin_wt2 = wt_obj.compute_basin_wts(1)
        assert_allclose(basin_wt1 + basin_wt2, np.ones(grid.size))
        sum1 = np.sum(basin_wt1 * value_array)
        sum2 = np.sum(basin_wt2 * value_array)
        print(sum1, sum2)
        print(np.round(basin_wt1.reshape(n_x, n_x, n_x), 3))
        print('=======')
        print(np.round(basin_wt2.reshape(n_x, n_x, n_x), 3))
        print('=======')
        print(np.round(value_array.reshape(n_x, n_x, n_x), 3))
        # # assert_almost_equal(sum1, sum2)
        print('=======')
        index_seq = np.array(wt_obj.sort_points(value_array))
        seq_check = np.zeros(tot)
        for i in range(tot):
            seq_check[index_seq[i]] = i
        print(seq_check.reshape(n_x, n_x, n_x))
        assert False
    '''
