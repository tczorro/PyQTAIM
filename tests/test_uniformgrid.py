from unittest import TestCase
from pyqtaim.uniformgrid import UniformGrid, UniformGrid3D

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


class TestUniformGrid(TestCase):
    def test_grid(self):
        for _ in range(10):
            x, y, z = np.random.randint(2, 10, 3)
            xs = np.arange(x, dtype=float)
            ys = np.arange(y, dtype=float)
            zs = np.arange(z, dtype=float)
            grid = UniformGrid3D(xs, ys, zs)
            assert grid.size == x * y * z
            assert grid.weights == 1.0
            assert len(grid.points) == x * y * z

    def test_grid_6_neighbours(self):
        x, y, z = np.random.randint(5, 10, 3)
        xs = np.arange(x, dtype=float)
        ys = np.arange(y, dtype=float)
        zs = np.arange(z, dtype=float)
        grid = UniformGrid3D(xs, ys, zs)
        # 10 random tests
        for _ in range(10):
            index = np.random.randint(0, grid.size)
            nb_index = grid.neighbours_indices_of_grid_point(index, 6)
            assert 3 <= len(nb_index) <= 6
            dis = np.linalg.norm(grid.points - grid.points[index], axis=-1)
            dis_one_indices = np.where(dis == 1)[0]
            assert_array_equal(np.sort(nb_index), np.sort(dis_one_indices))

    def test_grid_26_neighbours(self):
        x, y, z = np.random.randint(5, 10, 3)
        xs = np.arange(x, dtype=float)
        ys = np.arange(y, dtype=float)
        zs = np.arange(z, dtype=float)
        grid = UniformGrid3D(xs, ys, zs)
        # 10 random tests
        for _ in range(10):
            index = np.random.randint(0, grid.size)
            nb_index = grid.neighbours_indices_of_grid_point(index, 26)
            assert 7 <= len(nb_index) <= 26
            dis = np.linalg.norm(grid.points - grid.points[index], axis=-1)
            dis_one_indices = np.intersect1d(np.where(dis < 1.74), np.where(dis > 0.1))
            assert_array_equal(np.sort(nb_index), np.sort(dis_one_indices))

    def test_grid_6_neighbours_and_neighbour_distance(self):
        for _ in range(10):
            x, y, z = np.random.randint(5, 10, 3)
            xs = np.arange(x, dtype=float)
            ys = np.arange(y, dtype=float)
            zs = np.arange(z, dtype=float)
            grid = UniformGrid3D(xs, ys, zs)
            for _ in range(10):
                index = np.random.randint(0, grid.size)
                # test 6 neighbours
                nbh_pt = grid.neighbours_indices_of_grid_point(index, 6)
                nhb_dis = grid.neighbours_indices_of_distance(index, 1)
                assert_array_equal(np.sort(nbh_pt), np.sort(nhb_dis))
            for _ in range(10):
                # test nearby 12 neighbours
                index = np.random.randint(0, grid.size)
                nbh_ids = grid.neighbours_indices_of_distance(index, 2)
                dis = np.linalg.norm(grid.points[nbh_ids] - grid.points[index], axis=-1)
                assert_allclose(dis - np.sqrt(2), np.zeros(len(dis)))
                # test contain all points
                dis_all = np.linalg.norm(grid.points - grid.points[index], axis=-1)
                ref_ind = np.arange(len(grid.points))[np.isclose(dis_all, np.sqrt(2))]
                assert_array_equal(np.sort(nbh_ids), np.sort(ref_ind))
            for _ in range(10):
                # test nearby 8 neighbours
                index = np.random.randint(0, grid.size)
                nbh_ids = grid.neighbours_indices_of_distance(index, 3)
                dis = np.linalg.norm(grid.points[nbh_ids] - grid.points[index], axis=-1)
                assert_allclose(dis - np.sqrt(3), np.zeros(len(dis)))
                # test contain all points
                dis_all = np.linalg.norm(grid.points - grid.points[index], axis=-1)
                ref_ind = np.arange(len(grid.points))[np.isclose(dis_all, np.sqrt(3))]
                assert_array_equal(np.sort(nbh_ids), np.sort(ref_ind))
