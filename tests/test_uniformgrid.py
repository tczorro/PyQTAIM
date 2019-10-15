from unittest import TestCase
from pyqtaim.uniformgrid import UniformGrid

import numpy as np
from numpy.testing import assert_array_equal


class TestUniformGrid(TestCase):

    def test_grid(self):
        for _ in range(10):
            x, y, z = np.random.randint(2, 10, 3)
            xs = np.arange(x, dtype=float)
            ys = np.arange(y, dtype=float)
            zs = np.arange(z, dtype=float)
            grid = UniformGrid(xs, ys, zs)
            assert grid.size == x * y * z
            assert grid.weights == 1.
            assert len(grid.points) == x * y * z

    def test_grid_six_neighbours(self):
        x, y, z = np.random.randint(5, 10, 3)
        xs = np.arange(x, dtype=float)
        ys = np.arange(y, dtype=float)
        zs = np.arange(z, dtype=float)
        grid = UniformGrid(xs, ys, zs)
        # 10 random tests
        for _ in range(10):
            index = np.random.randint(0, grid.size)
            nb_index = grid.neighbours_indices_of_grid_point(index, 6)
            assert 3 <= len(nb_index) <= 6
            dis = np.linalg.norm(grid.points - grid.points[index], axis=-1)
            dis_one_indices = np.where(dis == 1)[0]
            assert_array_equal(np.sort(nb_index), np.sort(dis_one_indices))
