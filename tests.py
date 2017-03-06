import unittest
import matplotlib.pyplot as plt
from model_generator import Grid, ActiveGrid, RandomPoints



class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        xmin = 0
        xmax = 100
        ymin = 0
        ymax = 100
        self.nx = 10
        self.ny = 10
        self.n_points = 10
        self.n_dim = 2
        self.random_points = RandomPoints(
            xmin, ymin, xmax, ymax, self.n_points, self.n_dim
            )
        self.active_grid = ActiveGrid(xmin, ymin, xmax, ymax, self.nx, self.ny)
        


    def tearDown(self):

        self.random_points = None
        self.active_grid = None

    def test_random_points(self):
        """
        Testing random points and convex hull generation
        """
        self.assertEqual(
            self.n_points,
            len(self.random_points.convex_hull.points)
        )

    
    def test_boundary_ibound(self):
        """ Vaidation of ibound and boundary """
        self.active_grid.set_ibound(self.random_points.polygon)

        # plt.imshow(self.active_grid.ibound, interpolation="nearest")
        # plt.show()
        # plt.imshow(self.active_grid.bound_ibound, interpolation="nearest")
        # plt.show()

        # If ibound cell is 0, boundary cell has to be 0.
        # If bondary cell is 1, ibound cell has to be one.
        for i in range(self.ny):
            for j in range(self.nx):
                if self.active_grid.ibound[i][j] == 0:
                    self.assertEqual(self.active_grid.bound_ibound[i][j], 0)
                if self.active_grid.bound_ibound[i][j] == 1:
                    self.assertEqual(self.active_grid.ibound[i][j], 1)


if __name__ == '__main__':
    unittest.main()
