import unittest
from model_generator import Grid
from model_generator import ActiveGrid
from model_generator import RandomPoints
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        xmin = 0
        xmax = 100
        ymin = 0
        ymax = 100
        nx = 100
        ny = 100
        self.n_points = 10
        self.n_dim = 2
        self.random_points = RandomPoints(
            xmin, ymin, xmax, ymax, nx, ny, self.n_points, self.n_dim
            )
        self.active_grid = ActiveGrid(xmin, ymin, xmax, ymax, nx, ny)
        self.active_grid.set_ibound(self.random_points.polygon)


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


        plt.imshow(self.active_grid.ibound, interpolation="nearest")
        plt.show()


if __name__ == '__main__':
    unittest.main()
