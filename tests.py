import unittest
from model_generator import Grid
from model_generator import ActiveGrid
from model_generator import RandomPoints
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

class TestModelGenerator(unittest.TestCase):

    def setUp(self):

        self.n_points = 30
        self.n_dim = 2
        self.random_points = RandomPoints(self.n_points, self.n_dim)
        self.active_grid = ActiveGrid(0, 0, 100, 100, 100, 100)


    def tearDown(self):

        self.random_points = None
        self.polygon = None
        self.active_grid = None

    def test_random_points(self):
        """
        Testing random points and convex hull generation
        """
        self.assertEqual(
            self.n_points,
            len(self.random_points.convex_hull.points)
        )

    # def test_active_grid(self):
    #     p=Polygon(self.polygon)
    #     print(p)
        # print(p.area)
        self.active_grid.give_grid_points(self.random_points.polygon)
        plt.imshow(self.active_grid.ibound, interpolation="nearest")
        plt.show()
        # print(self.active_grid.ibound)
        


if __name__ == '__main__':
    unittest.main()
