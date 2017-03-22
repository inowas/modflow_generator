import unittest
import matplotlib.pyplot as plt
from .model_generator import ActiveGrid, VectorSource, ModelBoundary



class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        xmin = 0
        xmax = 100
        ymin = 0
        ymax = 50
        self.nx = 100
        self.ny = 50
        self.n_points = 10
        self.n_dim = 2
        self.num_of_bounds = 3
        self.vector_source = VectorSource(
            xmin, ymin, xmax, ymax, self.n_points, self.n_dim
            )
        self.active_grid = ActiveGrid(xmin, ymin, xmax, ymax, self.nx, self.ny)
        self.model_boundary = ModelBoundary(self.active_grid)

    def tearDown(self):

        self.vector_source = None
        self.active_grid = None
        self.model_boundary = None

    def test_random_points(self):
        """
        Testing random points and convex hull generation
        """
        self.assertEqual(
            self.n_points,
            len(self.vector_source.convex_hull.points)
        )


    def test_boundary_ibound(self):
        """ Vaidation of ibound and boundary """
        self.active_grid.set_ibound(self.vector_source.polygon)

        # If ibound cell is 0, boundary cell has to be 0.
        # If bondary cell is 1, ibound cell has to be one.
        for i in range(self.ny):
            for j in range(self.nx):
                if self.active_grid.ibound[i][j] == 0:
                    self.assertEqual(self.active_grid.bound_ibound[i][j], 0)
                if self.active_grid.bound_ibound[i][j] == 1:
                    self.assertEqual(self.active_grid.ibound[i][j], 1)

        # plt.imshow(self.active_grid.ibound, interpolation="nearest")
        # plt.show()

    def test_boundaries(self):
        """ Validation of model boundaries """
        line_segments = self.model_boundary.set_line_segments(self.vector_source.polygon)

        self.assertEqual(len(line_segments), len(self.vector_source.polygon)-1)

        self.model_boundary.set_chd_spd(self.num_of_bounds)

        self.assertEqual(sum(self.model_boundary.nums_of_segments), len(line_segments))
        self.assertEqual(len(self.model_boundary.nums_of_segments), self.num_of_bounds)
        


if __name__ == '__main__':
    unittest.main()
