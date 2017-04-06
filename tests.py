import unittest
import numpy as np
import matplotlib.pyplot as plt
from model_generator import ActiveGrid, ModelTime, DataSource, ModelBoundary, VectorSource


class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 50
        self.nper = 200
        self.perlen = [100]
        self.nstp = [100]
        self.steady = True
        self.nx = 100
        self.ny = 50
        self.n_points = 10
        self.n_dim = 2
        self.b_types = {
            'NFL':{},
            'CHD': {'min': 100, 'max': 1000,
                    'periods': [3, 5, 6, 10]},
            'RIV': {'min': 6, 'max': 10,
                    'periods': [3, 4, 6, -1]}
            }

        self.vector_source = VectorSource(
            self.xmin, self.ymin, self.xmax, self.ymax, self.n_points, self.n_dim
            )
        self.data_source = DataSource(
            self.nper,
            self.b_types
        )
        self.active_grid = ActiveGrid(
            self.xmin, self.ymin, self.xmax, self.ymax, self.nx, self.ny
            )
        self.model_time = ModelTime(
            self.nper, self.perlen, self.nstp, self.steady
            )
        self.model_boundary = ModelBoundary(
            self.active_grid, self.model_time, self.data_source
            )

        self.active_grid.set_ibound(self.vector_source.polygon)
        self.data_source.generate_data()
        self.model_boundary.set_line_segments(
            self.vector_source.polygon
            )
        self.model_boundary.set_boundaries_spd()

    def tearDown(self):

        self.vector_source = None
        self.data_source = None
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

    def test_random_data(self):
        """ Test data source generation"""
        self.assertEqual(len(self.data_source.b_data), len(self.b_types))
        for key in self.b_types:
            if key != 'NFL':
                self.assertEqual(
                    len(self.data_source.b_data[key]), self.nper
                    )
        # plt.plot(model_data['RIV'])
        # plt.show()
        # plt.plot(model_data['CHD'])
        # plt.show()


    def test_boundary_ibound(self):
        """ Vaidate ibound and boundary grid """
        # If ibound cell is 0, boundary cell has to be 0.
        # If bondary cell is 1, ibound cell has to be 1.
        for i in range(self.ny):
            for j in range(self.nx):
                if self.active_grid.ibound[i][j] == 0:
                    self.assertEqual(self.active_grid.bound_ibound[i][j], 0)
                if self.active_grid.bound_ibound[i][j] == 1:
                    self.assertEqual(self.active_grid.ibound[i][j], 1)

        # plt.imshow(self.active_grid.ibound, interpolation="nearest")
        # plt.show()

    def test_boundary_segments(self):
        """ Validate model boundaries' segments """
        total_cells = 0
        for key in self.model_boundary.line_segments:
            total_cells += len(self.model_boundary.line_segments[key])

        self.assertEqual(
            total_cells,
            np.count_nonzero(self.active_grid.bound_ibound)
            )

        # print(self.model_boundary.line_segments)

    def test_boundary_spd(self):
        """ Validate boundaries's SPD """
        self.assertEqual(
            len(self.model_boundary.boundaries_spd),
            len(self.b_types)
        )
        for key in self.model_boundary.boundaries_spd:
            self.assertTrue(
                key in self.b_types
            )
            self.assertEqual(
                len(self.model_boundary.boundaries_spd[key]),
                self.nper
            )



if __name__ == '__main__':
    unittest.main()
