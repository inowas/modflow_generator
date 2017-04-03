import unittest
import matplotlib.pyplot as plt
from model_generator import ActiveGrid, ModelTime, DataSource, ModelBoundary, VectorSource


class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        xmin = 0
        xmax = 100
        ymin = 0
        ymax = 50
        self.nper = 200
        perlen = [100]
        nstp = [100]
        steady = True
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
            xmin, ymin, xmax, ymax, self.n_points, self.n_dim
            )
        self.data_source = DataSource(
            self.nper,
            self.b_types
        )
        self.active_grid = ActiveGrid(xmin, ymin, xmax, ymax, self.nx, self.ny)
        self.model_time = ModelTime(self.nper, perlen, nstp, steady)
        self.model_boundary = ModelBoundary(self.active_grid, self.model_time, self.data_source)

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
        model_data = self.data_source.generate_data()
        self.assertEqual(len(model_data), len(self.b_types))
        for key in self.b_types:
            if key != 'NFL':
                self.assertEqual(len(model_data[key]), self.nper)
        plt.plot(model_data['RIV'])
        plt.show()
        plt.plot(model_data['CHD'])
        plt.show()


    def test_boundary_ibound(self):
        """ Vaidation of ibound and boundary """
        self.active_grid.set_ibound(self.vector_source.polygon)

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

    # def test_boundaries(self):
    #     """ Validation of model boundaries """
    #     line_segments, nums_of_segments = self.model_boundary.set_line_segments(
    #         self.vector_source.polygon
    #         )

    #     self.assertEqual(len(line_segments), len(self.vector_source.polygon)-1)
    #     self.assertEqual(sum(nums_of_segments), len(line_segments))
    #     self.assertEqual(len(nums_of_segments), len(self.b_types))



if __name__ == '__main__':
    unittest.main()
