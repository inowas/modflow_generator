import unittest
import numpy as np
import flopy
import matplotlib.pyplot as plt
from model_generation import Solver, Model, ActiveGrid, \
                            ModelTime, DataSource, ModelBoundary, \
                            VectorSource, ModelLayer, PropSource


class TestModelGenerator(unittest.TestCase):
    """ """

    def setUp(self):
        # Model paramenters
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 100
        self.nlay = 1
        self.nper = 100
        self.perlen = [1] * self.nper
        self.nstp = [1] * self.nper
        self.steady = False
        self.nx = 100
        self.ny = 100
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.n_points = 10
        self.n_dim = 2
        self.b_types = {
            'NFL':{},
            'CHD': {'min': 100, 'max': 120,
                    'periods': [3, 5, 6, 10]},
            'RIV': {'min': 80, 'max': 90,
                    'periods': [3, 4, 6]},
            'WEL': {'min': 1000, 'max': 5000,
                    'periods': [6, 10]},
            }
        self.layer_props_params = {
            'hk': {'min': 0.1, 'max': 10.},
            'hani': {'min': 0.5, 'max': 2},
            'vka': {'min': 1, 'max': 10},
            'top': {'min': 100, 'max': 150},
            'botm': {'min': 0, 'max': 50},
        }
        self.headtol = 0.01
        self.maxiterout = 100
        self.mxiter = 100
        self.iter1 = 100
        self.hclose = 0.1
        self.rclose = 0.1
        self.model_name = 'model_1'
        self.workspace = 'models\\' + self.model_name
        self.version = 'mfnwt'
        self.exe_name = 'MODFLOW-NWT.exe'
        self.verbose = False

        # Vector input data source
        self.vector_source = VectorSource(
            self.xmin, self.ymin, self.xmax, self.ymax, self.n_points, self.n_dim
            )
        # Time-series input data source
        self.data_source = DataSource(
            self.nper,
            self.b_types
        )
        self.data_source.generate_data()
        # Layer properties data source
        self.properties_source = PropSource(
            self.layer_props_params, self.vector_source.random_points
        )
        self.properties_source.set_params_data()
        # Model grid
        self.active_grid = ActiveGrid(
            self.xmin, self.ymin, self.xmax, self.ymax, self.nx, self.ny, self.nlay
            )
        self.active_grid.set_ibound(self.vector_source.polygon)
        # Model time
        self.model_time = ModelTime(
            self.nper, self.perlen, self.nstp, self.steady
            )
        # Model boundary
        self.model_boundary = ModelBoundary(
            self.active_grid, self.model_time, self.data_source
            )
        self.model_boundary.set_line_segments(self.vector_source.polygon)
        self.model_boundary.set_well(self.vector_source.inner_points)
        self.model_boundary.set_boundaries_spd()

        # Model layer
        self.model_layer = ModelLayer(
            self.properties_source, self.active_grid
        )
        self.model_layer.set_properties()
        self.model_layer.reshape_properties()
        # Model solver
        self.model_solver = Solver(
            self.headtol,
            self.maxiterout,
            self.mxiter,
            self.iter1,
            self.hclose,
            self.rclose
            )
        # Flopy model
        self.model = Model(
            model_name=self.model_name,
            workspace=self.workspace,
            version=self.version,
            exe_name=self.exe_name,
            verbose=self.verbose,
            model_solver=self.model_solver,
            model_time=self.model_time,
            model_grid=self.active_grid,
            model_boundary=self.model_boundary,
            model_layer=self.model_layer,
        )
        # Flopy packages
        self.mf = self.model.get_mf()
        self.pcg = self.model.get_pcg(self.mf)
#        self.nwt = self.model.get_nwt(self.mf)
        self.dis = self.model.get_dis(self.mf)
        self.bas = self.model.get_bas(self.mf)
        self.lpf = self.model.get_lpf(self.mf)
        self.chd = self.model.get_chd(self.mf)
        self.riv = self.model.get_riv(self.mf)
        self.wel = self.model.get_wel(self.mf)

    def tearDown(self):

        self.vector_source = None
        self.data_source = None
        self.active_grid = None
        self.model_boundary = None
        self.model = None
        self.mf = None
        self.pcg = None
#        self.nwt = None
        self.dis = None
        self.bas = None
        self.lpf = None
        self.chd = None
        self.riv = None
        self.wel = None

    def test_print_results(self):
        # PLOT TIME SERIES
        plt.plot(self.data_source.b_data['CHD'])
        plt.xlabel('Stress periods')
        plt.ylabel('Head, m')
        plt.title('Generated time-variant head boundary series')
        plt.show()
        # PLOT POINTS AND CONVEX HULL
        points = self.vector_source.convex_hull.points
        hull = self.vector_source.convex_hull
        n_segments = self.model_boundary.nums_of_segments
        plt.plot(points[:,0], points[:,1], 'o')
        i = 0
        btype = 0
        for simplex in hull.simplices:
            if i >= n_segments[btype]:
                btype += 1
                i = 0
            if btype == 0:
                nfl = plt.plot(points[simplex, 0], points[simplex, 1], 'k-', label = 'No flow')
            elif btype == 1:
                chd = plt.plot(points[simplex, 0], points[simplex, 1], 'r-', label = 'Constant head')
            elif btype == 2:
                riv = plt.plot(points[simplex, 0], points[simplex, 1], 'g-', label = 'River')
            i += 1
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.title('Generated random points, convex hull and boundary types')
        plt.show()
        # PLOT PROPERTIES RASTERS
        mask = np.flipud(self.active_grid.ibound == 0)
        layer = self.model_layer.properties['hk'][0]
        layer[mask] = np.nan
        plt.imshow(
            layer, interpolation='nearest'
            )
        plt.xlabel('columns')
        plt.ylabel('rows')
        plt.title('Generated hydraulic cunductivity')
        bar = plt.colorbar()
        bar.set_label('Kx, m')
        plt.show()

        layer = self.model_layer.properties['top'][0]
        layer[mask] = np.nan
        plt.imshow(
            layer, interpolation='nearest'
            )
        plt.xlabel('columns')
        plt.ylabel('rows')
        plt.title('Generated model top elevation')
        bar = plt.colorbar()
        bar.set_label('Elevation, m')
        plt.show()

        layer = self.model_layer.properties['botm'][0]
        layer[mask] = np.nan
        plt.imshow(
            layer, interpolation='nearest'
            )
        plt.xlabel('columns')
        plt.ylabel('rows')
        plt.title('Generated bottom elevation of the first layer')
        bar = plt.colorbar()
        bar.set_label('Elevation, m')
        plt.show()

        # PLOT GRID
        mask = self.active_grid.ibound == 1
        for idx, i in enumerate(self.active_grid.ibound):
            for jdx, j in enumerate(i):
                x1 = self.xmin + jdx * self.dx
                x2 = x1 + self.dx
                y1 = self.ymin + idx * self.dy
                y2 = y1 + self.dy
                if mask[idx][jdx]:
                    plt.plot([x1, x2, x2, x1, x1],[y1, y1, y2, y2, y1], color = 'k')
        plt.title('Example model grid and resulting locations of wells')
        plt.show()
        # PLOT SURFACE
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import Rbf
        x = [1, 100, 10, 90, 40]
        y = [2, 99, 99, 5, 40]
        z = [.05, .07, .1, .1, .5]
        rbfi = Rbf(x, y, z)
        grid_x, grid_y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        grid_z = rbfi(grid_x, grid_y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x, grid_y, grid_z)
        plt.title('Hypothetical probability distribution of optimal well location over model domain')
        ax.set_zlabel('Probability')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        plt.show()
    # def test_random_points(self):
    #     """
    #     Testing random points and convex hull generation
    #     """
    #     self.assertEqual(
    #         self.n_points,
    #         len(self.vector_source.convex_hull.points)
    #     )

    # def test_random_data(self):
    #     """ Test data source generation"""
    #     self.assertEqual(len(self.data_source.b_data), len(self.b_types))
    #     for key in self.b_types:
    #         if key != 'NFL':
    #             self.assertEqual(
    #                 len(self.data_source.b_data[key]), self.nper
    #                 )
    #     # plt.plot(model_data['RIV'])
    #     # plt.show()
    #     # plt.plot(model_data['CHD'])
    #     # plt.show()


    # def test_boundary_ibound(self):
    #     """ Vaidate ibound and boundary grid """
    #     # If ibound cell is 0, boundary cell has to be 0.
    #     # If bondary cell is 1, ibound cell has to be 1.
    #     for i in range(self.ny):
    #         for j in range(self.nx):
    #             if self.active_grid.ibound[i][j] == 0:
    #                 self.assertEqual(self.active_grid.bound_ibound[i][j], 0)
    #             if self.active_grid.bound_ibound[i][j] == 1:
    #                 self.assertEqual(self.active_grid.ibound[i][j], 1)

    #     plt.imshow(self.active_grid.bound_ibound, interpolation="nearest")
    #     plt.show()

    # def test_boundary_segments(self):
    #     """ Validate model boundaries' segments """
    #     total_cells = 0
    #     for key in self.model_boundary.line_segments:
    #         total_cells += len(self.model_boundary.line_segments[key])

    #     self.assertEqual(
    #         total_cells,
    #         np.count_nonzero(self.active_grid.bound_ibound)
    #         )

    # def test_boundary_spd(self):
    #     """ Validate boundaries's SPD """
    #     self.assertEqual(
    #         len(self.model_boundary.boundaries_spd),
    #         len(self.b_types)
    #     )
    #     for key in self.model_boundary.boundaries_spd:
    #         self.assertTrue(
    #             key in self.b_types
    #         )
    #         self.assertEqual(
    #             len(self.model_boundary.boundaries_spd[key]),
    #             self.nper
    #         )


    # def test_property_source(self):
    #     """ Validate creation of random properties """
    #     self.assertEqual(
    #         len(self.properties_source.params_data),
    #         len(self.layer_props_params)
    #     )
    #     for key in self.properties_source.params_data:
    #         self.assertTrue(
    #             len(self.properties_source.params_data[key]['z']) == \
    #             len(self.properties_source.params_data[key]['x']) == \
    #             len(self.properties_source.params_data[key]['y']) == \
    #             len(self.vector_source.random_points)
    #         )
    #         for z in self.properties_source.params_data[key]['z']:
    #             self.assertTrue(
    #                 z <= self.layer_props_params[key]['max']
    #             )
    #             self.assertTrue(
    #                 z >= self.layer_props_params[key]['min']
    #             )

    # def test_prop_rasters(self):
    #     """ Validate generated property rasters """
        
    #     plt.imshow(
    #         self.model_layer.properties['botm'], interpolation='nearest'
    #         )
    #     plt.colorbar()
    #     plt.show()

    # def test_model(self):
    #     self.mf.write_input()
    #     self.mf.plot()
    #     success, buff = self.mf.run_model()



if __name__ == '__main__':
   unittest.main()
