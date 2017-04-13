# Procedural generation of Modflow groundawter
# models

import numpy as np 
from scipy.spatial import ConvexHull
from scipy.interpolate import Rbf
from random import randint
from copy import deepcopy
import flopy
from utils_model_generation import xy_to_colrow, get_polygon_grid, get_line_grid


class Model(object):
    """ Flopy Model class """
    def __init__(self, model_name, workspace, version, exe_name, verbose, model_solver,
                 model_time, model_grid, model_boundary, model_layer):
        self.model_name = model_name
        self.workspace = workspace
        self.version = version
        self.exe_name = exe_name
        self.verbose = verbose
        self.model_solver = model_solver
        self.model_time = model_time
        self.model_grid = model_grid
        self.model_boundary = model_boundary
        self.model_layer = model_layer

    def get_mf(self):
        mf = flopy.modflow.Modflow(
            modelname=self.model_name,
            version=self.version,
            model_ws=self.workspace,
            exe_name=self.exe_name,
            verbose=self.verbose
        )
        return mf

    def get_nwt(self, mf):
        nwt = flopy.modflow.ModflowNwt(
            mf,
            headtol=self.model_solver.headtol,
            maxiterout=self.model_solver.maxiterout
        )
        return nwt

    def get_pcg(self, mf):
        pcg = flopy.modflow.ModflowPcg(
            mf,
            mxiter=self.model_solver.mxiter,
            iter1=self.model_solver.iter1,
            hclose=self.model_solver.hclose,
            rclose=self.model_solver.rclose,
        )
        return pcg

    def get_dis(self, mf):
        dis = flopy.modflow.ModflowDis(
            mf,
            nlay=self.model_grid.nlay,
            ncol=self.model_grid.nx,
            nrow=self.model_grid.ny,
            delr=self.model_grid.dy,
            delc=self.model_grid.dx,
            nper=self.model_time.nper,
            perlen=self.model_time.perlen,
            nstp=self.model_time.nstp,
            steady=self.model_time.steady,
            top=self.model_layer.properties['top'][0],
            botm=self.model_layer.properties['botm']
        )
        return dis

    def get_bas(self, mf):
        bas = flopy.modflow.ModflowBas(
            mf,
            ibound=self.model_grid.ibound_reshaped,
            strt=self.model_layer.properties['strt']
        )
        return bas

    def get_lpf(self, mf):
        lpf = flopy.modflow.ModflowLpf(
            mf,
            laytyp=self.model_layer.properties['laytyp'],
            chani=self.model_layer.properties['chani'],
            hani=self.model_layer.properties['hani'],
            vka=self.model_layer.properties['vka'],
            hk=self.model_layer.properties['hk'],
            ss=self.model_layer.properties['ss'],
            sy=self.model_layer.properties['sy'],
        )
        return lpf

    def get_chd(self, mf):
        if len(self.model_boundary.boundaries_spd['CHD'][0]) == 0:
            return None

        chf = flopy.modflow.ModflowChd(
            mf,
            stress_period_data=self.model_boundary.boundaries_spd['CHD']
        )
        # print(self.model_boundary.boundaries_spd['CHD'])
        return chf

    def get_riv(self, mf):
        if len(self.model_boundary.boundaries_spd['RIV'][0]) == 0:
            return None
        # print(self.model_boundary.boundaries_spd['RIV'])
        riv = flopy.modflow.ModflowRiv(
            mf,
            stress_period_data=self.model_boundary.boundaries_spd['RIV']
        )
        # print(self.model_boundary.boundaries_spd['RIV'][0])
        return riv

    def get_wel(self, mf):
        if len(self.model_boundary.boundaries_spd['WEL'][0]) == 0:
            return None
        # print(self.model_boundary.boundaries_spd['WEL'])
        wel = flopy.modflow.ModflowWel(
            mf,
            stress_period_data=self.model_boundary.boundaries_spd['WEL']
        )
        return wel


class Solver(object):
    """ Modflow NWT input """
    def __init__(self, headtol, maxiterout, mxiter, iter1, hclose, rclose):
        self.headtol = headtol
        self.maxiterout = maxiterout
        self.mxiter = mxiter
        self.iter1 = iter1
        self.hclose = hclose
        self.rclose = rclose

class Grid(object):
    """ Model grid  """
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny, nlay):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.nx = nx
        self.ny = ny
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.nlay = nlay


class ActiveGrid(Grid):
    """ Extends Model grid class. """
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny, nlay):
        super().__init__(xmin, ymin, xmax, ymax, nx, ny, nlay)
        self.ibound = np.zeros((self.ny, self.nx))
        self.bound_ibound = np.zeros((self.ny, self.nx))
        self.ibound_mask = np.zeros((self.ny, self.nx), dtype=bool)

    def set_ibound(self, polygon):
        """ Sets IBOUND array. """
        # Translate polygon coolrdinates to col/row
        polygon_converted = []
        for i, point in enumerate(polygon):
            point_converted = xy_to_colrow(
                point[0],
                point[1],
                self.xmin,
                self.ymin,
                self.dx,
                self.dy
            )
            polygon_converted.append(point_converted)
        # 2D Array
        polygon_grid = get_polygon_grid(polygon_converted, self.nx, self.ny)
        # Cols, rows, intersected by line
        line_grid = []
        for i in range(len(polygon_converted) - 1):
            for j in get_line_grid(polygon_converted[i], polygon_converted[i+1]):
                line_grid.append(j)
        # print(len(line_grid))
        # Add line grid cells to polygon grid
        for point in line_grid:
            polygon_grid[point[1], point[0]] = 1
            self.bound_ibound[point[1], point[0]] = 1

        self.ibound = polygon_grid
        self.ibound_mask = self.ibound > 0
        return self.ibound
    
    @property
    def ibound_reshaped(self):
        return self.ibound.reshape(self.nlay, self.ny, self.nx)


class ModelTime(object):
    """ Time characteristics of the model """
    def __init__(self, nper=1, perlen=100, nstp=1, steady=True):
        self.nper = nper
        self.perlen = perlen
        self.nstp = nstp
        self.steady = steady


class ModelLayer(object):
    """ Layer properties of the model """
    def __init__(self, prop_source, grid):
        self.grid = grid
        self.prop_source = prop_source
        self.properties = {
            'laytyp': 0,
            'hani': 1.0,
            'vka': 1.0,
            'hk': 1.0,
            'ss': 1e-5,
            'sy': 0.15,
            'top': 0,
            'botm': -10,
            'strt': 100,
            'chani': 0
        }

    def set_properties(self):
        """ Assign propeties to layer """
        for key in self.prop_source.params_data:
            property_grid = self.interpolate(
                self.grid.xmin,
                self.grid.ymin,
                self.grid.dx,
                self.grid.dy,
                self.grid.nx,
                self.grid.ny,
                self.prop_source.params_data[key]['x'],
                self.prop_source.params_data[key]['y'],
                self.prop_source.params_data[key]['z']
            )

            self.properties[key] = self.normalize(
                property_grid,
                self.prop_source.params[key]['min'],
                self.prop_source.params[key]['max'],
                self.grid.ibound_mask
            )

    @staticmethod
    def interpolate(xmin, ymin, dx, dy, nx, ny, x, y, z):
        """ Interpolate point data to grid coverage """
        # convert coordinates to cols, rows
        x = ((np.array(x) - xmin) / dx).astype(int)
        y = ((np.array(y) - ymin) / dy).astype(int)
        # scipy's RBF interpolation
        rbfi = Rbf(x, y, z)

        grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
        grid_z = rbfi(grid_x, grid_y)

        return grid_z

    @staticmethod
    def normalize(grid, new_min, new_max, mask):
        """ Normalizes grid to given min max
            (b-a)*(xi-xmin)/(xmax-xmin) + a
        """

        norm_grid = (new_max - new_min) * \
                    (grid - grid[mask].min()) / (grid[mask].max() - \
                    grid[mask].min()) + new_min

        return norm_grid

    def reshape_properties(self):
        """ Adds nlay dimension to arrays """
        for key in self.properties:
            try:
                self.properties[key] = self.properties[key].reshape(
                    self.grid.nlay,
                    self.grid.nx,
                    self.grid.ny
                )
            except AttributeError:
                pass





class ModelBoundary(object):
    """ Model boundaries class """
    def __init__(self, grid, model_time, data_source):
        self.grid = grid
        self.model_time = model_time
        self.data_source = data_source
        self.well_cells = []
        self.line_segments = {
            'NFL': [],
            'CHD': [],
            'RIV': []
        }
        self.boundaries_spd = {
            'NFL': None,
            'CHD': None,
            'RIV': None,
            'WEL': None
        }

    def set_well(self, inner_points):
        """ Sets x, y of a randomly selected point """
        if len(inner_points) == 0:
            return
        well = inner_points[
            randint(0, len(inner_points) - 1)
        ]
        well = xy_to_colrow(
            well[0],
            well[1],
            self.grid.xmin,
            self.grid.ymin,
            self.grid.dx,
            self.grid.dy
        )
        self.well_cells.append(well)

    def set_line_segments(self, line):
        """ Returns line segments in col/rows """
        line_row_col = []
        # transform x, y coordinates to columns and rows
        line_converted = []
        for i, point in enumerate(line):
            point_converted = xy_to_colrow(
                point[0],
                point[1],
                self.grid.xmin,
                self.grid.ymin,
                self.grid.dx,
                self.grid.dy
            )
            line_converted.append(point_converted)
        # get cells intersected by line segments
        for i in range(len(line_converted) - 1):
            line_row_col.append(get_line_grid(line_converted[i], line_converted[i+1]))
        # delete last cell of each segment
        for i in line_row_col:
            del i[-1]
        # generate number of segments for each boundary
        nums_of_segments = self.rand_seg_nums(
            len(line_row_col),
            len(self.data_source.linear_boundaries)
            )
        # assign cells to each boundary type
        for idx, key in enumerate(self.data_source.linear_boundaries):
            for i in range(nums_of_segments[idx]):
                self.line_segments[key].extend(
                    line_row_col[i]
                    )
            del line_row_col[:nums_of_segments[idx]]

        return self.line_segments

    def set_boundaries_spd(self):
        """ Set boundaries's SPD """
        for key in self.data_source.b_data:
            if key == 'NFL':
                self.boundaries_spd['NFL'] = self.construct_spd(
                    b_type='NFL',
                    nper=self.model_time.nper,
                    cells=self.line_segments['NFL']
                )
            elif key == 'CHD':
                self.boundaries_spd['CHD'] = self.construct_spd(
                    b_type='CHD',
                    nper=self.model_time.nper,
                    values_1=self.data_source.b_data['CHD'],
                    values_2=self.data_source.b_data['CHD'],
                    cells=self.line_segments['CHD']
                )
            elif key == 'RIV':
                self.boundaries_spd['RIV'] = self.construct_spd(
                    b_type='RIV',
                    nper=self.model_time.nper,
                    values_1=self.data_source.b_data['RIV'],
                    values_2=np.ones(len(self.data_source.b_data['RIV'])),
                    values_3=np.ones(len(self.data_source.b_data['RIV']))*80,
                    cells=self.line_segments['RIV']
                )
            elif key == 'WEL' and len(self.well_cells) > 0:
                self.boundaries_spd['WEL'] = self.construct_spd(
                    b_type='WEL',
                    nper=self.model_time.nper,
                    values_1=self.data_source.b_data['WEL'],
                    cells=self.well_cells
                )

    @staticmethod
    def construct_spd(b_type, nper, cells, values_1=None, values_2=None, values_3=None):
        """ Returns SPD dictionary """
        spd = {}
        for step in range(nper):
            step_data = []
            for cell in cells:
                if b_type == 'RIV':
                    step_data.append(
                        [
                            0,
                            cell[1],
                            cell[0],
                            values_1[step],
                            values_2[step],
                            values_3[step]
                        ]
                    )
                elif b_type == 'CHD':
                    step_data.append(
                        [
                            0,
                            cell[1],
                            cell[0],
                            values_1[step],
                            values_2[step]
                        ]
                    )
                elif b_type == 'NFL':
                    step_data.append(
                        [
                            0,
                            cell[0],
                            cell[1]
                        ]
                    )
                elif b_type == 'WEL':
                    step_data.append(
                        [
                            0,
                            cell[0],
                            cell[1],
                            values_1[step]
                        ]
                    )
            spd[step] = step_data
        return spd


    @staticmethod
    def rand_seg_nums(seg_number, bound_number):
        """ Returns list of random number of segments """
        if bound_number > seg_number:
            bound_number = seg_number
        result = []
        segments_left = seg_number
        for i in range(bound_number):
            if i == bound_number - 1:
                result.append(segments_left)
            else:
                seg_number_i = randint(0, segments_left)
                segments_left -= seg_number_i
                result.append(seg_number_i)

        return result



class VectorSource(object):
    """ Random points for generation of model properties. """
    def __init__(self, xmin, ymin, xmax, ymax, n_points, n_dim=2):

        self.random_points = self.give_rand_points(
            n_points=n_points,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            n_dim=n_dim
            )
        self.convex_hull = self.give_convex_hull(self.random_points)
        self.polygon = self.give_polygon(self.convex_hull.vertices, self.random_points)
        self.inner_points = self.give_inner_points(self.convex_hull.vertices,
                                                   self.random_points)

    @staticmethod
    def give_rand_points(n_points, xmin, xmax, ymin, ymax, n_dim=2):
        """ Returns 2d coordinates of random points. """
        random_points = np.random.rand(n_points, n_dim)
        random_points[:, 0] = random_points[:, 0]*(xmax-xmin)+xmin
        random_points[:, 1] = random_points[:, 1]*(ymax-ymin)+ymin

        return random_points

    @staticmethod
    def give_convex_hull(rand_points):
        """ Returns scipy's convex hull object for given points. """
        return ConvexHull(rand_points)

    @staticmethod
    def give_polygon(vertices, points):
        """ Returns polygon object out of the convex hull. """
        polygon = np.zeros((len(vertices), 2))
        for i, vertex in enumerate(vertices):
            polygon[i] = points[vertex]
        # End point of a polygon equals to start point
        polygon = polygon.tolist()
        if polygon[-1] != polygon[0]:
            polygon.append(polygon[0])
        return polygon

    @staticmethod
    def give_inner_points(vertices, points):
        inner_points = points.tolist()
        for vertex in np.sort(vertices)[::-1]:
            del inner_points[vertex]
        return inner_points


class DataSource(object):
    """ Time series data for the model """
    def __init__(self, nper, b_types):
        self.nper = nper
        self.b_types = b_types
        self.b_data = {}
        self.linear_boundaries = [
            key for key in b_types if key != 'WEL'
            ]

    def generate_data(self):
        """ Generates data series for given b_types """

        for key in self.b_types:
            if key != 'NFL':
                if self.nper == 1:
                    self.b_data[key] = np.random.uniform(
                        self.b_types[key]['min'],
                        self.b_types[key]['max'],
                        (1,)
                    )
                else:
                    self.b_data[key] = self. set_fourier_data(
                        self.nper,
                        self.b_types[key]['periods'],
                        self.b_types[key]['min'],
                        self.b_types[key]['max']
                    )
            else:
                self.b_data[key] = {}
        return self.b_data

    @staticmethod
    def set_fourier_data(nper, periods, min_value, max_value):
        """ Returns an inverse of a random descrete Fourier serie """

        fourier = np.zeros((nper,), dtype=complex)
        for period in periods:
            if period in range(nper+1):
                fourier[period] = np.exp(
                    1j * np.random.uniform(
                        0,
                        2*np.pi
                    )
                )
            else:
                print('WARNING: period', period, 'is out of NPER range')

        serie = np.fft.ifft(fourier).real

        low = -2 * np.pi / nper
        high = 2 * np.pi / nper
        # denormalize to min..max
        serie_denorm = ((serie - low) / (high - low)) * (max_value - min_value) + min_value

        return serie_denorm


class PropSource(object):
    """ Source of property data for model layers """
    def __init__(self, params, random_points):
        self.params = params
        self.rand_points = random_points
        self.params_data = {}

    def set_params_data(self):
        """ Set random parameters data X, Y, Z """
        for key in self.params:
            self.params_data[key] = {}
            self.params_data[key]['x'] = [i[0] for i in self.rand_points]
            self.params_data[key]['y'] = [i[1] for i in self.rand_points]
            self.params_data[key]['z'] = self.generate_random_data(
                min_=self.params[key]['min'],
                max_=self.params[key]['max'],
                len_=len(self.rand_points)
            )
        return self.params_data

    @staticmethod
    def generate_random_data(min_, max_, len_):
        """ Returns array of random values """
        return np.random.uniform(min_, max_, len_)
