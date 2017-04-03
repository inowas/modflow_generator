# Procedural generation of Modflow groundawter
# models

import numpy as np 
from scipy.spatial import ConvexHull
from random import randint
from utils import xy_to_colrow, get_polygon, get_line


class Grid(object):
    """ Model grid  """
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.nx = nx
        self.ny = ny
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny


class ActiveGrid(Grid):
    """ Extends Model grid class. """
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny):
        super().__init__(xmin, ymin, xmax, ymax, nx, ny)
        self.ibound = np.zeros((self.ny, self.nx))
        self.bound_ibound = np.zeros((self.ny, self.nx))

    def set_ibound(self, polygon):
        """ Sets IBOUND array. """
        # Translate polygon coolrdinates to col/row
        for i, point in enumerate(polygon):
            point = xy_to_colrow(
                point[0],
                point[1],
                self.xmin,
                self.ymin,
                self.dx,
                self.dy
            )
            polygon[i] = point
        # 2D Array
        polygon_grid = get_polygon(polygon, self.nx, self.ny)
        # Cols, rows, intersected by line
        line_grid = []
        for i in range(len(polygon) - 1):
            for j in get_line(polygon[i], polygon[i+1]):
                line_grid.append(j)
        # Add line grid cells to polygon grid
        for point in line_grid:
            polygon_grid[point[1], point[0]] = 1
            self.bound_ibound[point[1], point[0]] = 1

        self.ibound = polygon_grid
        return self.ibound


class ModelTime(object):
    """ Time characteristics of the model """
    def __init__(self, nper=1, perlen=100, nstp=1, steady=True):
        self.nper = nper
        self.perlen = perlen
        self.nstp = nstp
        self.steady = steady

class DataSource(object):
    """ Time series data for the model """
    def __init__(self, nper, b_types):
        self.nper = nper
        self.b_types = b_types
        self.b_data = {}

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

class ModelBoundary(object):
    """ """
    def __init__(self, grid, model_time, data_source):
        self.grid = grid
        self.model_time = model_time
        self.data_source = data_source

        self.line = []
        self.nums_of_segments = None
        self.boundaries = {}

    def set_line_segments(self, line):
        """ Returns line segments in col/rows """
        for i, point in enumerate(line):
            point = xy_to_colrow(
                point[0],
                point[1],
                self.grid.xmin,
                self.grid.ymin,
                self.grid.dx,
                self.grid.dy
            )
            line[i] = point

        for i in range(len(line) - 1):
            self.line.append(get_line(line[i], line[i+1]))

        self.nums_of_segments = self.rand_seg_nums(
            len(self.line),
            len(self.data_source.b_data)
            )
        return self.line, self.nums_of_segments
    
    # def set_chd(self):
    #     values = self.data_source.b_data[]
    #     spd = self.construct_spd(
    #         self.model_time.nstp,
    #         self.values,
    #         self.line
        # )

    @staticmethod
    def construct_spd(nstp, values, cells):
        """ Returns SPD dictionary """
        spd = {}
        for step in range(nstp):
            step_data = []
            for i, cell in enumerate(cells):
                step_data.append(
                    cell[0],
                    cell[1],
                    cell[2],
                    values[step][i][0],
                    values[step][i][1]
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

        self.rand_points = self.give_rand_points(
            n_points=n_points,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            n_dim=n_dim
            )
        self.convex_hull = self.give_convex_hull(self.rand_points)
        self.polygon = self.give_polygon(self.convex_hull.vertices, self.rand_points)
        # self.boundary = self.give_boundary()

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

