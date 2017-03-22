# Procedural generation of Modflow groundawter
# models

import numpy as np
from scipy.spatial import ConvexHull
from random import randint
from .utils import *


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

class TimeSeriesSource(object):
    """ Time series for the model """
    def __init__(self, model_time, head_mean):
        self.model_time = model_time


class ModelBoundary(object):
    """ """
    # chd_spd = {}
    # any_spd = {}

    def __init__(self, grid):
        self.grid = grid
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

        return self.line
    
    def set_chd_spd(self, num_of_bounds):
        self.nums_of_segments = self.rand_seg_nums(len(self.line), num_of_bounds)


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

