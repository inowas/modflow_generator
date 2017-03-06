# Procedural generation of Modflow groundawter
# models

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, LineString, box
import matplotlib.pyplot as plt

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
        self.ibound = np.zeros((self.nx, self.ny))
        self.bound_ibound = np.zeros((self.nx, self.ny))


    def set_ibound(self, polygon):
        """ Sets IBOUND array. """
        if not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.append(polygon, [polygon[0]], axis=0)

        exterior = Polygon(polygon)
        boundary = LineString(polygon)
        # x, y = exterior.exterior.xy
        # plt.plot(x, y)
        # plt.show()

        for i in range(self.ny):
            i_reversed = self.ny - i - 1
            for j in range(self.nx):
                # Center points of grid cells
                cell_x = self.xmin + self.dx * .5 + j * self.dx
                cell_y = self.ymin + self.dy * .5 + i_reversed * self.dy
                # Grid cell bbox
                cell = box(
                    self.xmin + self.dx * j,
                    self.ymin + self.dy * i_reversed,
                    self.xmin + self.dx * (j + 1),
                    self.ymin + self.dy * (i_reversed + 1)
                    )
                # Check if central point within a polygon
                if Point(cell_x, cell_y).within(Polygon(exterior)):
                    self.ibound[i][j] = 1
                # Check if boundary intersects the cell
                if boundary.intersects(cell):
                    self.bound_ibound[i][j] = 1
                    self.ibound[i][j] = 1




class RandomPoints(object):
    """ Random points for generation of model properties. """
    def __init__(self, xmin, ymin, xmax, ymax, n_points, n_dim=2):

        self.rand_points = self.give_rand_points(
            n_points=n_points,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            n_dim=2
            )
        self.convex_hull = self.give_convex_hull(self.rand_points)
        self.polygon = self.give_polygon(self.convex_hull.vertices, self.rand_points)


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
        return polygon

    @staticmethod
    def give_boundary(vertices, points):
        polygon = np.zeros((len(vertices), 2))
        for i, vertex in enumerate(vertices):
            polygon[i] = points[vertex]
        return polygon
