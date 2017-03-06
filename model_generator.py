# Procedural generation of Modflow groundawter
# models

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
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

    def set_ibound(self, polygon):
        """ Sets IBOUND array. """
        exterior = Polygon(polygon)
        x, y = exterior.exterior.xy
        plt.plot(x, y)
        plt.show()

        for i in range(self.ny):
            for j in range(self.nx):
                # Center points of grid cells
                cell_x = self.xmin + self.dx * .5 + j * self.dx
                cell_y = self.ymax - self.dy * .5 - i * self.dy
                # Check if central point within a polygon
                if Point(cell_x, cell_y).within(Polygon(exterior)):
                    self.ibound[i][j] = 1


class RandomPoints(Grid):
    """ Random points for generation of model properties. """
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny, n_points, n_dim=2):
        super().__init__(xmin, ymin, xmax, ymax, nx, ny)

        self.rand_points = self.give_rand_points(
            n_points=n_points,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            n_dim=2
            )
        self.convex_hull = self.give_convex_hull(self.rand_points)
        self.polygon = self.give_polygon(self.convex_hull.vertices, self.rand_points)


    @staticmethod
    def give_rand_points(n_points, xmin, xmax, ymin, ymax, n_dim=2):
        """ Returns random 2d points. """
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
        polygon = []
        for vertex in vertices:
            polygon.append(points[vertex])
        return np.array(polygon)
