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
    def __init__(self, xmin, ymin, xmax, ymax, nx, ny):
        super().__init__(xmin, ymin, xmax, ymax, nx, ny)
        self.ibound = None

    def give_grid_points(self, polygon):
        self.ibound = np.zeros((self.nx, self.ny))
        poly = Polygon(polygon)
        x, y = poly.exterior.xy
        plt.plot(x, y)
        plt.show()

        for i in range(self.ny):
            for j in range(self.nx):
                cell_x = self.xmin + self.dx * .5 + j * self.dx
                cell_y = self.ymax - self.dy * .5 - i * self.dy
            
                if Point(cell_x, cell_y).within(Polygon(polygon)):
                    self.ibound[i][j] = 1


class RandomPoints(object):
    def __init__(self, n_points, n_dim=2):
        self.n_points = n_points
        self.rand_points = self.give_rand_points(n_points, n_dim=2)
        self.convex_hull = self.give_convex_hull(self.rand_points)
        self.polygon = self.give_polygon(self.convex_hull.vertices, self.rand_points)


    @staticmethod
    def give_rand_points(n_points, n_dim=2):
        """ Returns random 2d points. """
        return np.random.rand(n_points, n_dim)

    @staticmethod
    def give_convex_hull(rand_points):
        """ Returns scipy's convex hull for given points. """
        return ConvexHull(rand_points)
    
    @staticmethod
    def give_polygon(vertices, points):
        polygon = []
        for vertex in vertices:
            polygon.append(points[vertex])
        return np.array(polygon)
