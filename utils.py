import numpy as np
from matplotlib.path import Path

def xy_to_colrow(x, y, start_x, start_y, dx, dy):
    """ Transform x, y coordinates to column/row """
    # print(x,start_x, dx)
    col = int((x - start_x)/dx)
    row = int((y - start_y)/dy)
    return col, row

def get_polygon(vertices, nx, ny):
    """ Matplotilbs Path, to define inner cells
    nx, ny = 15, 15
    vertices = [(1,1), (10,1), (10,9),(3,2),(1,1)]
    """
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    path = Path(vertices)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))
    return grid.astype(int)

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    points1 = get_line((0, 0), (3, 4))
    points2 = get_line((3, 4), (0, 0))
    print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points