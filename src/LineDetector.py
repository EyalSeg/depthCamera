

import numpy as np
import math
from scipy import optimize
import sys

loss_magic_num = 2

# find the position in the line to join.
# x y z are arrays of datapoints in the line, where x is width, y is height and z is depth
# distance is how far from the last datapoint to stand

# returns {next_pos, angle, line_polynomial}
# next_position is the desired position.
# angle is the angle between the desired position and the last position.
# the line_polynomial is the interpolation polynomial representing the line
def find_line_position(x, y, z, distance = 1, max_degree = 3):
    # assume the datapoints are on a straight plane, so we can ignore the y array
    last_position = np.array([x[-1], z[-1]])
    weights = generate_weights(x)

    line_polynomial = curve_fit(x, z, max_degree, weights= weights)

    next_position = find_next_position(line_polynomial, last_position, distance)
    angle = angle_between(next_position, last_position)

    return {'next_pos': next_position, "angle": angle, "line_polynomial": line_polynomial}



## THE CODE FROM HERE ON ASSUMES A 2D PLANE ##
# meaning that the y axis here on is usually the z axis in 3d space (since we assume the datapoints are on a flat plane,
#  height is meaningless)

def curve_fit(x, y, max_degree, weights = None):
    # test different degrees

    best_loss = sys.maxsize
    best_curve = None

    for degree in range(1, max_degree + 1):
        curve, [resid, rank, sv, rcond] = \
            np.polynomial.polynomial.Polynomial.fit(x, y, degree, w=weights, full=True)

        loss = resid * (loss_magic_num ** degree) # punishes higher-order polynomials

        if loss < best_loss:
            best_loss = loss
            best_curve = curve

    return best_curve


def generate_weights(data):
    # dimish the weight of a point the further away it is from the line's ending
    weights = [1 / i for i in reversed(range(1, len(data) + 1))]

    # "encourage" the line to pass through the last point
    weights[-1] = len(data)

    return weights

# find a position [x, y] on the given polynomial so that [x, y] is distance away from the given position
def find_next_position(polynomial, position, distance):

    #result_x = position[0] + distance
    #return [result_x, polynomial(result_x)]

    # calculates the distance between a and the given position
    norm_func = lambda a : np.linalg.norm(np.array([a, polynomial(a)]) - position)

    # a root of this function is distance away from the given position
    desired_distance_func = lambda a : norm_func(a) - distance

    result_x = optimize.bisect(desired_distance_func, position[0] - 0.1, position[0] + distance * 2)
    #result_x = optimize.newton(desired_distance_func, position[0])
    return [result_x, polynomial(result_x)]

#returns the clockwise angle from p1 to p2 in degrees
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))
