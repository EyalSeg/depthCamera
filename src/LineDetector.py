

import numpy as np
import math
from scipy import optimize


# find the position in the line to join.
# x y z are arrays of datapoints in the line, where x is width, y is height and z is depth
# distance is how far from the last datapoint to stand

# returns {next_pos, angle, line_polynomial}
# next_position is the desired position.
# angle is the angle between the desired position and the last position.
# the line_polynomial is the interpolation polynomial representing the line
def find_line_position(x, y, z, distance = 1):

    #interpolation_polynomial_degree =  math.floor((len(x) - 1) * 2 / 3)
    #interpolation_polynomial_degree = (len(x) - 1)
    interpolation_polynomial_degree = 2

    # assume the datapoints are on a straight plane, so we can ignore the y array
    last_position = np.array([x[-1], z[-1]])
    weights = generate_weights(x)

    # copy the last point and move to the right
    # so that the interpolation polynomial will approach infinity slower
#    x_new = x.copy()
#    z_new = z.copy()
#    interpolation_polynomial_degree += 1
#    x_new.append(last_position[0] + distance)
#    z_new.append(last_position[1])

    # TODO: Add weights so that the last datapoints are worth more
    line_polynomial = np.polynomial.polynomial.Polynomial.fit(x, z, interpolation_polynomial_degree, w= weights)

    next_position = find_next_position(line_polynomial, last_position, distance)
    angle = angle_between(next_position, last_position)

    return {'next_pos': next_position, "angle": angle, "line_polynomial": line_polynomial}



## THE CODE FROM HERE ON ASSUMES A 2D PLANE ##
# meaning that the y axis here on is usually the z axis in 3d space (since we assume the datapoints are on a flat plane,
#  height is meaningless)

def generate_weights(data):
    return [(x / (len(data) + 1)) for x in range(1, len(data) + 1)]

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
