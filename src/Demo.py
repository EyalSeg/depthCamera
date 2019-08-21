import LineDetector as ln
import matplotlib.pyplot as plt
import numpy as np

#x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.4]
#z = [1, 1.2, 1.3, 1.7, 2.1, 2.8, 4, 6, 9, 13, 16, 20]

x = [0, 1, 2, 3, 4, 5]
z = [1, 1.2, 1.1, 0.9, 1.2, 1.1]
y = [0, 0, 0]

distance = 1

plt.plot(x, z, 'bo')

result = ln.find_line_position(x, y, z, distance)
poly = result["line_polynomial"]
new_pos = result["next_pos"]

# plot line
poly_x = np.linspace(x[0] - 0.1, x[-1] + distance, 100)
poly_y = poly(poly_x)
plt.plot(poly_x, poly_y)

#plot the new pos
plt.plot(*new_pos, "ro")
plt.annotate('new position', xy=new_pos, xytext=(new_pos[0], new_pos[1] + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.show()