#!/usr/bin/python

import ad
import matplotlib.pyplot as plt
import numpy as np
import random

from math import cos, sin, pi, radians

cell_size = 100

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def easing5(t):
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 6 * t5 - 15 * t4 + 10 * t3

def unit_vector(angle):
    return (cos(angle), sin(angle))

def perlin2D(angle_lower_left, angle_lower_right,
             angle_upper_left, angle_upper_right,
             x, y, easing):
    x_interp = easing(x)
    y_interp = easing(y)
    lower_left = np.dot(unit_vector(angle_lower_left), [x, y])
    lower_right = np.dot(unit_vector(angle_lower_right), [x - 1.0, y])
    upper_left = np.dot(unit_vector(angle_upper_left), [x, y - 1.0])
    upper_right = np.dot(unit_vector(angle_upper_right), [x - 1.0, y - 1.0])
    return lerp(lerp(lower_left, lower_right, x_interp),
                lerp(upper_left, upper_right, x_interp),
                y_interp)

def perlin3x3():
    grid_size = 3
    rows = []
    for grid_i in range(grid_size):
        row = []
        for grid_j in range(grid_size):
            random.seed((grid_i, grid_j))
            a1 = random.uniform(0, 2 * pi)
            random.seed((grid_i + 1, grid_j))
            a2 = random.uniform(0, 2 * pi)
            random.seed((grid_i, grid_j + 1))
            a3 = random.uniform(0, 2 * pi)
            random.seed((grid_i + 1, grid_j + 1))
            a4 = random.uniform(0, 2 * pi)
            print grid_i, grid_j, a1, a2, a3, a4
            def img_gen(i, j):
                return perlin2D(a1, a2, a3, a4,
                                (i % cell_size) / cell_size,
                                (j % cell_size) / cell_size,
                                easing5)
            cell = np.fromfunction(np.vectorize(img_gen),
                                   (cell_size, cell_size))
            row.append(cell)
        rows.append(np.concatenate(row, axis=1))
    img = np.concatenate(rows, axis=0)
    return img

plt.figure(1)
plt.imshow(perlin3x3(), cmap='Greys')
plt.colorbar()

a1 = pi / 4
a2 = 3 * pi / 4
a3 = - pi / 4
a4 = - 3 * pi / 4

def img_gen(i, j):
    return perlin2D(a1, a2, a3, a4, i / cell_size, j / cell_size, easing5)

img = np.fromfunction(np.vectorize(img_gen), (cell_size, cell_size))

plt.figure(2)
plt.imshow(img)
plt.colorbar()
plt.show()
