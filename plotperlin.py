#!/usr/bin/python

import matplotlib.pyplot as plt
import random

from math import pi, radians

from perlin import *

def perlin3x3():
    grid_size = 3
    rows = []
    for grid_i in range(grid_size):
        row = []
        for grid_j in range(grid_size):
            random.seed((grid_i, grid_j))
            a_ll = random.uniform(0, 2 * pi)
            random.seed((grid_i + 1, grid_j))
            a_lr = random.uniform(0, 2 * pi)
            random.seed((grid_i, grid_j + 1))
            a_ul = random.uniform(0, 2 * pi)
            random.seed((grid_i + 1, grid_j + 1))
            a_ur = random.uniform(0, 2 * pi)
            print grid_i, grid_j, a_ll, a_lr, a_ul, a_ur
            def img_gen(i, j):
                return perlin2D(easing5)(
                    a_ll, a_lr, a_ul, a_ur,
                    (i % cell_size) / cell_size,
                    (j % cell_size) / cell_size)
            cell = np.fromfunction(np.vectorize(img_gen),
                                   (cell_size, cell_size))
            row.append(cell)
        rows.append(np.concatenate(row, axis=1))
    img = np.concatenate(rows, axis=0)
    return img

plt.figure(1)
plt.imshow(perlin3x3(), cmap='Greys')
plt.colorbar()

a_ll = pi / 4
a_lr = 3 * pi / 4
a_ul = - pi / 4
a_ur = - 3 * pi / 4

def img_gen(i, j):
    return perlin2D(easing5)(a_ll, a_lr, a_ul, a_ur, i / cell_size, j / cell_size)

img = np.fromfunction(np.vectorize(img_gen), (cell_size, cell_size))

plt.figure(2)
plt.imshow(img)
plt.colorbar()

def grad_gen(i, j):
    return perlin2D_gradient_magnitude(easing5)([a_ll, a_lr, a_ul, a_ur, i / cell_size, j / cell_size])

img = np.fromfunction(np.vectorize(grad_gen), (cell_size, cell_size))

plt.figure(3)
plt.imshow(img)
plt.colorbar()

plt.show()
