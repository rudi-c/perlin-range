#!/usr/bin/python

import matplotlib.pyplot as plt
import random

from math import pi, radians

from perlin import *

def perlin3x3(fn):
    grid_size = 3
    cell_size3x3 = cell_size / grid_size
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
                return fn(
                    a_ll, a_lr, a_ul, a_ur,
                    (i % cell_size3x3) / cell_size3x3,
                    (j % cell_size3x3) / cell_size3x3)
            cell = np.fromfunction(np.vectorize(img_gen),
                                   (cell_size3x3, cell_size3x3))
            row.append(cell)
        rows.append(np.concatenate(row, axis=1))
    img = np.concatenate(rows, axis=0)
    return img

def plot_perlin_2d(fig, easing, a_ll, a_lr, a_ul, a_ur):
    def img_gen(i, j):
        return perlin2D(easing)(a_ll, a_lr, a_ul, a_ur, i / cell_size, j / cell_size)

    def grad_gen(i, j):
        return perlin2D_gradient_magnitude(easing)([a_ll, a_lr, a_ul, a_ur, i / cell_size, j / cell_size])

    img = np.fromfunction(np.vectorize(img_gen), (cell_size, cell_size))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.colorbar()

    img = np.fromfunction(np.vectorize(grad_gen), (cell_size, cell_size))

    fig.add_subplot(1, 2, 2)
    plt.imshow(img)
    plt.colorbar()

fig = plt.figure(1, figsize=(15,6))
fig.add_subplot(1, 2, 1)
plt.imshow(perlin3x3(perlin2D(easing5)), cmap='Greys')
plt.colorbar()

fig.add_subplot(1, 2, 2)
plt.imshow(perlin3x3(lambda *args: perlin2D_gradient_magnitude(easing5)(args)))
plt.colorbar()

fig = plt.figure(2, figsize=(15,6))
plot_perlin_2d(fig, easing5, pi / 4, 3 * pi / 4, - pi / 4, - 3 * pi / 4)

fig = plt.figure(3, figsize=(15,6))
plot_perlin_2d(fig, easing3, radians(141.98), radians(345.72), radians(274.1), radians(60.79))

fig = plt.figure(4, figsize=(15,6))
plot_perlin_2d(fig, easing5, radians(245.51), radians(114.48), radians(114.48), radians(245.51))

plt.show()
