#!/usr/bin/python

import math
import random
import sys

import numpy as np
import matplotlib.pyplot as plt

from autograd import grad
from math import pi, degrees, radians

from perlin import *

def uniform3_sample(iterations, fn):
    results = []

    for i in xrange(iterations):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)
        val = fn([x, y, z])
        results.append(val)

    return results

def sin_noise(x, y, z):
    return 0.03 * np.sin(x * 50.0) * np.sin(y * 50.0) * np.sin(z * 50.0)

sin_noise_grad = gradient_magnitude(sin_noise)

def perlin2D_sample(angle_iterations, pos_iterations, fn):
    """Get samples from 'fn', a function taking parameters of 2D Perlin Noise."""

    results = []

    for i in xrange(angle_iterations):
        if i % 100 == 0:
            sys.stdout.write("Angle iterations %d\r" % i)
            sys.stdout.flush()

        # Random values for the gradient vector at each corner
        # lower/upper left/right
        a_ll = random.uniform(0, 2 * pi)
        a_lr = random.uniform(0, 2 * pi)
        a_ul = random.uniform(0, 2 * pi)
        a_ur = random.uniform(0, 2 * pi)

        for i in xrange(pos_iterations):
            # Random starting positions
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)

            x0 = [a_ll, a_lr, a_ul, a_ur, x, y]

            val = fn(x0)

            if val:
                results.append(val)

    return results

def perlin3D_sample(angle_iterations, pos_iterations, fn):
    """Get samples from 'fn', a function taking parameters of 3D Perlin Noise."""

    results = []

    for i in xrange(angle_iterations):
        if i % 100 == 0:
            sys.stdout.write("Angle iterations %d\r" % i)
            sys.stdout.flush()

        # Random values for the gradient vector at each corner
        # bottom/top lower/upper left/right
        t_bll, p_bll = sphere_random()
        t_blr, p_blr = sphere_random()
        t_bul, p_bul = sphere_random()
        t_bur, p_bur = sphere_random()
        t_tll, p_tll = sphere_random()
        t_tlr, p_tlr = sphere_random()
        t_tul, p_tul = sphere_random()
        t_tur, p_tur = sphere_random()

        for i in xrange(pos_iterations):
            # Random starting positions
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            z = random.uniform(0, 1)

            x0 = [t_bll, p_bll, t_blr, p_blr, t_bul, p_bul, t_bur, p_bur,
                  t_tll, p_tll, t_tlr, p_tlr, t_tul, p_tul, t_tur, p_tur,
                  x, y, z]

            val = fn(x0)

            if val:
                results.append(val)

    return results

def plot_distribution(results, bin_count, histogram_range):
    print np.percentile(results, [0, 5, 10, 20, 50, 80, 90, 95, 100])
    plt.hist(results, bins=bin_count, range=histogram_range, normed=True, cumulative=True)
    plt.show()

def perlin2D_and_gradient(easing):
    perlin = lambda args: perlin2D(easing)(*args)
    grad = perlin2D_gradient_magnitude(easing)
    def f(args):
        g = grad(args)
        if g > 1.0:
            return (perlin(args), grad(args))
    return f

print ">>> Sin noise distribution"
results = uniform3_sample(10000, lambda args: sin_noise(*args))
plot_distribution(results, 100, (-0.03, 0.03))

print ">>> Sin noise gradient distribution"
results = uniform3_sample(10000, sin_noise_grad)
plot_distribution(results, 100, (0, 1.0))

print ">>> Perlin 2D distribution"
results = perlin2D_sample(1000, 100, lambda args: perlin2D(easing5)(*args))
plot_distribution(results, 100, (-0.75, 0.75))

print ">>> Perlin 2D gradient distribution"
results = perlin2D_sample(1000, 100, perlin2D_gradient_magnitude(easing5))
plot_distribution(results, 100, (0, 2.2))

print ">>> Perlin 3D distribution"
results = perlin3D_sample(1000, 100, lambda args: perlin3D(easing5)(*args))
plot_distribution(results, 100, (-0.90, 0.90))

print ">>> Perlin 3D gradient distribution"
results = perlin3D_sample(1000, 100, perlin3D_gradient_magnitude(easing5))
plot_distribution(results, 100, (0, 2.8))

print ">>> Improved Perlin 2D gradient distribution"
results = perlin3D_sample(1000, 100, perlin2D_gradient_magnitude(easing5, True))
plot_distribution(results, 100, (0, 2.8))

print ">>> Improved Perlin 3D gradient distribution"
results = perlin3D_sample(1000, 100, perlin3D_gradient_magnitude(easing5, True))
plot_distribution(results, 100, (0, 2.8))

print ">>> Value v.s. gradient"
results = perlin2D_sample(1000, 20, perlin2D_and_gradient(easing5))
plt.hist2d(*zip(*results), bins=[50, 50], range=[(-0.75, 0.75), (0, 2.2)])
plt.show()
