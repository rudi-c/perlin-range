#!/usr/bin/python

import math
import random

from autograd import grad
from math import pi, degrees, radians
from scipy.optimize import minimize

from perlin import *

# Return a theta and psi representing a vector on a unit sphere distributed
# such that they correctly distribute points over a unit sphere.
# http://mathworld.wolfram.com/SpherePointPicking.html
def sphere_random():
    return (random.uniform(0, 2 * pi), math.acos(random.uniform(-1, 1)))

def perlin2d_gradient_descent(tolerance, iterations, fn):
    results = {}

    # optimize.minimize and autograd.grad require a function that
    # takes in an array of arguments
    gradient = grad(fn)

    for i in range(iterations):
        if i % 20 == 0:
            print "Minimization %d" % i

        # Random starting values
        a_ll = random.uniform(0, 2 * pi)
        a_lr = random.uniform(0, 2 * pi)
        a_ul = random.uniform(0, 2 * pi)
        a_ur = random.uniform(0, 2 * pi)

        x    = random.uniform(0, 1)
        y    = random.uniform(0, 1)

        x0 = [a_ll, a_lr, a_ul, a_ur, x, y]

        result = minimize(
            fn,
            x0,
            bounds=((0, 2 * pi), (0, 2 * pi), (0, 2 * pi), (0, 2 * pi), (0, 1), (0, 1)),
            method='L-BFGS-B',
            jac=gradient,
            tol=tolerance)

        key = tuple(round(var, 3) for var in result.x)
        if key in results:
            results[key] = min(results[key], result.fun)
        else:
            results[key] = result.fun

    # Return the best results
    return sorted((val, key) for key, val in results.iteritems())

def perlin3d_gradient_descent(tolerance, iterations, fn):
    results = {}

    # optimize.minimize and autograd.grad require a function that
    # takes in an array of arguments
    gradient = grad(fn)

    for i in range(iterations):
        if i % 20 == 0:
            print "Minimization %d" % i

        # Random starting values
        t_bll, p_bll = sphere_random()
        t_blr, p_blr = sphere_random()
        t_bul, p_bul = sphere_random()
        t_bur, p_bur = sphere_random()
        t_tll, p_tll = sphere_random()
        t_tlr, p_tlr = sphere_random()
        t_tul, p_tul = sphere_random()
        t_tur, p_tur = sphere_random()

        x    = random.uniform(0, 1)
        y    = random.uniform(0, 1)
        z    = random.uniform(0, 1)

        x0 = [t_bll, p_bll, t_blr, p_blr, t_bul, p_bul, t_bur, p_bur,
              t_tll, p_tll, t_tlr, p_tlr, t_tul, p_tul, t_tur, p_tur,
              x, y, z]

        result = minimize(
            fn,
            x0,
            bounds=((0, 2 * pi), (0, pi)) * 8 + ((0, 1),) * 3,
            method='L-BFGS-B',
            jac=gradient,
            tol=tolerance)

        key = tuple(round(var, 3) for var in result.x)
        if key in results:
            results[key] = min(results[key], result.fun)
        else:
            results[key] = result.fun

    # Return the best results
    return sorted((val, key) for key, val in results.iteritems())

def perlin2d_gradient_ascent(tolerance, iterations, fn):
    best = perlin2d_gradient_descent(tolerance, iterations, lambda args: -fn(args))

    largest_smaller = next(index for (index, (val, key))
                                 in enumerate(best)
                                 if round(val, 5) > round(best[0][0], 5))

    for val, (a_ll, a_lr, a_ul, a_ur, x, y) in best[:max(6, largest_smaller + 2)]:
        print -val, (round(degrees(a_ll), 2),
                     round(degrees(a_lr), 2),
                     round(degrees(a_ul), 2),
                     round(degrees(a_ur), 2),
                     x, y)
        print "    also %.3fpi %.3fpi %.3fpi %.3fpi" % (a_ll / pi, a_lr / pi, a_ul / pi, a_ur / pi)

def perlin3d_gradient_ascent(tolerance, iterations, fn):
    best = perlin3d_gradient_descent(tolerance, iterations, lambda args: -fn(args))

    largest_smaller = next(index for (index, (val, key))
                                 in enumerate(best)
                                 if round(val, 5) > round(best[0][0], 5))

    for val, key in best[:max(6, largest_smaller + 2)]:
        t_bll, p_bll, t_blr, p_blr, t_bul, p_bul, t_bur, p_bur = key[0:8]
        t_tll, p_tll, t_tlr, p_tlr, t_tul, p_tul, t_tur, p_tur = key[8:16]
        x, y, z = key[16:19]
        print -val, [round(degrees(angle), 2) for angle in key[0:16]], (x, y, z)
        print "    also {}pi".format([round(angle / pi, 3) for angle in key[0:16]])

print ">>> Maximize 2D Perlin Noise with 3rd order interpolant"
perlin2d_gradient_ascent(1e-10, 100, lambda args: perlin2D(easing3)(*args))
print ">>> Maximize 2D Perlin Noise with 5th order interpolant"
perlin2d_gradient_ascent(1e-10, 100, lambda args: perlin2D(easing5)(*args))
print ">>> Maximize 2D Perlin Noise's Gradient Magnitude with 3rd order interpolant"
perlin2d_gradient_ascent(1e-10, 200, perlin2D_gradient_magnitude(easing3))
print ">>> Maximize 2D Perlin Noise's Gradient Magnitude with 5th order interpolant"
perlin2d_gradient_ascent(1e-10, 200, perlin2D_gradient_magnitude(easing5))

print ">>> Maximize 3D Perlin Noise with 3rd order interpolant"
perlin3d_gradient_ascent(1e-10, 200, lambda args: perlin3D(easing3)(*args))
print ">>> Maximize 3D Perlin Noise with 5th order interpolant"
perlin3d_gradient_ascent(1e-10, 200, lambda args: perlin3D(easing5)(*args))
print ">>> Maximize 3D Perlin Noise's Gradient Magnitude with 3rd order interpolant"
perlin3d_gradient_ascent(1e-10, 400, perlin3D_gradient_magnitude(easing3))
print ">>> Maximize 3D Perlin Noise's Gradient Magnitude with 5th order interpolant"
perlin3d_gradient_ascent(1e-10, 400, perlin3D_gradient_magnitude(easing5))
