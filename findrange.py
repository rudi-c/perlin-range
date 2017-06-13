#!/usr/bin/python

import math
import random
import sys

from autograd import grad
from math import pi, degrees, radians
from scipy.optimize import minimize

from perlin import *

def sort_gradient_results(results, digits=3):
    """Given a list of optimize.result objects, group the similar ones
    together (i.e. ended at the same local maximum/minimum), and sort
    by largest/smallest maximum/minimum"""

    processed = {}

    for result in results:
        # key is a tuple of arguments that locally maximize/minimize
        # the function. It is rounded since the exact point will not be hit
        key = tuple(round(var, digits) for var in result.x)
        if key in processed:
            processed[key] = min(processed[key], result.fun)
        else:
            processed[key] = result.fun

    # Sort by the best results.
    return sorted((val, key) for key, val in processed.iteritems())

def perlin2d_gradient_descent(tolerance, iterations, fn):
    """Find local minima in 'fn', a function taking parameters of
    2D Perlin Noise, using 'iterations' starting point."""

    results = []

    # optimize.minimize and autograd.grad require a function that
    # takes in an array of arguments
    gradient = grad(fn)

    for i in range(iterations):
        if i % 10 == 0:
            sys.stdout.write("Minimization %d\r" % i)
            sys.stdout.flush()

        # Random starting positions
        x    = random.uniform(0, 1)
        y    = random.uniform(0, 1)

        # Random values for the gradient vector at each corner
        # lower/upper left/right
        a_ll = random.uniform(0, 2 * pi)
        a_lr = random.uniform(0, 2 * pi)
        a_ul = random.uniform(0, 2 * pi)
        a_ur = random.uniform(0, 2 * pi)

        x0 = [a_ll, a_lr, a_ul, a_ur, x, y]

        result = minimize(
            fn,
            x0,
            bounds=((0, 2 * pi),) * 4 + ((0, 1),) * 2,
            method='L-BFGS-B',
            jac=gradient,
            tol=tolerance)

        results.append(result)

    return results

def perlin3d_gradient_descent(tolerance, iterations, fn):
    """Find local minima in 'fn', a function taking parameters of
    3D Perlin Noise, using 'iterations' starting point."""

    results = []

    # optimize.minimize and autograd.grad require a function that
    # takes in an array of arguments
    gradient = grad(fn)

    for i in range(iterations):
        if i % 10 == 0:
            sys.stdout.write("Minimization %d\r" % i)
            sys.stdout.flush()

        # Random starting positions
        x    = random.uniform(0, 1)
        y    = random.uniform(0, 1)
        z    = random.uniform(0, 1)

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

        results.append(result)

    return results

def perlin2d_gradient_ascent(tolerance, iterations, fn):
    """Find local maxima in 'fn', a function taking parameters of
    2D Perlin Noise, using 'iterations' starting point."""
    return perlin2d_gradient_descent(tolerance, iterations, lambda args: -fn(args))

def perlin3d_gradient_ascent(tolerance, iterations, fn):
    """Find local maxima in 'fn', a function taking parameters of
    3D Perlin Noise, using 'iterations' starting point."""
    return perlin3d_gradient_descent(tolerance, iterations, lambda args: -fn(args))

def perlin2d_gradient_ascent_best(tolerance, iterations, fn):
    results = sort_gradient_results(perlin2d_gradient_ascent(tolerance, iterations, fn))

    # Get the number of results that tie for #1
    ties = next(index for (index, (val, key))
                                 in enumerate(results)
                                 if round(val, 5) > round(results[0][0], 5))

    number_of_results_to_print = max(6, ties + 2)
    for val, key in results[:number_of_results_to_print]:
        print "Value %f at position %s" % (-val, key[4:6])
        print "      with angles %s" % [round(degrees(angle), 2) for angle in key[0:4]]
        print "      equivalently %spi" % [round(angle / pi, 3) for angle in key[0:4]]

def perlin3d_gradient_ascent_best(tolerance, iterations, fn):
    results = sort_gradient_results(perlin3d_gradient_ascent(tolerance, iterations, fn))

    # Get the number of results that tie for #1
    ties = next(index for (index, (val, key))
                                 in enumerate(results)
                                 if round(val, 5) > round(results[0][0], 5))

    number_of_results_to_print = max(6, ties + 2)
    for val, key in results[:number_of_results_to_print]:
        print "Value %f at position %s" % (-val, key[16:19])
        print "      with angles %s" % [round(degrees(angle), 2) for angle in key[0:8]]
        print "      equivalently %spi" % [round(angle / pi, 3) for angle in key[8:16]]

print ">>> Maximize 2D Perlin Noise with 3rd order interpolant"
perlin2d_gradient_ascent_best(1e-10, 50, lambda args: perlin2D(easing3)(*args))
print ">>> Maximize 2D Perlin Noise with 5th order interpolant"
perlin2d_gradient_ascent_best(1e-10, 50, lambda args: perlin2D(easing5)(*args))
print ">>> Maximize 2D Perlin Noise's Gradient Magnitude with 3rd order interpolant"
perlin2d_gradient_ascent_best(1e-10, 100, perlin2D_gradient_magnitude(easing3))
print ">>> Maximize 2D Perlin Noise's Gradient Magnitude with 5th order interpolant"
perlin2d_gradient_ascent_best(1e-10, 100, perlin2D_gradient_magnitude(easing5))

print ">>> Maximize 3D Perlin Noise with 3rd order interpolant"
perlin3d_gradient_ascent_best(1e-10, 100, lambda args: perlin3D(easing3)(*args))
print ">>> Maximize 3D Perlin Noise with 5th order interpolant"
perlin3d_gradient_ascent_best(1e-10, 100, lambda args: perlin3D(easing5)(*args))
print ">>> Maximize 3D Perlin Noise's Gradient Magnitude with 3rd order interpolant"
perlin3d_gradient_ascent_best(1e-10, 200, perlin3D_gradient_magnitude(easing3))
print ">>> Maximize 3D Perlin Noise's Gradient Magnitude with 5th order interpolant"
perlin3d_gradient_ascent_best(1e-10, 200, perlin3D_gradient_magnitude(easing5))
