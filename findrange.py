#!/usr/bin/python

import random

from autograd import grad
from math import pi, degrees, radians
from scipy.optimize import minimize

from perlin import *

def perlin2d_gradient_descent(tolerance, fn):
    results = {}

    # optimize.minimize and autograd.grad require a function that
    # takes in an array of arguments
    gradient = grad(fn)

    for i in range(200):
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

def perlin2d_gradient_ascent(tolerance, fn):
    best = perlin2d_gradient_descent(tolerance, lambda args: -fn(args))
    for val, (a_ll, a_lr, a_ul, a_ur, x, y) in best[:10]:
        print -val, (round(degrees(a_ll), 2),
                     round(degrees(a_lr), 2),
                     round(degrees(a_ul), 2),
                     round(degrees(a_ur), 2),
                     x, y)
        print "    also %.3fpi %.3fpi %.3fpi %.3fpi" % (a_ll / pi, a_lr / pi, a_ul / pi, a_ur / pi)

perlin2d_gradient_ascent(1e-10, perlin2D_gradient_magnitude(easing5))
perlin2d_gradient_ascent(1e-10, lambda args: perlin2D(easing5)(*args))
