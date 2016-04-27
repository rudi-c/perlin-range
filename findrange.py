#!/usr/bin/python

import random

from autograd import grad
from math import pi, radians
from scipy.optimize import minimize

from perlin import *

def perlin2d_gradient_descent(tolerance, easing):
    results = {}

    perlin2D_args = lambda args: perlin2D(easing)(*tuple(args))
    gradient = grad(perlin2D_args)

    for i in range(100):
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
            perlin2D_args,
            x0,
            bounds=((0, 2 * pi), (0, 2 * pi), (0, 2 * pi), (0, 2 * pi), (0, 1), (0, 1)),
            method='L-BFGS-B',
            jac=gradient,
            tol=tolerance)

        key = tuple(round(var, 4) for var in result.x)
        if key in results:
            results[key] = min(results[key], result.fun)
        else:
            results[key] = result.fun

    # Print the top 10 best results
    best = sorted((val, key) for key, val in results.iteritems())
    for val, key in best[:10]:
        print val, key

perlin2d_gradient_descent(1e-10, easing5)
