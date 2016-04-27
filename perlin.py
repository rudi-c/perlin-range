import autograd.numpy as np

from autograd import grad

cell_size = 100

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def easing3(t):
    t2 = t * t
    t3 = t2 * t
    return 3 * t2 - 2 * t3

def easing5(t):
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 6 * t5 - 15 * t4 + 10 * t3

def unit_vector(angle):
    return (np.cos(angle), np.sin(angle))

def perlin2D(easing):
    def f(angle_lower_left, angle_lower_right,
          angle_upper_left, angle_upper_right, x, y):
        x_interp = easing(x)
        y_interp = easing(y)
        lower_left = np.dot(unit_vector(angle_lower_left), [x, y])
        lower_right = np.dot(unit_vector(angle_lower_right), [x - 1.0, y])
        upper_left = np.dot(unit_vector(angle_upper_left), [x, y - 1.0])
        upper_right = np.dot(unit_vector(angle_upper_right), [x - 1.0, y - 1.0])
        return lerp(lerp(lower_left, lower_right, x_interp),
                    lerp(upper_left, upper_right, x_interp),
                    y_interp)
    return f

def perlin2D_gradient_magnitude(easing):
    # Take the gradient w.r.t to an _array_ of arguments.
    gradient_function = grad(lambda args: perlin2D(easing)(*args))
    def f(args):
        gradient = list(gradient_function(args))
        return np.sqrt(np.dot(gradient, gradient))
    return f
