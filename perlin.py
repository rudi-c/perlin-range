import ad.admath as am
import numpy as np

cell_size = 100

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def easing5(t):
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 6 * t5 - 15 * t4 + 10 * t3

def unit_vector(angle):
    return (am.cos(angle), am.sin(angle))

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
