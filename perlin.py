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

def unit_vector3d(theta, psi):
    return (np.cos(theta) * np.sin(psi), np.sin(theta) * np.sin(psi), np.cos(psi))

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

def perlin3D(easing):
    def f(theta_bottom_lower_left,  psi_bottom_lower_left,
          theta_bottom_lower_right, psi_bottom_lower_right,
          theta_bottom_upper_left,  psi_bottom_upper_left,
          theta_bottom_upper_right, psi_bottom_upper_right,
          theta_top_lower_left,  psi_top_lower_left,
          theta_top_lower_right, psi_top_lower_right,
          theta_top_upper_left,  psi_top_upper_left,
          theta_top_upper_right, psi_top_upper_right,
          x, y, z):
        x_interp = easing(x)
        y_interp = easing(y)
        z_interp = easing(z)
        bottom_lower_left = np.dot(
            unit_vector3d(theta_bottom_lower_left, psi_bottom_lower_left), [x, y, z])
        bottom_lower_right = np.dot(
            unit_vector3d(theta_bottom_lower_right, psi_bottom_lower_right), [x - 1.0, y, z])
        bottom_upper_left = np.dot(
            unit_vector3d(theta_bottom_upper_left, psi_bottom_upper_left), [x, y - 1.0, z])
        bottom_upper_right = np.dot(
            unit_vector3d(theta_bottom_upper_right, psi_bottom_upper_right), [x - 1.0, y - 1.0, z])
        top_lower_left = np.dot(
            unit_vector3d(theta_top_lower_left, psi_top_lower_left), [x, y, 1.0 - z])
        top_lower_right = np.dot(
            unit_vector3d(theta_top_lower_right, psi_top_lower_right), [x - 1.0, y, 1.0 - z])
        top_upper_left = np.dot(
            unit_vector3d(theta_top_upper_left, psi_top_upper_left), [x, y - 1.0, 1.0 - z])
        top_upper_right = np.dot(
            unit_vector3d(theta_top_upper_right, psi_top_upper_right), [x - 1.0, y - 1.0, 1.0 - z])
        return lerp(lerp(lerp(bottom_lower_left, bottom_lower_right, x_interp),
                         lerp(bottom_upper_left, bottom_upper_right, x_interp),
                         y_interp),
                    lerp(lerp(top_lower_left, top_lower_right, x_interp),
                         lerp(top_upper_left, top_upper_right, x_interp),
                         y_interp),
                    z_interp)

    return f

def gradient_magnitude(fn):
    # Take the gradient w.r.t to an _array_ of arguments.
    gradient_function = grad(lambda args: fn(*args))
    def f(args):
        gradient = list(gradient_function(args))
        return np.sqrt(np.dot(gradient, gradient))
    return f

def perlin2D_gradient_magnitude(easing):
    return gradient_magnitude(perlin2D(easing))

def perlin3D_gradient_magnitude(easing):
    return gradient_magnitude(perlin3D(easing))
