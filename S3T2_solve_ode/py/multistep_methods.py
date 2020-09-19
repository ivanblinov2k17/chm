import numpy as np
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


#  coefficients for Adams methods
adams_coeffs = {
    1: [1],
    2: [-1 / 2, 3 / 2],
    3: [5 / 12, -4 / 3, 23 / 12],
    4: [-3 / 8, 37 / 24, -59 / 24, 55 / 24],
    5: [251 / 720, -637 / 360, 109 / 30, -1387 / 360, 1901 / 720]
}


def adams(func, y_start, T, coeffs, one_step_method: OneStepMethod):
    """
    T: list of timestamps
    coeffs: list of coefficients
    one_step_method: method for initial steps
    return list of t (same as T), list of y
    """
    h = np.abs(T[1] - T[0])
    y = [y_start]
    rightside = [func(T[0], y_start)]
    for i in range(1, len(T)):
        if i < len(coeffs):
            y.append(one_step_method.step(func, T[i - 1], y[i - 1], h))
        else:
            y.append(y[i - 1] + coeffs @ (h * np.array(rightside[-len(coeffs):])))
        rightside.append(func(T[i], y[i]))
    return T, y
