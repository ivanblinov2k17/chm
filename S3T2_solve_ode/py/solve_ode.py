import enum
import numpy as np

from S3T2_solve_ode.py.one_step_methods import OneStepMethod
from S3T2_solve_ode.py.one_step_methods import ExplicitEulerMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, func, y_start, ts):
    """
    performs fix-step integration using one-step method
    ts: array of timestamps
    return: list of t's, list of y's
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(func, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, func, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    performs adaptive-step integration using one-step method
    t_span: (t0, t1)
    adapt_type: Runge or Embedded
    tolerances control the error:
        err <= atol
        err <= |y| * rtol
    return: list of t's, list of y's
    """
    y = y_start
    t, t_end = t_span

    ys = [y]
    ts = [t]

    p = method.p + 1
    tol = atol + np.linalg.norm(y) * rtol
    rside0 = func(t, y)
    delta = (1 / max(abs(t), abs(t_end))) ** (p + 1) + np.linalg.norm(rside0) ** (p + 1)
    h1 = (tol / delta) ** (1 / (p + 1))
    u1 = ExplicitEulerMethod().step(func, t, y, h1)
    tnew = t + h1
    rside0 = func(tnew, u1)
    delta = (1 / max(abs(t), abs(t_end))) ** (p + 1) + np.linalg.norm(rside0) ** (p + 1)
    h1new = (tol / delta) ** (1 / (p + 1))
    h = min(h1, h1new)
    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        if adapt_type == AdaptType.RUNGE:
            y1 = method.step(func, t, y, h)
            yhalf = method.step(func, t, y, h / 2)
            y2 = method.step(func, t + h / 2, yhalf, h / 2)
            error = (y2 - y1) / (2 ** p - 1)
            ybetter = y2 + error
        else:
            ybetter, error = method.embedded_step(func, t, y, h)

        if np.linalg.norm(error) < tol:
            ys.append(ybetter)
            ts.append(t + h)
            y = ybetter
            t += h
            print(t)

        h *= (tol / np.linalg.norm(error)) ** (1 / (p + 1)) * 0.9
    return ts, ys