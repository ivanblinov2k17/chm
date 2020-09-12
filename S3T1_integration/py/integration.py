import numpy as np
import math
from math import comb


def momA(x, a, alpha, s):
    ans = 0
    for i in range(0, s+1):
        ans += comb(s, i) * a**(s-i) * (x-a)**(i+1-alpha) / (i+1-alpha)
    return ans


def momentA(xl, xr, a, alpha, s):
        return momA(xr, a, alpha, s) - momA(xl,  a, alpha, s)


def momB(x, b, beta, s):
    ans = 0
    for i in range(0, s+1):
        ans += (-1)**(i+1) * comb(s, i) * b**(s-i) * (b - x)**(i+1-beta) / (i+1-beta)
    return ans

def momentB(xl, xr, b, beta, s):
        return momB(xr, b, beta, s) - momB(xl, b, beta, s)


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        return [momentA(xl, xr, a, alpha, s) for s in range(0, max_s + 1)]
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        return [momentB(xl, xr, b, beta, s) for s in range(0, max_s + 1)]

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """
    return - (np.log(abs((s2 - s1)/(s1 - s0)))/np.log(L))


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """
    _moments = moments(len(xs)-1, x0, x1, **kwargs)
    v = np.vander(xs)[:, ::-1].T
    a = np.linalg.solve(v, _moments)
    return a.dot(func(xs))


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    _moments = moments(2*n-1, x0, x1, **kwargs)
    left_moments = []
    for i in range(n):
        mom = []
        for j in range(i, n + i):
            mom.append(_moments[j])
        left_moments.append(mom)
    left_moments = np.array(left_moments)
    right_moments = []
    for i in range(n, 2*n):
        right_moments.append(-_moments[i])
    right_moments = np.array(right_moments)
    a = np.linalg.solve(left_moments, right_moments)
    xs = np.roots(np.append(a, 1)[::-1])
    A = np.linalg.solve(np.vander(xs)[:, ::-1].T, _moments[:n])
    return A.dot(func(xs))


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    bounds = np.linspace(x0, x1, n_intervals+1)
    ans = 0
    for i in range(n_intervals):
        ans += quad(func, bounds[i], bounds[i+1], np.linspace(bounds[i], bounds[i+1], n_nodes), **kwargs)
    return ans


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    steps = []
    h = abs(x1-x0)
    err2 = tol + 1
    L = 2
    n_nodes = 3
    while err2 > tol:
        steps = []
        for i in range(3):
            n_intervals = math.ceil((x1-x0) / (h / L**i))
            steps.append(composite_quad(func, x0, x1, n_intervals, n_nodes))
        m = aitken(steps[0], steps[1], steps[2], L)
        err, err2 = runge(steps[1], steps[2], m, L)
        h = (h/L**i) * math.pow(tol/abs(err2), 1/m)
        if err2 < tol:
            h *= 0.95
            steps = []
            for i in range(3):
                n_intervals = math.ceil((x1 - x0) / (h / L ** i))
                steps.append(composite_quad(func, x0, x1, n_intervals, n_nodes))
    return steps[2], err2
