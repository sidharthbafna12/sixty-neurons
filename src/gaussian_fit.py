""" gaussian_fit.py
    Double Gaussian fit for orientation tuning curves.

    Also contains some (unused) single Gaussian fit code for direction tuning.
"""

import numpy as np
from scipy.optimize import curve_fit

""" Wrapped single Gaussian
    Distance wraps around 2pi, that's all.
"""
def _wrapped_distance_1(x, theta):
    dL = np.abs(x - theta)
    dR = 2 * np.pi - dL
    return np.minimum(dL, dR)

def wrapped_single_gaussian(x, theta, sigma, c, w):
    d = _wrapped_distance_1(x, theta)
    return c + w * np.exp(-0.5 * (d ** 2) / (sigma ** 2))

def fit_wrapped_single_gaussian(x, y, p0=None):
    try:
        ps = curve_fit(wrapped_single_gaussian, x, y, p0=p0)
    except RuntimeError:
        ps = (p0, None)

    # Calculate R2 value
    ss_tot = np.sum((y - y.mean()) ** 2)
    yhat = wrapped_single_gaussian(x, *ps[0])
    ss_res = np.sum((yhat - y) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return ps[0], r2

"""
    Assume the function to be a double-wrapped Gaussian with peaks 180 degrees
    apart. Also, the spread of the two lobes is assumed to be the same, as are
    the peaks.
"""
def _wrapped_distance_2(x, theta):
    dL = np.abs(x - theta)
    dR = 2 * np.pi - dL
    d1 = np.minimum(dL, dR)
    d2 = np.pi - d1
    return np.minimum(d1, d2)
    
def wrapped_double_gaussian(x, theta, sigma, c, w):
    d = _wrapped_distance_2(x, theta)
    return c + w * np.exp(-0.5 * (d ** 2) / (sigma ** 2))

def fit_wrapped_double_gaussian(x, y, p0=None):
    try:
        ps = curve_fit(wrapped_double_gaussian, x, y, p0=p0)
    except RuntimeError:
        ps = (p0, None)

    # Calculate R2 value
    ss_tot = np.sum((y - y.mean()) ** 2)
    yhat = wrapped_double_gaussian(x, *ps[0])
    ss_res = np.sum((yhat - y) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return ps[0], r2
