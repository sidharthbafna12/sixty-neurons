import numpy as np
from scipy.optimize import curve_fit

"""
    Assume the function to be a double-wrapped Gaussian with peaks 180 degrees
    apart. Also, the spread of the two lobes is assumed to be the same, as are
    the peaks.
    So the function becomes:
        y = c + w * (exp(-d1 * 0.5 / sigma**2) + exp(-d2 * 0.5 / sigma**2))
    where d1 and d2 are the (wrapped) distances from the centres.
"""
def wrapped_distance(x, theta, sigma, c, w):
    dL = np.abs(x - theta)
    dR = 2 * np.pi - dL
    d1 = np.minimum(dL, dR)
    d2 = np.pi - d1
    return np.minimum(d1, d2)
    
def wrapped_double_gaussian(x, theta, sigma, c, w):
    d = wrapped_distance(x, theta, sigma, c, w)
    return c + w * np.exp(-0.5 * (d ** 2) / (sigma ** 2))

def fit_wrapped_double_gaussian(x, y, p0=None):
    try:
        ps = curve_fit(wrapped_double_gaussian, x, y, p0=p0)
    except RuntimeError:
        ps = (p0, None)

    yhat = wrapped_double_gaussian(x, *ps[0])

    # Calculate R2 value
    ss_tot = np.sum((y - y.mean()) ** 2)
    yhat = wrapped_double_gaussian(x, *ps[0])
    ss_res = np.sum((yhat - y) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return ps[0], r2
