""" ssim.py
"""

import numpy as np

def ssim(img_1, img_2):
    assert img_1.shape == img_2.shape

    m1 = np.mean(img_1)
    m2 = np.mean(img_2)
    s1 = np.std(img_1)
    s2 = np.std(img_2)
    x1 = img_1.flatten() - m1
    x2 = img_2.flatten() - m2

    n_px = x1.shape[0]
    s12 = np.dot(x1, x2) / (n_px - 1)

    k1 = 0.01
    k2 = 0.03
    L = 2.0
    C1 = k1 * L
    C2 = k2 * L

    return ((2.0 * m1 * m2 + C1) * (2.0 * s12 + C2))\
        / ((m1 * m1 + m2 * m2 + C1)*(s1 * s1 + s2 * s2 + C2))
