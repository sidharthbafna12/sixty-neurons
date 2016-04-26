""" reliability.py
    Computes response reliability.
"""

import numpy as np

def reliability(rsp):
    S, L, R = rsp.shape
    rel = np.zeros(S)
    for i in range(S):
        r = rsp[i,:,:]
        r -= r.mean()
        for ii in range(R):
            for ij in range(ii+1,R):
                ri, rj = r[:,ii], r[:,ij]
                rel[i] += np.dot(ri, rj) / np.sqrt(np.dot(ri,ri)*np.dot(rj,rj))
        rel[i] /= (R * (R-1)) / 2.0
    return rel

