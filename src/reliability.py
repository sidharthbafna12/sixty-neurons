""" reliability.py
    Computes response reliability.
"""

import numpy as np

def reliability(rsp):
    S, N, L, R = rsp.shape
    rel = np.zeros((N,S))
    for i_n in range(N):
        for i_s in range(S):
            r = rsp[i_s,i_n,:,:]
            r -= r.mean()
            for ii in range(R):
                for ij in range(ii+1,R):
                    ri, rj = r[:,ii], r[:,ij]
                    if np.all(ri == 0.0) or np.all(rj == 0.0):
                        print i_n, i_s, ii, ij
                    rel[i_n,i_s] += np.dot(ri, rj) \
                                 / np.sqrt(np.dot(ri,ri)*np.dot(rj,rj))
            rel[i_n,i_s] /= (R * (R-1)) / 2.0
    return rel

