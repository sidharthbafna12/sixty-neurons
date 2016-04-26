""" correlation.py
    Definitions for signal and noise correlations.
"""

import numpy as np

def signal_correlation(rsp_orig):
    S,N,L,R = rsp_orig.shape
    rsp = rsp_orig - rsp_orig.mean()

    avg = np.mean(rsp, axis=3)

    corr = np.zeros((N,N))
    for i in range(N):
        ri = avg[:,i,:]
        for j in range(N):
            rj = avg[:,j,:]
            c = 0.0
            for i_s in range(S):
                c += np.dot(ri[i_s,:], rj[i_s,:]) \
                    / np.sqrt(np.dot(ri[i_s,:],ri[i_s,:])
                            * np.dot(rj[i_s,:],rj[i_s,:]))
            c /= float(S)
            corr[i,j] = c
    return corr

def noise_correlation(rsp_orig):
    S,N,L,R = rsp_orig.shape

    mean_rsps = np.reshape([np.mean(rsp_orig[:,i,:,:]) for i in range(N)],
                           (1,N,1,1))
    rsp = rsp_orig - np.tile(mean_rsps, (S, 1, L, R))

    corr = np.zeros((N,N))
    for i in range(N):
        ri = rsp[:,i,:,:]
        for j in range(N):
            rj = rsp[:,j,:,:]
            c = 0.0
            for i_s in range(S):
                for i_r in range(R):
                    c += np.dot(ri[i_s,:,i_r], rj[i_s,:,i_r]) \
                        / np.sqrt(np.dot(ri[i_s,:,i_r],ri[i_s,:,i_r])
                                * np.dot(rj[i_s,:,i_r],rj[i_s,:,i_r]))
            c /= float(S * R)
            corr[i,j] = c
    return corr

