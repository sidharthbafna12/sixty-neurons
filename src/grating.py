import numpy as np

from params.grating.stimulus_params import *

def grating_movie(dirn, N):
    # returns a grating movie as (T,X,Y)
    movie = np.zeros((N, N_PX[1], N_PX[0]))
    X, Y = np.meshgrid(np.arange(N_PX[0]), np.arange(N_PX[1]))
    X -= N_PX[0] / 2 # centring
    Y -= N_PX[1] / 2
    K = 2 * np.pi * SPATIAL_FREQUENCY
    OMEGA = 2 * np.pi * TEMPORAL_FREQUENCY

    for i in range(N):
        movie[i,:,:] = np.cos(K * X * np.cos(dirn) + K * Y * np.sin(dirn)
                            - (OMEGA * i - PHI_T))
    return movie
