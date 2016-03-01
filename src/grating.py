import numpy as np

from params.stimulus_params import *

def grating_movie(dirn, N):
    movie = np.zeros((N_PX[0], N_PX[1], N))
    X, Y = np.meshgrid(np.arange(N_PX[1]), np.arange(N_PX[0]))
    X -= N_PX[1] / 2 # centring
    Y -= N_PX[0] / 2
    K = 2 * np.pi * SPATIAL_FREQUENCY
    OMEGA = 2 * np.pi * TEMPORAL_FREQUENCY

    for i in range(N):
        movie[:,:,i] = np.cos(K * X * np.cos(dirn) + K * Y * np.sin(dirn)
                            - (OMEGA * i - PHI_T))
    return movie
