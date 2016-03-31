import numpy as np

from params.grating.stimulus_params import *

_grating_movies = {}
def _generate_grating_movies(T):
    def _compute_movie(dirn):
        # returns a grating movie as (Y,X,T)
        movie = np.zeros((N_PX[1], N_PX[0], T))
        X, Y = np.meshgrid(np.arange(N_PX[0]), np.arange(N_PX[1]))
        X -= N_PX[0] / 2 # centring
        Y -= N_PX[1] / 2
        K = 2 * np.pi * SPATIAL_FREQUENCY
        OMEGA = 2 * np.pi * TEMPORAL_FREQUENCY

        for i in range(T):
            movie[:,:,i] = np.cos(K * X * np.cos(dirn) + K * Y * np.sin(dirn)
                                - (OMEGA * i - PHI_T))
        return movie

    for dirn in DIRECTIONS:
        _grating_movies[dirn] = _compute_movie(np.radians(dirn))

def grating_movie(dirn, T):
    if dirn not in _grating_movies:
        _generate_grating_movies(T)
    return _grating_movies[dirn]
