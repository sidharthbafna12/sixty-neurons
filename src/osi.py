""" osi.py
    Computes the orientation selectivity index and preferred orientation using
    the formulae in 'Robust quantification of orientation selectivity and
    direction selectivity' by Mazurek, Kager and Van Hooser, 2014.
"""

import numpy as np
from params.grating.stimulus_params import *
from response import Response

def _pref_dir_vector(R, orientation_flag=True):
    """ Incoming R is of the shape (8/16, N, 40, 10).
                                     S    N  Ti  Tr
                                            Time Trials
        8 if orientation_flag is set, otherwise 16.
        Calculate orientation/direction sensitivity.
    """
    if orientation_flag:
        assert R.shape[0] == len(ORIENTATIONS)
        # In this space, the basis vector opposite you corresponds to the one
        # at 90 degrees in the stimulus space.
        basis = np.exp(2j * np.radians(ORIENTATIONS))
    else:
        assert R.shape[0] == len(DIRECTIONS)
        basis = np.exp(1j * np.radians(DIRECTIONS))

    R_avgd = np.mean(R, axis=(2, 3))
    
    dir_vecs = np.dot(basis, R_avgd) / np.sum(R_avgd, axis=0)
    return dir_vecs

def selectivity_index(R, orientation_flag=True):
    vecs = _pref_dir_vector(R, orientation_flag)
    return np.absolute(vecs)

def pref_direction(R, orientation_flag=True):
    vecs = _pref_dir_vector(R, orientation_flag)
    if orientation_flag:
        # Divide by 2 because of the way the basis is defined in
        # _pref_dir_vector.
        return (np.angle(vecs, deg=True) % 360.0) / 2.0
    else:
        return np.angle(vecs, deg=True) % 360.0
