# Stimulus parameters.
import numpy as np

N_MOVIES = 5
N_TRIALS = 10
STIMULUS_DURATION = 4 # in seconds
MOVIE_REFRESH_RATE = 60 # in Hz
STIMULUS_LENGTH = MOVIE_REFRESH_RATE * STIMULUS_DURATION
MOVIE_DOWNSAMPLE_FACTOR = 3 # brought down to 20 Hz in paper
CA_SAMPLING_RATE = 50 # Hz
L_RSP = CA_SAMPLING_RATE * STIMULUS_DURATION

PIXELS_PER_DEGREE = 256. / 54. # 54 and 256 given in paper
N_PX = np.array([256, 256])

MAX_ACORR_LAG = 100e-3 # 100 ms in seconds
