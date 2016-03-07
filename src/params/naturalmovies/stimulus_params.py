# Stimulus parameters.
import numpy as np

NUM_MOVIES = 5
NUM_TRIALS = 10
STIMULUS_DURATION = 4 # in seconds
SAMPLING_RATE = 50 # in Hz
STIMULUS_LENGTH = SAMPLING_RATE * STIMULUS_DURATION

PIXELS_PER_DEGREE = 256. / 54. # 54 and 256 given in paper
N_PX = np.array([256, 256])
