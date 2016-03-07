# Stimulus parameters.
import numpy as np

DIRECTIONS = np.linspace(0.0, 360.0, num=16, endpoint=False) # StimSeq in degs
ORIENTATIONS = DIRECTIONS[:len(DIRECTIONS)/2]

NUM_TRIALS = 10
GRATING_DURATION = 2 # in seconds
GRAY_SCREEN_TIME = 4 # total duration of stimulus = 6s
SAMPLING_RATE = 20 # in Hz
TOTAL_NUM_SAMPLES = SAMPLING_RATE * (GRAY_SCREEN_TIME + GRATING_DURATION) \
                                  * NUM_TRIALS \
                                  * len(DIRECTIONS)
NUM_STIMULUS_PRESENTATIONS = NUM_TRIALS * len(DIRECTIONS)
STIMULUS_LENGTH = TOTAL_NUM_SAMPLES / NUM_STIMULUS_PRESENTATIONS

PIXELS_PER_DEGREE = 21.3 # 1 degree of visual space ~ 21.3 pixels on the screen
CPD = 0.03 # cycles per degree
SPATIAL_FREQUENCY = CPD / PIXELS_PER_DEGREE # cycles per pixel
TEMPORAL_FREQUENCY = 2.0 / SAMPLING_RATE #2.0 Hz -> (temporal) cycles per sample
PHI_T = np.pi / 2 # grating phase shift in radians
PHI_X = 0.0
PHI_Y = 0.0

from pathway_params import FOV
N_PX = np.ceil(FOV * PIXELS_PER_DEGREE).astype(int)