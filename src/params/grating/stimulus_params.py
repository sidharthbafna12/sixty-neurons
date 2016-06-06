# Stimulus parameters.
import numpy as np

DIRECTIONS = np.linspace(0.0, 360.0, num=16, endpoint=False) # StimSeq in degs
ORIENTATIONS = DIRECTIONS[:len(DIRECTIONS)/2]

N_MOVIES = len(DIRECTIONS)
N_TRIALS = 10
GRATING_DURATION = 2 # in seconds
GRAY_SCREEN_TIME = 4 # total duration of stimulus = 6s
CA_SAMPLING_RATE = 20 # in Hz
L_RSP = GRATING_DURATION * CA_SAMPLING_RATE

# TODO : Remove this later.
PIXELS_PER_DEGREE = 5.0 # 1 degree of visual space ~ 21.3 pixels on the screen
# PIXELS_PER_DEGREE = 21.3 # 1 degree of visual space ~ 21.3 pixels on the screen

CPD = 0.03 # cycles per degree
SPATIAL_FREQUENCY = CPD / PIXELS_PER_DEGREE # cycles per pixel
TEMPORAL_FREQUENCY = 2.0 / CA_SAMPLING_RATE #2.0Hz->(temporal) cycles per sample
PHI_T = np.pi / 2 # grating phase shift in radians
PHI_X = 0.0
PHI_Y = 0.0

from pathway_params import FOV
N_PX = np.ceil(FOV * PIXELS_PER_DEGREE).astype(int)
