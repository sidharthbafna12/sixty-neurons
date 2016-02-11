""" params.py
    Stores experiment parameters.
    Supposed to be imported everywhere.
"""
import numpy as np
import os

MICE_NAMES = ['A', 'B', 'C', 'D', 'E']
DATA_DIR = 'data/SharedNeuralData'
DATA_LOCS = [os.path.join(DATA_DIR, 'Mouse-%c/Data-Mouse%c.mat' % (c, c))
             for c in MICE_NAMES]
ORI_LOCS = [os.path.join(DATA_DIR, 'Mouse-%c/Solutions/Ori.mat' % c)
            for c in MICE_NAMES]
PLOTS_DIR = 'plots'

# Stimulus parameters.
PIXELS_PER_DEGREE = 21.3 # 1 degree of visual space ~ 21.3 pixels on the screen
SPATIAL_FREQUENCY = 0.03 # cycles per degree
TEMPORAL_FREQUENCY = 10.0 # TODO: put real value here
DIRECTIONS = np.linspace(0.0, 360.0, num=16, endpoint=False) # StimSeq in degs
ORIENTATIONS = DIRECTIONS[:len(DIRECTIONS)/2]

NUM_TRIALS = 10
TRIAL_DURATION = 2 # in seconds
GRAY_SCREEN_TIME = 4 # total duration of stimulus = 6s
SAMPLING_RATE = 20 # in Hz
TOTAL_NUM_SAMPLES = SAMPLING_RATE * (GRAY_SCREEN_TIME + TRIAL_DURATION) \
                                  * NUM_TRIALS \
                                  * len(DIRECTIONS)
NUM_STIMULUS_PRESENTATIONS = NUM_TRIALS * len(DIRECTIONS)
STIMULUS_LENGTH = TOTAL_NUM_SAMPLES / NUM_STIMULUS_PRESENTATIONS

# FFT_WIDTH = SAMPLING_RATE * TRIAL_DURATION
FFT_WIDTH = 64

N_FLAT_CLUSTERS = 3

STA_WINDOW = 5

TIMIT_SAMPLERATE = 16000
TIMIT_STEP = TIMIT_SAMPLERATE / SAMPLING_RATE
TIMIT_L = TIMIT_SAMPLERATE / SAMPLING_RATE # length of single sample window
TIMIT_N_FFT = 128
TIMIT_FRAME_OVERLAP = TIMIT_L - TIMIT_STEP
N_MEL = 24

MAX_ACORR_LAG = 200e-3 # in secs
