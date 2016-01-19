""" params.py
    Stores experiment parameters.
    Supposed to be imported everywhere.
"""
import numpy as np
MICE_NAMES = ['A', 'B', 'C', 'D', 'E']

# Stimulus parameters.
PIXELS_PER_DEGREE = 21.3 # 1 degree of visual space ~ 21.3 pixels on the screen
SPATIAL_FREQUENCY = 0.03 # cycles per degree
DIRECTIONS = np.linspace(0.0, 360.0, num=16, endpoint=False) # StimSeq in degs
ORIENTATIONS = DIRECTIONS[:len(DIRECTIONS)/2]

NUM_TRIALS = 10
DIR_PRESENT_TIME = 2 # in seconds
GRAY_SCREEN_TIME = 4 # total duration of stimulus = 6s
SAMPLING_RATE = 20 # in Hz
TOTAL_NUM_SAMPLES = SAMPLING_RATE * (GRAY_SCREEN_TIME + DIR_PRESENT_TIME) \
                                  * NUM_TRIALS \
                                  * len(DIRECTIONS)
NUM_STIMULUS_PRESENTATIONS = NUM_TRIALS * len(DIRECTIONS)
STIMULUS_LENGTH = TOTAL_NUM_SAMPLES / NUM_STIMULUS_PRESENTATIONS

# FFT_WIDTH = SAMPLING_RATE * DIR_PRESENT_TIME
FFT_WIDTH = 64

DATA_DIR = 'data/SharedNeuralData'
PLOTS_DIR = 'plots'

# Plotting parameters
PLOTTING_AVERAGE_RESPONSE = False
PLOTTING_DIRWISE = False
PLOTTING_CELLWISE = False
