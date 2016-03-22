# Parameters for the pathway simulated from the retina to the primary visual
# cortex.
import numpy as np

# Field of view covered by the screen
FOV = np.array([96.0, 54.0]) # in x and y, in degrees

# degrees of visual space
RGC_CENTRE_WIDTH = 1.0
RGC_SURR_WIDTH = 2.0
RGC_CELL_SPACING = 1.0

RGC_P = 1.0 # p('on' centre) vs p('off' centre)
