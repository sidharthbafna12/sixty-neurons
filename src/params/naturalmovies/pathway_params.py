# Parameters for the pathway simulated from the retina to the primary visual
# cortex.
import numpy as np

# Field of view covered by the screen
# Paper says 96x54, but 256x256 means that the image can't really cover
# different number of degrees in different directions. Maybe it was pillarboxed.
# Anyway, I am taking the lower of the two. Should be fine for now.
# FOV = np.array([96.0, 54.0]) # in x and y, in degrees
FOV = np.array([54.0, 54.0]) # in x and y, in degrees

# degrees of visual space
RGC_CENTRE_WIDTH = 1.0
RGC_SURR_WIDTH = 2.0
RGC_CELL_SPACING = 0.5
RGC_N_CELLS = 1 + (FOV / RGC_CELL_SPACING).astype(int)

RGC_P = 1.0 # p('on' centre) vs p('off' centre)
