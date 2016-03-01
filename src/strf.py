import numpy as np

# Shape taken from
# 'Spectro-Temporal Response Field Characterization with Dynamic Ripples in
# Ferret Primary Auditory Cortex' - Depireux, Simon et al (2000)
def STRF(n_channels, max_delay):
    RF = np.zeros((n_channels, max_delay))

    # Preferred frequency, delay
    f = np.random.randint(3, n_channels - 3)
    d = np.random.randint(0, max_delay - 2)

    # Centre-surround RF at given delay
    RF[f-1:f+2,d] = 1
    RF[f-2,d] = 0
    RF[f+2,d] = 0

    # Centre-surround RF at given frequency band
    if d > 0:
        RF[f-1:f+2,d-1] = -0.5
    if d < max_delay - 1:
        RF[f-1:f+2,d+1] = -1
        RF[f-2,d+1] = 0.5
        RF[f+2,d+1] = 0.5
        RF[f-3,d+1] = -0.5
        RF[f+3,d+1] = -0.5
    if d < max_delay - 2:
        RF[f-1:f+2,d+2] = 0.5
        RF[f-2,d+2] = -0.5
        RF[f+2,d+2] = -0.5
    if d < max_delay - 3:
        RF[f-1:f+2,d+3] = -0.5
    
    return RF
