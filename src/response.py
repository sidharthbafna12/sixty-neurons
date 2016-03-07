import numpy as np

from params.grating.stimulus_params import *

class Response:
    """ Data matrix convention for multiple trials:
        (S,T,L,N)-sized matrix implies:
            - S stimuli
            - T trials
            - L samples per trial
            - N neurons
        IMPORTANT : The values for these axes are NOT interchangeable!
                    Bad things will happen if these values are not used as is.
                    This is just to make the code more self-explanatory in some
                    places.
    """
    DirAxis=0
    TrialAxis=1
    TimeAxis=2
    CellsAxis=3

    def __init__(self, struct):
        self.struct = struct
        self.response = struct.Spks
        self.N = self.response.shape[0] # number of neurons

        # Something hacky:
        # The stimuli are preceded by a gray screen. The responses recorded in
        # that time should be the decaying responses to the preceding stimulus,
        # of course. So perhaps it will be useful later to organise the slices
        # that way, that is, to have the succeeding 4s gray screen be part of
        # the slice rather than the preceding one.
        # To do this, maybe I can just cyclic shift the responses for now, so
        # that they align properly...
        # The response to the stimulus is in the first 40 elements now though.
        self.response = np.roll(self.response,-SAMPLING_RATE * GRAY_SCREEN_TIME)
        self.slices = np.split(self.response, NUM_STIMULUS_PRESENTATIONS,
                               axis=1) # N-by-STIM_LEN slices.

        # GRAY_SCREEN_TIME * SAMPLING_RATE samples neglected.
        # Gives a (10, 40, N)-shaped array for each stimulus.
        # For 16 stimuli
        # So (16, 10, 40, N)-shaped array.
        self.response_dir \
                = np.array([[sl[:,:GRATING_DURATION * SAMPLING_RATE].T
                             for (index, sl) in enumerate(self.slices)
                             if struct.StimSeq[index] == stimulus]
                            for stimulus in sorted(DIRECTIONS)])

        # The response during the succeeding gray screen stimulus
        self.bg_dir \
                = np.array([[sl[:,GRATING_DURATION * SAMPLING_RATE:].T
                             for (index, sl) in enumerate(self.slices)
                             if struct.StimSeq[index] == stimulus]
                            for stimulus in sorted(DIRECTIONS)])
        
        # Response to orientation means average opposite directions.
        L = len(DIRECTIONS)
        self.response_ori \
                = np.array([0.5 * (self.response_dir[i]
                                 + self.response_dir[i + L/2])
                            for i in range(len(ORIENTATIONS))])
        self.bg_ori \
                = np.array([0.5 * (self.bg_dir[i]
                                 + self.bg_dir[i + L/2])
                            for i in range(len(ORIENTATIONS))])

    # Stimulus-wise response
    def sw_response(self, average_opposite_directions=True):
        if not average_opposite_directions:
            return self.response_dir
        else:
            return self.response_ori
