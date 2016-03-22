#!/usr/bin/env python

"""
    Attempting to reconstruct the original stimulus from the responses recorded.
"""

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.datafile_params import *
from src.params.stimulus_params import *
screen = np.zeros(N_PX)

# Reading in the data
import scipy.io as sio
from src.grating_response import GratingResponse
data = map(lambda L: GratingResponse(sio.loadmat(L, struct_as_record=False,
                                                 squeeze_me=True)['Data']),
           GRATING_DATA_LOCS)

# For stimulus movie
from src.grating import grating_movie
import scipy.misc

"""
from matplotlib import animation as anim
avconv_writer = anim.writers['avconv']
metadata = dict(title='Grating stimuli', artist='sg', comment='grating')
writer = avconv_writer(fps=SAMPLING_RATE, metadata=metadata)
"""

################################################################################
for index, m in enumerate(data):
    name = GRATING_MICE_NAMES[index]
    print 'Mouse %c' % name

    stim_seq = m.struct.StimSeq
    responses = m.slices

    m.ret_rf_centres = np.random.multivariate_normal(N_PX / 2,
                                                     np.diag(N_PX / 3),
                                                     RET_RF_N_CELLS)\
                         .astype(int)

    for i, (dirn, rsp) in enumerate(zip(stim_seq, responses)):
        print 'Movie %d' % i
        grating = grating_movie(dirn, GRATING_DURATION * SAMPLING_RATE)
        grating_diff = np.vstack((grating[0:1,:,:], np.diff(grating, axis=0)))
        # fig = plt.figure()
        
        if not os.path.isdir(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                          'Movies-%c'%name,'Movie-%d'%i)):
            os.makedirs(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                     'Movies-%c' % name, 'Movie-%d' % i))

        for j in range(GRATING_DURATION * SAMPLING_RATE):
            print '\tDumping frame %d' % j
            scipy.misc.toimage(grating[j,:,:], cmin=-1.0, cmax=1.0)\
                      .save(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         '%d.png' % j))
            scipy.misc.toimage(grating_diff[j,:,:], cmin=-1.0, cmax=1.0)\
                      .save(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'diff_%d.png' % j))

        """
        with writer.saving(fig, os.path.join(PLOTS_DIR, 'GratingStimuli',
                                             'Movies-%c' % name,
                                             'Movie-%d.mp4' % i), 100):
            # Save grating as video.
            for j in range(GRATING_DURATION * SAMPLING_RATE):
                print '\tDumping frame %d' % j
                plt.imshow(grating[:,:,j], cmap='gray', vmin=-1, vmax=1)
                writer.grab_frame()
        plt.close('all')
        """
