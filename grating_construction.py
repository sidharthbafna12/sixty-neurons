#!/usr/bin/python

# Trying a forward construction of the neural responses.
# How exactly to do this, will hopefully emerge as I proceed to do this.

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *
from src.params.grating.pathway_params import *
screen = np.zeros(N_PX)

# Reading in the data
import scipy.io as sio
from src.response import Response
data = map(lambda L: Response(sio.loadmat(L, struct_as_record=False,
                                          squeeze_me=True)['Data']),
           DATA_LOCS)

# For stimulus movie
from src.grating import grating_movie
import scipy.misc

# Retinal ganglion cell filters
from src.retina_filter import rgc_filter
# Thalamus filters
from src.thal_filter import thalamus_filter

################################################################################
for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %c' % name
    
    L_stim = GRATING_DURATION * SAMPLING_RATE
    stim_seq = m.struct.StimSeq
    responses = m.slices

    print 'Creating RGC filters...'
    # transpose for (x,y)->(i,j)
    m.rgc_rf_types = np.random.choice(['on', 'off'],
                                      size=RGC_N_CELLS,
                                      p=[RGC_P, 1 - RGC_P]).T
    m.rgc_filter = rgc_filter(m.rgc_rf_types)
    
    for i, (dirn, tot_rsp) in enumerate(zip(stim_seq, responses)):
        rsp = tot_rsp[:GRATING_DURATION * SAMPLING_RATE]
        print 'Movie %d' % i
        grating = grating_movie(dirn, L_stim)

        print 'Computing RGC responses...'
        m.rgc_rsp = m.rgc_filter(grating)
        # print 'Computing thalamic responses...'
        # m.thal_rsp = thalamus_filter(m.rgc_rsp)

        if not os.path.isdir(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                          'Movies-%c'%name,'Movie-%d'%i)):
            os.makedirs(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                     'Movies-%c' % name, 'Movie-%d' % i))
        
        for j in range(L_stim):
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
                                         'diff_%d.png' % j))a
            scipy.misc.toimage(m.rgc_rsp[j,:,:])\
                      .save(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'rgc_%d.png' % j))
            scipy.misc.toimage(m.thal_rsp[j,:,:])\
                      .save(os.path.join(PLOTS_DIR, 'GratingStimuli',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'thal_%d.png' % j))
