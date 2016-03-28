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
from src.grating_response import GratingResponse
data = map(lambda L: GratingResponse(sio.loadmat(L, struct_as_record=False,
                                                 squeeze_me=True)['Data']),
           DATA_LOCS)

# For stimulus movie
from src.grating import grating_movie
import scipy.misc

# Retinal ganglion cell filters
from src.retina_filter import rgc_filter2
# Thalamus filters
from src.thal_filter import thalamus_filter

# Direction/Orientation tuning
from src.osi import selectivity_index, pref_direction

# Gabor filters for V1
from skimage.filters import gabor_kernel

################################################################################
for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %c' % name

    m.rgc_rsps = []
    m.thal_rsps = []
    m.v1_rsps = []

    m.dsi = selectivity_index(m.response_dir, orientation_flag=False)
    m.pref_direction = pref_direction(m.response_dir, orientation_flag=False)

    # Fake Gabor filter RFs.
    gabors = [np.real(gabor_kernel(0.05, theta=np.radians(m.pref_direction[i]),
                                   sigma_x=20.0, sigma_y=20.0))
              for i in range(m.N)]
    temporal = np.arange(0, 4) * np.exp(-2 * np.arange(0, 4))

    
    L = GRATING_DURATION * CA_SAMPLING_RATE
    
    for i, (dirn, slic) in enumerate(zip(m.struct.StimSeq, m.slices)):
        print 'Movie %d' % i
        rsp = slic[:L]
        grating = grating_movie(dirn, L)
        rgc_rsp = rgc_filter2(grating)
        thal_rsp = thalamus_filter(rgc_rsp)

        # Fake V1 response.
        LY,LX = thal_rsp.shape[:2]

        windows = [thal_rsp[LY/2 - g.shape[0]/2 : LY/2 + g.shape[0]/2 + 1,
                            LX/2 - g.shape[1]/2 : LX/2 + g.shape[1]/2 + 1,:]
                   for g in gabors]
        spatial_rsps = [[np.sum(windows[i_n][:,:,t] * gabors[i_n])
                         for t in range(L)]
                        for i_n in range(m.N)]
        v1_rsps = np.array([np.convolve(temporal, sp_rsp)
                            for sp_rsp in spatial_rsps])

        m.rgc_rsps.append(rgc_rsp)
        m.thal_rsps.append(thal_rsp)
        m.v1_rsps.append(v1_rsps)

        if not os.path.isdir(os.path.join(PLOTS_DIR, 'Construction',
                                          'Movies-%c'%name,'Movie-%d'%i)):
            os.makedirs(os.path.join(PLOTS_DIR, 'Construction',
                                     'Movies-%c' % name, 'Movie-%d' % i,
                                     'Original'))
            os.makedirs(os.path.join(PLOTS_DIR, 'Construction',
                                     'Movies-%c' % name, 'Movie-%d' % i,
                                     'RGC'))
            os.makedirs(os.path.join(PLOTS_DIR, 'Construction',
                                     'Movies-%c' % name, 'Movie-%d' % i,
                                     'Saliency'))
            os.makedirs(os.path.join(PLOTS_DIR, 'Construction',
                                     'Movies-%c' % name, 'Movie-%d' % i,
                                     'GaborOutput'))
        for j in range(L):
            print '\tDumping frame %d' % j
            scipy.misc.toimage(grating[:,:,j], cmin=-1.0, cmax=1.0)\
                      .save(os.path.join(PLOTS_DIR, 'Construction',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'Original',
                                         '%d.png' % j))
            scipy.misc.toimage(rgc_rsp[:,:,j])\
                      .save(os.path.join(PLOTS_DIR, 'Construction',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'RGC',
                                         'rgc_%d.png' % j))
            scipy.misc.toimage(thal_rsp[:,:,j])\
                      .save(os.path.join(PLOTS_DIR, 'Construction',
                                         'Movies-%c' % name,
                                         'Movie-%d' % i,
                                         'Saliency',
                                         'thal_%d.png' % j))
        for k in range(m.N):
            fig = plt.figure()
            plt.plot(v1_rsps[k]);
            fig.savefig(os.path.join(PLOTS_DIR, 'Construction',
                                     'Movies-%c' % name,
                                     'Movie-%d' % i,
                                     'GaborOutput',
                                     'v1_rsp_%d.eps' % k),
                        bbox_inches='tight')
            plt.close()


"""
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
    m.rgc_filter = rgc_filter(m.rgc_rf_types, RGC_CENTRE_WIDTH, RGC_SURR_WIDTH,
                              RGC_CELL_SPACING, PIXELS_PER_DEGREE, N_PX)
    
    for i, (dirn, tot_rsp) in enumerate(zip(stim_seq, responses)):
        rsp = tot_rsp[:GRATING_DURATION * SAMPLING_RATE]
        print 'Movie %d' % i
        grating = grating_movie(dirn, L_stim)

        print 'Computing RGC responses...'
        m.rgc_rsp = m.rgc_filter(grating)
        print 'Computing thalamic responses...'
        m.thal_rsp = thalamus_filter(m.rgc_rsp)
"""
