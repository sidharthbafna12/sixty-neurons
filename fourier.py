#!/usr/bin/python
""" fourier.py
    Compute Discrete Fourier Transform spectra of the trial-averaged neural
    responses.
"""

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.fourier_params import *
from src.params.grating.stimulus_params import *

# Basics, plus for FFT
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Reading in the data
from src.response import Response

locs = [os.path.join(DATA_DIR, '%s_ori.npy' % c) for c in MICE_NAMES]
data = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES, locs))
dirs_rad = np.radians(DIRECTIONS)

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %s' % name
    
    N_S = m.data.shape[0]
    N = m.data.shape[1]
    if N_S == 8:
        stim = ORIENTATIONS
    elif N_S == 16:
        stim = DIRECTIONS

    # Fourier coefficients for the responses.
    m.fft_coeffs= np.fft.rfft(m.data,axis=2, n=FFT_WIDTH)
    m.avg_fft_coeffs = np.mean(m.fft_coeffs, axis=3)
    m.freqs = np.fft.rfftfreq(FFT_WIDTH, 1.0 / CA_SAMPLING_RATE)
    
    ############################################################################
    # Plotting
    if not os.path.isdir(os.path.join(PLOTS_DIR,'fourier','mag-%s' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'fourier', 'mag-%s' % name))
    if not os.path.isdir(os.path.join(PLOTS_DIR,'fourier','phase-%s' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'fourier', 'phase-%s' % name))

    for i in range(N):
        print 'Neuron %d' % i
        # Magnitude spectrum in response to different stimuli
        fig = plt.figure(figsize=(30,15))
        fig.suptitle('Fourier magnitude spectrum for mouse %s neuron %d'
                     % (name, i))
        for d, s in enumerate(stim):
            sp = plt.subplot2grid((2,4), (d/4, d%4))
            sp.set_title('%.1f degrees' % s)
            sp.set_yscale('log')
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.scatter(m.freqs, np.absolute(m.avg_fft_coeffs[d,i,:]))
        fig.savefig(os.path.join(PLOTS_DIR, 'fourier/mag-%s/%d.eps'
                                            % (name, i)),
                    bbox_inches='tight')
        plt.close()

        # Phase spectrum
        fig = plt.figure(figsize=(30,15))
        fig.suptitle('Fourier phase spectrum for mouse %s neuron %d'
                     % (name, i))
        for d, s in enumerate(stim):
            sp = plt.subplot2grid((2,4), (d/4, d%4))
            sp.set_title('%.1f degrees' % s)
            plt.xlabel('Frequency')
            plt.ylabel('Phase (in degrees)')
            plt.scatter(m.freqs, np.degrees(np.angle(m.avg_fft_coeffs[d,i,:])))
        fig.savefig(os.path.join(PLOTS_DIR, 'fourier/phase-%s/%d.eps'
                                            % (name, i)),
                    bbox_inches='tight')
        plt.close()
