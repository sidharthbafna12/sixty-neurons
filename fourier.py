#!/usr/bin/python

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.fourier_params import *
from src.params.grating.stimulus_params import *

# Basics, plus for FFT
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Reading in the data
import scipy.io as sio
from src.response import Response

data = map(lambda L: Response(sio.loadmat(L, struct_as_record=False,
                                          squeeze_me=True)['Data']),
           DATA_LOCS)
dirs_rad = np.radians(DIRECTIONS)

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %c' % name

    # Fourier coefficients for the responses.
    m.fft_coeffs= np.fft.rfft(m.response_ori,axis=Response.TimeAxis,n=FFT_WIDTH)
    m.avg_fft_coeffs = np.mean(m.fft_coeffs, axis=Response.TrialAxis)
    m.freqs = np.fft.rfftfreq(FFT_WIDTH, 1.0 / SAMPLING_RATE)
    
    ############################################################################
    # Plotting
    if not os.path.isdir(os.path.join(PLOTS_DIR,'Fourier','Mag-%c' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'Fourier', 'Mag-%c' % name))
    if not os.path.isdir(os.path.join(PLOTS_DIR,'Fourier','Phase-%c' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'Fourier', 'Phase-%c' % name))

    for i in range(m.N):
        # Magnitude spectrum in response to different stimuli
        fig = plt.figure(figsize=(30,15))
        fig.suptitle('Fourier magnitude spectrum for mouse %c neuron %d'
                     % (name, i))
        for d, ori in enumerate(ORIENTATIONS):
            sp = plt.subplot2grid((2,4), (d/4, d%4))
            sp.set_title('%.1f degrees' % ori)
            sp.set_yscale('log')
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.scatter(m.freqs,
                        np.absolute(m.avg_fft_coeffs[d,:,i]))
        fig.savefig(os.path.join(PLOTS_DIR, 'Fourier/Mag-%c/%d.eps'
                                            % (name, i)),
                    bbox_inches='tight')
        plt.close()

        # Phase spectrum
        fig = plt.figure(figsize=(30,15))
        fig.suptitle('Fourier phase spectrum for mouse %c neuron %d'
                     % (name, i))
        for d, ori in enumerate(ORIENTATIONS):
            sp = plt.subplot2grid((2,4), (d/4, d%4))
            sp.set_title('%.1f degrees' % ori)
            plt.xlabel('Frequency')
            plt.ylabel('Phase (in degrees)')
            plt.scatter(m.freqs,
                        np.degrees(np.angle(m.avg_fft_coeffs[d,:,i])))
        fig.savefig(os.path.join(PLOTS_DIR, 'Fourier/Phase-%c/%d.eps'
                                            % (name, i)),
                    bbox_inches='tight')
        plt.close()
