#!/usr/bin/env python
"""
    Trying to reconstruct a ground-truth audio stimulus (TIMIT utterances) from
    an LNP model of cortical neurons.
"""

# Experiment/plotting/other parameters
from src.params.stimulus_params import *
from src.params.datafile_params import *
from src.params.timit_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Reading in the data
import glob
from scikits.audiolab import Sndfile, play

# Definitions of spectrograms, STRFs
from src.spectrogram import fourier_spectrogram, mel_spectrogram
from src.strf import STRF

mag_data = []
phase_data = []
window = np.hamming(TIMIT_L)

for f in glob.glob('data/timit/**/*.wav'):
    print 'Reading %s' % f
    sf = Sndfile(f, 'r')
    n = sf.nframes
    signal = sf.read_frames(n)
    mag_spectrogram = mel_spectrogram(signal,window,N_MEL,10.0,7500.0,8000.0,
                                      TIMIT_N_FFT, TIMIT_L, TIMIT_STEP)
    mag_data.append(mag_spectrogram)
    # phase_data.append(phase_spectrogram)

# Creating some STRFs
n_neurons = 10
# Spread preferred frequencies randomly.
pref_freq_idxs = np.random.randint(3, N_MEL - 3, n_neurons)
# Spread preferred delays randomly as well.
max_delay = 5
pref_delay_idxs = np.random.randint(0, max_delay - 2, n_neurons)
STRFs = np.zeros((N_MEL, max_delay, n_neurons))
for i in range(n_neurons):
    STRFs[:,:,i] = STRF(N_MEL, max_delay)

# LNP neurons considered here.
# Linear stage involves convolving the STRF with the stimulus.
lin_responses = []
base_response = 1.0
for s in mag_data:
    responses = []
    for i in range(n_neurons):
        r1 = np.array([np.convolve(STRFs[m,:,i], s[m,:], mode='same')
                       for m in range(N_MEL)])
        responses.append(base_response + np.sum(r1, axis=0))
    lin_responses.append(responses)

# Nonlinear stage
nlin_responses = []
def nonlinearity(r):
    return r ** 1000.0

for rs in lin_responses:
    nl_response = []
    for r in rs:
        nl_response.append(nonlinearity(r))
    nlin_responses.append(nl_response)

# Poisson spike train
# Here I create a series representing instantaneous spike rate
spike_rates = []
for rs in nlin_responses:
    spikes = []
    for r in rs:
        spks = []
        for rate in r:
            spks.append(np.random.poisson(rate))
        spikes.append(np.array(spks))
    spike_rates.append(spikes)

# Reconstruction begins here...
# The equations for these are taken from
# 'Influence of Context and Behavior on Stimulus Reconstruction from Neural
# Activity in Primary Auditory Cortex' - Mesgarani et al (2009)

# Reconstruction using optimal stimulus prior
max_lag_samples = int(MAX_ACORR_LAG * SAMPLING_RATE)
reconstructions_opt_mel = []
Gs_opt = []
for i, S in enumerate(mag_data):
    T = S.shape[1]
    R = np.zeros(((max_lag_samples + 1) * n_neurons, T))
    for k in range(n_neurons):
        r = spike_rates[i][k]
        r -= r.mean()
        for j in range(max_lag_samples + 1):
            R[k * (max_lag_samples + 1) + j, j:] = r[:T-j]
    C_RR = np.dot(R, R.T)
    C_RS = np.dot(R, S.T)
    G = np.dot(np.linalg.inv(C_RR), C_RS)
    Gs_opt.append(G)

# Reconstruct the spectrograms from the Gs_opt
for i, G in enumerate(Gs_opt):
    R = np.array(spike_rates[i])
    Rhat = R - R.mean()
    s_hat = np.zeros((N_MEL, R.shape[1]))
    for j in range(G.shape[1]):
        g = G[:,j]
        g_rect = g.reshape((n_neurons, max_lag_samples + 1))
        s1 = np.array([np.convolve(g_rect[k,:], Rhat[k,:], mode='same')
                       for k in range(n_neurons)])
        s_hat[j,:] = np.sum(s1, axis=0) + R.mean()
    reconstructions_opt_mel.append(s_hat)

# Flat stimulus prior (uses the STRF)
# First have to estimate the STRF from the responses.
reconstructions_flat_mel = []
Hs_opt = []
for i, s in enumerate(mag_data):
    T = s.shape[1]
    S = np.zeros(((max_lag_samples + 1) * N_MEL, T))
    for j in range(max_lag_samples + 1):
        S[j*N_MEL:(j+1)*N_MEL, j:] = s[:,:T-j]
    R = np.array(spike_rates[i])
    R -= R.mean(axis=1, keepdims=True)
    C_SS = np.dot(S, S.T)
    C_SR = np.dot(S, R.T)
    H = np.dot(np.linalg.inv(C_SS), C_SR)
    Hs_opt.append(H)

for i, H in enumerate(Hs_opt):
    # Reconstruct the stimulus from the Hs_opt
    F = np.dot(np.linalg.inv(np.dot(H, H.T)), H)
    R = np.array(spike_rates[i])
    R -= R.mean(axis=1, keepdims=True)
    S_hat = np.dot(F, R)
    T = S_hat.shape[1]
    s_hat = np.zeros((N_MEL, T))
    for j in range(max_lag_samples + 1):
        s_hat += np.pad(S_hat[j*N_MEL:(j+1)*N_MEL,j:], [(0,0),(0,j)],
                        mode='constant')
    s_hat /= max_lag_samples + 1
    s_hat += mag_data_means[i]
    reconstructions_flat_mel.append(s_hat)
