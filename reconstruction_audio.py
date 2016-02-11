#!/usr/bin/env python

"""
    Trying to reconstruct a ground-truth audio stimulus (TIMIT utterances) from
    an LNP model of cortical neurons.
"""

# Experiment/plotting/other parameters
from src.params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Reading in the data
import glob
from scikits.audiolab import Sndfile, play
from src.response import Response

# Defining the Mel filterbank to use for the magnitude spectrogram
def mel(f):
    return 1125.0 * np.log(1 + f/700.0)

def melinv(m):
    return 700 * (np.exp(m/1125.0) - 1.0)

maxF = TIMIT_SAMPLERATE / 2
mel_filterbank = np.zeros((TIMIT_N_FFT / 2 + 1, N_MEL))
melFreqs = np.linspace(0, mel(maxF), num=N_MEL)
freqs = np.linspace(0, maxF, num = TIMIT_N_FFT/2 + 1)

for i, mf in enumerate(melFreqs):
    f0 = melinv(mf)
    if i > 0:
        fL = melinv(melFreqs[i-1])
    else:
        fL = f0
    if i < N_MEL - 1:
        fR = melinv(melFreqs[i+1])
    else:
        fR = f0

    height = 2 / (fR - fL)

    for j, f in enumerate(freqs):
        if f < fL:
            continue
        elif f >= fL and f < f0:
            mel_filterbank[j,i] = height * (f - fL) / (f0 - fL)
        elif f >= f0 and f < fR:
            mel_filterbank[j,i] = height * (fR - f) / (fR - f0)
        else:
            continue

mag_data = []
mag_data_means = []
phase_data = []
window = np.hamming(TIMIT_L)

for f in glob.glob('timit/**/*.wav'):
    print 'Reading %s' % f
    sf = Sndfile(f, 'r')
    n = sf.nframes
    signal = sf.read_frames(n)
    signal = np.pad(signal, (0, TIMIT_L - signal.shape[0] % TIMIT_L),
                    mode='constant')
    windows = np.array([signal[i:i+TIMIT_L] * window
                        for i in range(0, n - n % TIMIT_L + 1, TIMIT_STEP)])
    fourier_coeffs = np.fft.rfft(windows, n=TIMIT_N_FFT, axis=1)
    phase_spectrogram = np.angle(fourier_coeffs.T)
    # Not taking power (square) because it seems to wipe out the lower values
    mag_spectrogram = np.absolute(np.dot(fourier_coeffs, mel_filterbank)).T

    mag_means = mag_spectrogram.mean(axis=0, keepdims=True)
    mag_spectrogram -= mag_means
    
    mag_data_means.append(mag_means)
    mag_data.append(mag_spectrogram)
    phase_data.append(phase_spectrogram)

# Creating some STRFs
# Shape taken from
# 'Spectro-Temporal Response Field Characterization with Dynamic Ripples in
# Ferret Primary Auditory Cortex' - Depireux, Simon et al (2000)

# Spread preferred frequencies randomly.
n_neurons = 10
pref_freq_idxs = np.random.randint(3, N_MEL - 3, n_neurons)
# Spread preferred delays randomly as well.
max_delay = 5
pref_delay_idxs = np.random.randint(0, max_delay - 2, n_neurons)
# STRF matrices.
n_freqs = mag_data[0].shape[0]
STRFs = np.zeros((n_freqs, max_delay, n_neurons))
for i in range(n_neurons):
    f, d = pref_freq_idxs[i], pref_delay_idxs[i]

    # Centre-surround RF at given delay
    STRFs[f-1:f+2,d,i] = 1
    STRFs[f-2,d,i] = 0
    STRFs[f+2,d,i] = 0

    # Centre-surround RF at given frequency band
    if d > 0:
        STRFs[f-1:f+2,d-1,i] = -0.5
    if d < max_delay - 1:
        STRFs[f-1:f+2,d+1,i] = -1
        STRFs[f-2,d+1,i] = 0.5
        STRFs[f+2,d+1,i] = 0.5
        STRFs[f-3,d+1,i] = -0.5
        STRFs[f+3,d+1,i] = -0.5
    if d < max_delay - 2:
        STRFs[f-1:f+2,d+2,i] = 0.5
        STRFs[f-2,d+2,i] = -0.5
        STRFs[f+2,d+2,i] = -0.5
    if d < max_delay - 3:
        STRFs[f-1:f+2,d+3,i] = -0.5

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
