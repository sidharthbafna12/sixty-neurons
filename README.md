# sixty-neurons
Various signal processing algorithms applied on Ca recordings from mouse V1
neurons. The stimuli under consideration fall in two categories:
1. Drifting sinusoidal gratings of constant spatial frequency and 16 different
   drift directions
2. Natural scene movies from the van Hateren dataset (5 movies, 11 experiments)

The responses were collected by two-photon microscopy from mice injected with
GCaMP6f via a virus. The responses to the gratings were sampled at 20 Hz and the
ones to the natural movies at 50 Hz. The grating stimuli were repeated 10 times
for each direction (160 presentations in total) and the natural movies 5-10
times (variable across experiments). This leads to 19200 recorded frames for the
grating stimuli (including the periods when a gray screen was shown between
trials), and 6000-20000 frames for the natural movies (again including the
inter-trial gray screen).

# Data format
The original dataset was a set of .mat structs, which contained raw fluorescence
values for the cells studied, the sliding-window-normalised fluorescence values
for the same cells, and the spike rates inferred therefrom. The sliding window
computation was done for a window of length 400 frames, using the simple formula
            dFF = (F - F0) / F0
where F0 is the baseline fluorescence computed over this window. This normalised
fluorescence is then used to infer the spiking rate using the Vogelstein
deconvolution algorithm, which is also already available in the original dataset.

To make working with this data easier, the structs were read into NumPy arrays
of the given format:
    (S, N, L, R) -> (stimulus index, neuron index, sample index, trial index)
where the array now holds only the inferred spike rates and nothing else.

--------------------------------------------------------------------------------

# Experiments
## Computing population response properties
- Orientation tuning : tuning.py
- Hierarchical agglomerative clustering : clustering.py
- Reliability computation : reliability.py
    - Effect of clustering : currently hacked into reliability.py (TODO: fix)

## Decoding video identity on single-trial basis
- Using a nearest-neighbour approach : classify_single_trial.py

## Classifying new video classes using simulated V1 responses
- train_mlp_response_model.py : to fit a multilayer perceptron to V1 responses
  using Theano.
- compute_simulated_responses.py : running the MLP on unseen videos from the
  breakfast dataset.
- hmm/data_to_htk.py : converting the MLP output from .npy to .htk
- hmm/create_mlf.py : splitting the simulated responses into a train and a test
  set
- hmm/run_hmms.py : running an HMM-based classification model on the simulated
  responses, to verify the discrimination ability of the simulated V1 output.
- Comparison with random projections : compute_random_projections.py, replacing
  the computation in compute_simulated_responses.py. The rest of the pipeline
  propceeds the same way.

## Reconstructing stimulus from recorded responses
- Linear regression-based reconstruction (reverse correlation)
    - Forward and backward models : reconstruction.py
    - Effect of regularisation : TODO
