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

# Experiments performed
## Computing population response properties
- Orientation tuning
- Hierarchical agglomerative clustering
- Reliability computation
    - Effect of clustering

## Decoding video identity on single-trial basis
- Template matching
- Comparison with other feature-extraction methods
- Comparison with random projections

## Reconstructing stimulus from recorded responses
- Linear regression-based reconstruction (reverse correlation)
    - Forward and backward models
    - Effect of regularisation
