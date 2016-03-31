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
times (variable across experiments).

# Internal data format
(S, N, L, R) -> (stimulus index, neuron index, sample index, trial index)
All data files will be read into this internal array format.

# Computing population response properties
- Orientation tuning
- Hierarchical agglomerative clustering

# Reconstructing stimulus from recorded responses
- Template matching
- Spike-triggered averaging
