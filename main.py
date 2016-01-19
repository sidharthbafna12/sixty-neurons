"""
IMPORTANT - Read and understand this document first before commencing any
analysis.

About the experiment:
    Virus expressing GCaMP6f was injected into the V1 of mice.
    Approximately 3 weeks post infection, mice were imaged under a 2-photon
    microscope while sinusoidal drifting gratings were presented on a computer
    screen placed 3 inches from the mouse (1 degree of visual space ~ 21.3 
    pixels on the screen).  

Aims: (1) to map orientation tuning responses of excitatory pyramidal neurons
          in V1.
      (2) to determine response reliability at preferred orientation.
      (3) to determine signal and noise correlations between neuron as a
          function of orientation tuning.

============ Notes about stimulus
    Sinusoidal drifting gratings at 16 different directions (0:22.5:337.5).
    Spatial frequency was fixed at 0.03 cycles per degree.
    
    Each direction was repeated 10 times (i. e. 10 trials per direction). 
    Directions are presented in a randomized order.
    
    Each direction was presented for 2s and was always preceded by a 4s gray
    screen. Therefore the total duration of the stimulus is 6s.
    
    Calcium signals (GCaMP6f in awake mice) were acquired from awake mice at
    20 frames per seconds. Thus, sampling rate is 20Hz.

============ Notes about Data.mat
Data.mat contains 4 entries:
    Data.rawF    = raw fluorescence values.
                    Matrix size = number of cells x number of frames.
    Data.dFF     = fluorescence normalized to baseline (dFF =  (F-F0)/F0,
                   where F0 is the baseline fluorescence computed using a 
                   sliding window of 400 frames). Same size as above.
    Data.Spks    = inferred spike rate using the Vogelstein deconvolution
                   algorithm. Same size as above.
    Data.StimSeq = contains sequence of directions presented during that
                   experiment. Vector size = 160 x 1.

============ Notes about Ori.mat
Ori.mat contains 20 entries. The most pertinent entries are:
    Ori.OSI = orientation selectivity indices of the neurons
    Ori.OrFit = double-wrapped Gaussian fits
    Ori.OrFitQuality  = goodness of Gaussian fits. (Higher the
                        percentage value, the better the fit)
    Ori.Width = tuning width in degrees
    Ori.PrefOri = preferred orientation
    Ori.SpkResponse = Contains neural responses for each cells sorted
                      according to the different directions.
                      Size: 1xNumber of cell Cell array. Each cell entry 
                      contains a 1x Number of Direction Cell array, which
                      contains a Number of Frames x Trials matrix.

============ Task
    Work only with the data contained in Data.mat.

    For each cell, use the information contained in StimSeq. To sort the dFF 
    into various directions. Verify this result using Ori.CaResponse. Do the
    same with Spks.

    For each cell, use the sorted information to plot orientation-tuning curves,
    that is, average response (could be dFF or spks, average over stimulus
    epoch and time) vs. direction.  Sharply tuned cells should have responses
    that peak at 1 orientation, while broadly tuned cells have responses to
    almost all orientations.

    Read paper A , code the equations in these papers corresponding to OSI and
    preferred orientation. For each cell, compute OSI and preferred
    orientation. OSI is a measure of how strongly a cell responds to an
    orientation, so sharply tuned cells should have an OSI close to 1. Verify
    these results using Ori.OSI.
    
    Perform whatever further analysis on the
    data you desire. (Suggestions: compute correlation between cells and plot as
    function of OSI or preferred oriention, compute reliability between
    trials...)
"""

# Experiment and plotting parameters
from src.params import *

# Numerical stuff
import numpy as np
from scipy.optimize import curve_fit
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist

# Response properties calculated
from src.double_gaussian_fit import fit_wrapped_double_gaussian,\
                                    wrapped_double_gaussian
from src.osi import selectivity_index, pref_direction

# Plotting
from matplotlib import pyplot as plt

# Plotting arguments
import sys

################################################################################
## Reading in the data.
import os
import scipy.io
from src.mouse import Mouse

# Mouse recorded data location.
data_locs = [os.path.join(DATA_DIR,
                          'Mouse-%c/Data-Mouse%c.mat' % (c, c))
                                                      for c in MICE_NAMES]
# Orientation tuning responses.
ori_locs = [os.path.join(DATA_DIR,
                         'Mouse-%c/Solutions/Ori.mat' % c)
                                                      for c in MICE_NAMES]

mice_data = map(lambda loc : Mouse(scipy.io.loadmat(loc,
                                                    struct_as_record=False,
                                                    squeeze_me=True)['Data']),
                data_locs)

ori_data = map(lambda loc : scipy.io.loadmat(loc,
                                             struct_as_record=False,
                                             squeeze_me=True)['Ori'],
               ori_locs)

directions = np.radians(DIRECTIONS)
directions_finer = np.linspace(np.min(directions), np.max(directions),
                               num=200)

sigma0 = 2 * np.pi / len(DIRECTIONS) # initial tuning curve width

for index, m in enumerate(mice_data):
    name = MICE_NAMES[index]
    print "Mouse %c" % name

    # Get average response to different directions.
    # Average over trials and time.
    m.avg_dir_response = np.mean(m.dir_response, axis=(Mouse.TrialAxis,
                                                       Mouse.TimeAxis))
    
    # Find response correlation between neurons.
    m.ori_response_corr = \
            np.array([np.corrcoef(np.mean(m.ori_response,
                                          axis=Mouse.TrialAxis)[i,:,:].T)
                      for i in range(len(ORIENTATIONS))])

    
    # Find orientation-tuning curves
    m.dg_fit_params = [None for i in range(m.N)]
    m.dg_fit_r2 = [None for i in range(m.N)]
    init_thetas = [directions[np.argmax(m.avg_dir_response[:,i])] % np.pi
                   for i in range(m.N)]
    init_cs = [m.avg_dir_response[:,i].min() for i in range(m.N)]
    init_ws = [m.avg_dir_response[:,i].max() - m.avg_dir_response[:,i].min()
               for i in range(m.N)]
    init_sigmas = [sigma0 for i in range(m.N)]
    init_params = zip(init_thetas, init_sigmas, init_cs, init_ws)
    m.dg_fit_params, m.dg_fit_r2 = \
            zip(*[fit_wrapped_double_gaussian(directions,
                                              m.avg_dir_response[:,i],
                                              p0 = init_params[i])
                  for i in range(m.N)])

    # Find orientation sensitivity.
    m.osi = selectivity_index(m.ori_response, orientation_flag=True)
    m.pref_orientation = pref_direction(m.ori_response, True)

    # See if the response correlation is more among orientation-sensitive
    # neurons.
    m.osi_sort_idxs = np.argsort(m.osi)
    m.sorted_ori_response = m.ori_response[:,:,:,m.osi_sort_idxs]
    m.sorted_ori_response_corr = \
            np.array([np.corrcoef(np.mean(m.sorted_ori_response,
                                          axis=Mouse.TrialAxis)[i,:,:].T)
                      for i in range(len(ORIENTATIONS))])

    # Also if it is more for neurons with the same preferred orientation.
    m.pref_ori_sort_idxs = np.argsort(m.pref_orientation)
    m.sorted_pref_ori_response = m.ori_response[:,:,:,m.pref_ori_sort_idxs]
    m.sorted_pref_ori_response_corr = \
            np.array([np.corrcoef(np.mean(m.sorted_pref_ori_response,
                                          axis=Mouse.TrialAxis)[i,:,:].T)
                      for i in range(len(ORIENTATIONS))])

    # Might as well do this for the R^2 values.
    # m.r2_sort_idxs = np.argsort(m.dg_fit_r2)

    m.dsi = selectivity_index(m.dir_response, orientation_flag=False)
    m.pref_direction = pref_direction(m.dir_response, orientation_flag=False)
    
    # Fourier coefficients for the responses. Maybe there's something there?
    m.fft_coeffs = np.fft.rfft(m.dir_response, axis=Mouse.TimeAxis, n=FFT_WIDTH)
    m.avg_fft_coeffs = np.mean(m.fft_coeffs, axis=Mouse.TrialAxis)
    m.freqs = np.fft.rfftfreq(FFT_WIDTH, 1.0 / SAMPLING_RATE)

    # Hierarchical agglomerative clustering.
    avg_response_timeseries = np.mean(m.ori_response, axis=(Mouse.TrialAxis))
    dists = [(pdist(i.T, 'correlation') / 2.0).clip(0.0, 1.0)
             for i in avg_response_timeseries]
    linkages = [hac.linkage(dist, method='single', metric='correlation')
                for dist in dists]
    m.hac_idxs = [hac.dendrogram(Z)['leaves'] for Z in linkages]
    m.hac_ori_response = \
            np.array([m.ori_response[i][:,:,idxs]
                      for i, idxs in enumerate(m.hac_idxs)])
    m.hac_ori_response_corr = \
            np.array([np.corrcoef(np.mean(m.hac_ori_response,
                                          axis=Mouse.TrialAxis)[i,:,:].T)
                      for i in range(len(ORIENTATIONS))])

    if PLOTTING_AVERAGE_RESPONSE:
        # Save the average response matrix as an image.
        rows = 1
        cols = 2
        fig = plt.figure(figsize=(cols * 7, rows * 4))
        normalised_avg_rsp = m.avg_dir_response / np.sum(m.avg_dir_response,
                                                         axis=0)

        sp = fig.add_subplot(rows, cols, 1)
        sp.set_title('Normalised Average response')
        plt.imshow(normalised_avg_rsp, cmap='gray', interpolation='nearest')
        plt.xlabel('Neuron index')
        plt.ylabel('Direction (divided by 22.5 degrees)')

        # Also see whether OSI induces any clustering...
        sp = fig.add_subplot(rows, cols, 2)
        sp.set_title('Normalised Average response (neurons sorted by OSI)')
        plt.imshow(normalised_avg_rsp[:,m.osi_sort_idxs], cmap='gray',
                   interpolation='nearest')
        plt.xlabel('Neuron index')
        plt.ylabel('Direction (divided by 22.5 degrees)')

        # And add R^2 for good measure
        # sp = fig.add_subplot(rows, cols, 3)
        # sp.set_title('Normalised Average response (neurons sorted by R^2)')
        # plt.imshow(normalised_avg_rsp[:,m.r2_sort_idxs], cmap='gray',
        #            interpolation='nearest')
        # plt.xlabel('Neuron index')
        # plt.ylabel('Direction (divided by 22.5 degrees)')

        fig.savefig(os.path.join(PLOTS_DIR, 'OrientationTuning/AvgRsp-%c.eps'
                                            % (name,)),
                    bbox_inches='tight')
        plt.close("all")
        
    if PLOTTING_DIRWISE:
        for d, dirn in enumerate(ORIENTATIONS):
            print '%.1f degrees' % dirn

            rows = 2
            cols = 3

            fig = plt.figure(figsize=(cols * 5,rows * 5))
            fig.suptitle('Mouse %c - %.1f degrees' %(name,dirn))
                    
            # OSI values when sorting neurons by that value.
            sp = fig.add_subplot(rows, cols, 1)
            sp.set_title('OSI')
            plt.plot(m.osi[m.osi_sort_idxs])

            # Preferred orientations for these neurons. They should overlap, I
            # suppose?
            sp = fig.add_subplot(rows, cols, 2)
            sp.set_title('Preferred orientation')
            plt.plot(m.pref_orientation[m.osi_sort_idxs])
            plt.grid(True)
            plt.xlim(0, m.N)
                
            # Response correlation in original data.
            sp = fig.add_subplot(rows, cols, 3)
            sp.set_title('Response correlation')
            plt.imshow(m.ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')

            # Response correlations in neurons grouped by preferred orientation.
            sp = fig.add_subplot(rows, cols, 4)
            sp.set_title('Response correlation in neurons \n'
                         'sorted by preferred orientation')
            plt.imshow(m.sorted_pref_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')

            # Response correlations in neurons grouped/sorted by
            # orientation-sensitivity.
            sp = fig.add_subplot(rows, cols, 5)
            sp.set_title('Response correlation in OSI-sorted neurons')
            plt.imshow(m.sorted_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')
            
            # Response correlations in neurons grouped by HAC.
            sp = fig.add_subplot(rows, cols, 6)
            sp.set_title('Response correlation in HAC neurons')
            plt.imshow(m.hac_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')

            fig.savefig(os.path.join(PLOTS_DIR, 'Correlations/%c_%d.eps'
                                                % (name, d)),
                        bbox_inches='tight')
            plt.close()

            # Also save the correlation plots individually.
            # Response correlation in original data.
            fig = plt.figure()
            plt.title('Response correlation')
            plt.imshow(m.ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'Correlations/Original/%c_%d.eps'
                                     % (name, d)),
                        bbox_inches='tight')
            plt.close()

            # Response correlations in neurons grouped by preferred orientation.
            fig = plt.figure()
            plt.title('Response correlation in neurons sorted by '
                      'preferred orientation')
            plt.imshow(m.sorted_pref_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'Correlations/PrefOri_Sorted/%c_%d.eps'
                                     % (name, d)),
                        bbox_inches='tight')
            plt.close()

            # Response correlations in neurons grouped by OSI.
            fig = plt.figure()
            plt.title('Response correlation in OSI-sorted neurons')
            plt.imshow(m.sorted_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'Correlations/OSI_Sorted/%c_%d.eps'
                                     % (name, d)),
                        bbox_inches='tight')
            plt.close()

            # Response correlations in neurons grouped by HAC.
            fig = plt.figure()
            plt.title('Response correlation in HAC neurons')
            plt.imshow(m.hac_ori_response_corr[d],
                       vmin=-1, vmax=1,
                       cmap='bwr',
                       interpolation='nearest')
            plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
            plt.axis('off')
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'Correlations/HAC/%c_%d.eps'
                                     % (name, d)),
                        bbox_inches='tight')
            plt.close()

    if PLOTTING_CELLWISE:
        for i in range(m.N):
            print 'neuron %d' % i
            # Store the orientation tuning curve along with the fit.
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 8)
            fig.suptitle('Mouse %c Neuron %d orientation-tuning curve'
                         ' (R-squared %.2f)'
                         % (name, i, m.dg_fit_r2[i]))
            plt.xlabel('Stimulus')
            plt.ylabel('Average response')
            plt.scatter(DIRECTIONS, m.avg_dir_response[:,i],
                        label='recorded')
            plt.plot(np.degrees(directions_finer),
                     wrapped_double_gaussian(directions_finer,
                                             *m.dg_fit_params[i]),
                     label='double gaussian fit')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.savefig(os.path.join(PLOTS_DIR, 'OrientationTuning/OT-%c_%d.eps'
                                                % (name,i)),
                        bbox_inches='tight')
            plt.close()

            # Polar plots showing average response to each orientation.
            fig = plt.figure()
            ax = plt.subplot(111, polar=True)
            ax.plot(directions,
                    m.avg_dir_response[:,i] / np.sum(m.avg_dir_response[:,i]),
                    color='r', linewidth=3)
            ax.set_rmax(0.5)
            ax.grid(True)
            ax.set_title('Mouse %c Neuron %d orientation responses' \
                         % (MICE_NAMES[index], i))

            fig.savefig(os.path.join(PLOTS_DIR, 'OSI/CirVecs-%c_%d.eps'
                                                % (name,i)),
                        bbox_inches='tight')
            plt.close()

            # FFT coefficients (magnitude) of the average response.
            fig = plt.figure(figsize=(25,25))
            fig.suptitle('Fourier magnitude spectrum for mouse %c neuron %d'
                         % (MICE_NAMES[index], i))
            for d, dirn in enumerate(DIRECTIONS):
                sp = plt.subplot2grid((4,4), (d/4, d%4))
                sp.set_title('%.1f degrees' % dirn)
                sp.set_yscale('log')
                plt.xlabel('Frequency')
                plt.ylabel('Magnitude')
                plt.scatter(m.freqs,
                            np.absolute(m.avg_fft_coeffs[d,:,i]))
            fig.savefig(os.path.join(PLOTS_DIR, 'DFT/Magnitude-%c_%d.eps'
                                                % (name, i)),
                        bbox_inches='tight')
            plt.close()

            # Why ignore the phase then?
            fig = plt.figure(figsize=(25,25))
            fig.suptitle('Fourier phase spectrum for mouse %c neuron %d'
                         % (MICE_NAMES[index], i))
            for d, dirn in enumerate(DIRECTIONS):
                sp = plt.subplot2grid((4,4), (d/4, d%4))
                sp.set_title('%.1f degrees' % dirn)
                plt.xlabel('Frequency')
                plt.ylabel('Phase (in degrees)')
                plt.scatter(m.freqs,
                            np.degrees(np.angle(m.avg_fft_coeffs[d,:,i])))
            fig.savefig(os.path.join(PLOTS_DIR, 'DFT/Phase-%c_%d.eps'
                                                % (name, i)),
                        bbox_inches='tight')
            plt.close()

            # Maybe it will be more useful to have all responses to a particular
            # stimulus in one place.
            for d, dirn in enumerate(DIRECTIONS):
                rows = 2
                cols = 2

                fig = plt.figure(figsize=(cols * 5,rows * 5))
                fig.suptitle('Mouse %c Neuron %d - %.1f degrees' %(name,i,dirn))
                    
                # Plot responses for this particular direction.
                sp = fig.add_subplot(rows, cols, 1)
                sp.set_title('Responses')
                sp.set_xlabel('t')
                sp.set_ylabel('Response')
                plt.plot(m.dir_response[d,:,:,i].T, color='gray')
                # Might as well add the average response...
                plt.plot(np.mean(m.dir_response[d,:,:,i], axis=0),
                         color='red',
                         linewidth=5.0)

                # DFT magnitude spectrum
                sp = fig.add_subplot(rows, cols, 2)
                sp.set_title('Magnitude spectrum')
                sp.set_xlabel('Frequency')
                sp.set_ylabel('Magnitude')
                sp.set_yscale('log')
                plt.plot(m.freqs,
                         np.absolute(m.avg_fft_coeffs[d,:,i]))

                # Phase spectrum now.
                sp = fig.add_subplot(rows, cols, 4)
                sp.set_title('Phase spectrum')
                sp.set_xlabel('Frequency')
                sp.set_ylabel('Phase (in degrees)')
                plt.plot(m.freqs,
                         np.degrees(np.angle(m.avg_fft_coeffs[d,:,i])))

                fig.savefig(os.path.join(PLOTS_DIR, 'DirWise/%c/%d_%d.eps'
                                                    % (name, i, d)),
                            bbox_inches='tight')
                plt.close()
