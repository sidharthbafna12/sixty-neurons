#!/usr/bin/env python
""" generate_reconstructed_movies.py
    Generates movies out of image frames dumped by linear_reconstruction.py
"""

import os
import time

exp_type = 'natural'
if exp_type == 'grating':
    from src.params.grating.datafile_params import PLOTS_DIR
    from src.params.grating.stimulus_params import CA_SAMPLING_RATE
elif exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import PLOTS_DIR
    from src.params.naturalmovies.stimulus_params import CA_SAMPLING_RATE


for rtype in sorted(os.listdir(os.path.join(PLOTS_DIR,
                                            'linear-reconstruction'))):
    rtype_dir = os.path.join(PLOTS_DIR, 'linear-reconstruction', rtype)
    for mname in sorted(os.listdir(rtype_dir)):
        mdir = os.path.join(rtype_dir, mname)
        for trial_num in sorted(os.listdir(mdir)):
            frames_dir = os.path.join(mdir, trial_num)
            first = int(sorted(os.listdir(frames_dir))[0].replace('.png', ''))
            command = 'ffmpeg -framerate %d -start_number %d '\
                      '-i %s/%%d.png %s/video.mp4'\
                      % (CA_SAMPLING_RATE, first, frames_dir, frames_dir)
            print command
            os.system(command)
            time.sleep(0.5)
