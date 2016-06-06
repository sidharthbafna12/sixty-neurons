#!/usr/bin/env python

import shutil, os
import cv2
import numpy as np

from scipy.signal import resample, decimate

basedir = './data/original-data/Breakfast_Final/vid/'
labeldir = './data/original-data/Breakfast_Final/lab_raw/'
outdir = './data/breakfast_sorted'
spatial_downsample_factor = 4

selected_actions = ['pour_milk', 'pour_coffee',
                    'take_knife', 'take_cup',
                    'crack_egg', 'fry_egg',
                    'peel_fruit', 'cut_fruit',
                    'add_saltnpepper', 'add_teabag']
selected_cams = ['cam01']

for dir_name in os.listdir(basedir):
    if not os.path.isdir(os.path.join(basedir, dir_name)):
        continue

    cam_dirs = os.listdir(os.path.join(basedir, dir_name))

    for cd in cam_dirs:
        if cd not in selected_cams:
            continue

        files = os.listdir(os.path.join(basedir, dir_name, cd))
        files = sorted([f for f in files if f.endswith('.avi')])
        for i, name in enumerate(files):
            filepath = os.path.join(basedir, dir_name, cd, name)
            labelpath = os.path.join(labeldir, dir_name,
                                     name.replace('.avi', '.coarse'))
            print '%s %s' % (cd, name)

            cap = cv2.VideoCapture(filepath)

            # Find frame numbers.
            frame_nums = []
            with open(labelpath) as lf:
                text = lf.read()
                lines = text.rstrip().split(' \n')[1:-1]
                for line in lines:
                    try:
                        print '\t', line
                        frame_range, action = line.split(' ')

                        if action not in selected_actions:
                            continue

                        start, end = map(int, frame_range.split('-'))
                        frames = [cv2.cvtColor(cap.read(i)[1],
                                               cv2.COLOR_RGB2GRAY)
                                  for i in range(start, end)]

                        clip_outdir = os.path.join(outdir, cd, action,
                                            'D%d'%spatial_downsample_factor)
                        if not os.path.isdir(clip_outdir):
                            os.makedirs(clip_outdir)
                        clip_outpath = os.path.join(clip_outdir,
                                                    name.replace('.avi',
                                                                 '_%d_%d'\
                                                                % (start, end)))
                        clip = np.dstack(frames)
                        
                        # Take the middle 256x256 from 320x240 input
                        clip = np.pad(clip, ((8,8),(0,0),(0,0)),mode='constant')
                        clip = clip[:,32:-32,:]

                        # Rescale to [-1,1]
                        clip = (clip - 128.0) / 128.0

                        # Downsample frames to a manageable size
                        clip = decimate(decimate(clip,spatial_downsample_factor,
                                                 axis=0),
                                        spatial_downsample_factor, axis=1)

                        # Resample to 10 Hz
                        n_samples_in = clip.shape[2]
                        in_sampling_rate = 15
                        out_sampling_rate = 10
                        n_samples_out = int((n_samples_in * out_sampling_rate)\
                                            / in_sampling_rate)
                        clip_out = resample(clip, n_samples_out, axis=2)

                        np.save(clip_outpath, clip_out)
                    except Exception, e:
                        print e
                        pass
