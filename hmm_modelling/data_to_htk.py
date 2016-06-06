#!/usr/bin/env python
import struct
import os

import numpy as np

f_dir = './video_features_txt/'
f_dir_h = './video_features_htk/'

labels = {'0' : 'add_teabag', '1' : 'cut_fruit',
          '2' : 'take_knife', '3' : 'take_cup',
          '4' : 'add_saltnpepper', '5' : 'pour_coffee',
          '6' : 'peel_fruit', '7' : 'pour_milk',
          '8' : 'crack_egg', '9' : 'fry_egg'}

for n_dir in os.listdir(f_dir):
    n_path = os.path.join(f_dir, n_dir)
    for c_dir in os.listdir(n_path):
        c_label = labels[c_dir]
        c_path = os.path.join(n_path, c_dir)
        for r_file in os.listdir(c_path):
            r_path = os.path.join(c_path, r_file)

            seq = np.loadtxt(r_path, delimiter=' ')
            print r_path, seq.shape

            # http://custom-made-square-wheel.blogspot.in
            # /2009/01/writing-htk-binary-feature-file.html
            n_samples, n_features = seq.shape
            flat_seq = seq.flatten()

            h = struct.pack(
                    '>iihh', # the beginning '>' says write big-endian
                    n_samples,
                    1000000, # sample_period
                    n_features * 4, # float takes 4 bytes
                    9) # user features
            assert len(h) == 12

            s = ''
            for i in range(n_samples * n_features):
                s += struct.pack('>f', flat_seq[i])

            # Write to output file
            out_dir = os.path.join(f_dir_h, n_dir, c_label)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, 
                            c_label + r_file.replace('txt', 'htk'))

            with open(out_path, 'wb') as outfile:
                outfile.write(h)
                outfile.write(s)
