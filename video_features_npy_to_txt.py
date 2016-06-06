#!/usr/bin/env python
""" video_features_npy_to_txt.py
"""

import os
import numpy as np

features_dir = './output/video_features'
mean_rsp_dir = './output/mean_rsps'

mean_rsps = [np.load(os.path.join(mean_rsp_dir, '%d.npy' % p))
             for p in range(len(os.listdir(mean_rsp_dir)))]

for i in range(11):
    net_features_dir = os.path.join(features_dir, 'net%d' % i)
    claspaths = [os.path.join(net_features_dir, class_index)
                 for class_index in os.listdir(net_features_dir)]
    for c in claspaths:
        files = os.listdir(c)
        filepaths = [os.path.join(c, p) for p in files]

        seqs = map(lambda p : np.load(p) + mean_rsps[i], filepaths)
        outdir = c.replace('video_features', 'video_features_txt')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        print outdir
        outfilepaths = [os.path.join(outdir, p.replace('npy', 'txt'))
                        for p in files]

        for seq, out in zip(seqs, outfilepaths):
            np.savetxt(out, seq, fmt='%.4e', delimiter=' ')
