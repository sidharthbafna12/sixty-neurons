#!/usr/bin/env python

import os
import struct
import numpy as np
import random

import shutil
f_dir = './video_features_htk/'
tr_scp_dir = './tr_scp'
te_scp_dir = './te_scp'
proto_dir = './protos'
mlf_out_dir = './mlf_out/'
n_states = 8
train_fraction = 0.7

for i in range(11):
    for j in range(4):
        if os.path.exists('./hmm%d_%d' % (i,j)):
            shutil.rmtree('./hmm%d_%d' % (i,j))
if os.path.exists('./train.mlf'):
    os.remove('./train.mlf')
if os.path.exists('./test.mlf'):
    os.remove('./test.mlf')
for d in [tr_scp_dir, te_scp_dir, proto_dir, mlf_out_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

for d in [tr_scp_dir, te_scp_dir, proto_dir]:
    if not os.path.isdir(d):
        os.makedirs(d)

train_test_split_done = False

n_features = None

tr_files = []
te_files = []
tr_mlf_lines = ['#!MLF!#']
te_mlf_lines = ['#!MLF!#']

n_dir = 'net0'
n_path = os.path.join(f_dir, n_dir)
for c_dir in os.listdir(n_path):
    c_path = os.path.join(n_path, c_dir)
    files = os.listdir(c_path)

    for f in files:
        path = os.path.join(c_path, f)

        if random.random() < train_fraction:
            tr_files.append(path)
            tr_mlf_lines.append('"*/%s"' % f.replace('htk', 'lab'))
            tr_mlf_lines.append(c_dir)
            tr_mlf_lines.append('.')
        else:
            te_mlf_lines.append('"*/%s"' % f.replace('htk', 'lab'))
            te_mlf_lines.append(c_dir)
            te_mlf_lines.append('.')
            te_files.append(path)

        if n_features is None:
            with open(path, 'rb') as df:
                header = df.read(12)
                n_samples, period, vecsize, param_kind = \
                        struct.unpack('>iihh', header)
                n_features = vecsize / 4

with open('train.mlf', 'w') as of:
    of.write('\n'.join(tr_mlf_lines))
    of.write('\n') # needed it seems
with open('test.mlf', 'w') as of:
    of.write('\n'.join(te_mlf_lines))
    of.write('\n') # needed it seems

for n_dir in os.listdir(f_dir):
    n_features = None
    n_path = os.path.join(f_dir, n_dir)
    for c_dir in os.listdir(n_path):
        c_path = os.path.join(n_path, c_dir)
        files = os.listdir(c_path)

        for f in files:
            path = os.path.join(c_path, f)
            if n_features is None:
                with open(path, 'rb') as df:
                    header = df.read(12)
                    n_samples, period, vecsize, param_kind = \
                            struct.unpack('>iihh', header)
                    n_features = vecsize / 4

    n_tr_files = [f.replace('net0', n_dir) for f in tr_files]
    n_te_files = [f.replace('net0', n_dir) for f in te_files]
    with open(os.path.join(tr_scp_dir, 'train_%s.scp' % n_dir), 'w') as of:
        of.write('\n'.join(n_tr_files))
    with open(os.path.join(te_scp_dir, 'test_%s.scp' % n_dir), 'w') as of:
        of.write('\n'.join(n_te_files))

    with open(os.path.join(proto_dir, 'proto_%s' % n_dir), 'w') as of:
        proto_lines = ['~o <VecSize> %d <USER>' % n_features, 
                       '~h "proto_%s"' % n_dir]
        proto_lines.append('<BeginHMM>')
        proto_lines.append('\t<NumStates> %d' % (n_states+2))
        for state in range(2,n_states+2):
            proto_lines.append('\t<State> %d' % state)
            proto_lines.append('\t\t<Mean> %d' % n_features)
            proto_lines.append('\t\t\t%s' % \
                                ' '.join(['0.0' for i in range(n_features)]))
            proto_lines.append('\t\t<Variance> %d' % n_features)
            proto_lines.append('\t\t\t%s' % \
                                ' '.join(['1.0' for i in range(n_features)]))
        proto_lines.append('\t<TransP> %d' % (n_states+2))
        transmat = np.ones((n_states+2, n_states+2))
        transmat[:,0] = 0.0
        transmat[-1,:] = 0.0
        for i in range(transmat.shape[0]):
            if i < n_states+1:
                transmat[i,:] /= transmat[i,:].sum()
            proto_lines.append('\t\t%s' % ' '.join(map(str, transmat[i,:])))
        proto_lines.append('<EndHMM>')
        of.write('\n'.join(proto_lines))
