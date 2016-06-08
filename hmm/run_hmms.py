#!/usr/bin/env python
""" run_hmms.py
    Attempts to automate the process of creating and running hmms etc.
    Steps taken from the HTKBook.
"""

import numpy as np
import os
import random

actions = ['add_saltnpepper', 'add_teabag', 'crack_egg', 'cut_fruit', 'fry_egg',
           'peel_fruit', 'pour_coffee', 'pour_milk', 'take_cup', 'take_knife']
for i in range(11):
    hmm_dir = 'hmm%d_0' % i
    if not os.path.isdir(hmm_dir):
        os.makedirs(hmm_dir)

    os.system('HCompV -C config -f 0.01 -m -S tr_scp/train_net%d.scp '
              '-M %s protos/proto_net%d' % (i,hmm_dir,i))
    
    with open(os.path.join(hmm_dir,'vFloors')) as vf:
        vfloors_text = vf.read()

    proto_name = 'proto_net%d' % i
    with open(os.path.join(hmm_dir, proto_name)) as proto:
        proto_text = proto.read()
        ind = proto_text.find('~h "%s"' % proto_name)
        header = proto_text[:ind]
        hmm_text = proto_text[ind:]

    with open(os.path.join(hmm_dir, 'macros'), 'w') as mf:
        mf.write('\n'.join([header, vfloors_text]))

    with open(os.path.join(hmm_dir, 'hmmdefs'), 'w') as defs:
        defs.write('\n'.join([hmm_text.replace(proto_name, action)
                              for action in actions]))

    for iter_reest in [1,2,3]:
        hmm_dir_old = hmm_dir
        hmm_dir = 'hmm%d_%d' % (i, iter_reest)
        if not os.path.isdir(hmm_dir):
            os.makedirs(hmm_dir)

        os.system('HERest -C config -I train.mlf -t 250.0 150.0 1000.0 '
                  '-S ./tr_scp/train_net%d.scp -H %s/macros -H %s/hmmdefs '
                  '-M %s wlist' % (i, hmm_dir_old, hmm_dir_old, hmm_dir))

    out_mlf_dir = 'mlf_out'
    if not os.path.isdir(out_mlf_dir):
        os.makedirs(out_mlf_dir)
    os.system('HVite -H %s/macros -H %s/hmmdefs -S te_scp/test_net%d.scp '
              '-l \'*\' -i %s/out_net%d.mlf -w wdnet -p 0.0 -s 5.0 dict wlist'\
                % (hmm_dir, hmm_dir, i, out_mlf_dir, i))
    os.system('HResults -I test.mlf wlist %s/out_net%d.mlf'\
                % (out_mlf_dir, i))

# Wisdom of the crowds now.
pred_paths = [os.path.join(out_mlf_dir, p) for p in os.listdir(out_mlf_dir)]
pred_files = map(open, pred_paths)
collected_outfile = os.path.join(out_mlf_dir, 'out_all.mlf')
with open(collected_outfile, 'w') as of:
    all_lines = map(lambda f : f.readlines(), pred_files)
    assert all([len(lines) == len(all_lines[0]) for lines in all_lines])
    for i in range(len(all_lines[0])):
        if i % 3 != 2:
            of.write(all_lines[0][i])
        else:
            pred_lines = [lines[i] for lines in all_lines]
            predictions = [line.split(' ')[2] for line in pred_lines]
            start, end, _, loglik = line.split(' ')
            counts = {p : predictions.count(p) for p in set(predictions)}
            rev_counts = [[] for i in range(12)]
            for p in counts:
                rev_counts[counts[p]].append(p)
            for c in rev_counts[::-1]:
                if len(c) > 0:
                    line = ' '.join([start, end, random.choice(c), loglik])
                    of.write(line)
                    break
for f in pred_files:
    f.close()

os.system('HResults -I test.mlf wlist %s/out_all.mlf'\
            % (out_mlf_dir))
