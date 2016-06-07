""" io.py
"""

from response import Response
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as mani

import os
import scipy.misc

def load_responses(exp_type):
    if exp_type == 'grating':
        from params.grating.datafile_params import DATA_DIR, MICE_NAMES
        data_locs = [os.path.join(DATA_DIR,'%s_dir.npy'%c) for c in MICE_NAMES]
        data = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES,data_locs))
    elif exp_type == 'natural':
        from params.naturalmovies.datafile_params import DATA_DIR
        data_locs = [os.path.join(DATA_DIR, '%d.npy' % i) for i in range(11)]
        data = [Response(str(i), data_locs[i]) for i in range(11)]

    return data

def load_movies(exp_type, movie_type, downsample_factor=1):
    if exp_type == 'grating':
        from params.grating.datafile_params import MOVIE_DIR
        from params.grating.stimulus_params import N_MOVIES
    elif exp_type == 'natural':
        from params.naturalmovies.datafile_params import MOVIE_DIR
        from params.naturalmovies.stimulus_params import N_MOVIES

    if downsample_factor > 1:
        movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s_down' % movie_type,
                                   '%d.npy' % downsample_factor)
                      for s in range(N_MOVIES)]
    else:
        movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s.npy' % movie_type)
                      for s in range(N_MOVIES)]
    movies = map(np.load, movie_locs)
    return movies

# First outputting each frame as png or something.
# Then join them using avconv/ffmpeg later.
def dump_movie(output_dir, movie, fps, movie_type=''):
    if movie_type is None or len(movie_type) == 0:
        suffix = ''
    else:
        suffix = '_%s' % movie_type

    frames_dir_path = os.path.join(output_dir, 'frames%s' % suffix)
    if not os.path.isdir(frames_dir_path):
        os.makedirs(frames_dir_path)

    # Save movie array itself.
    movie_path = os.path.join(output_dir, 'movie%s.npy' % suffix)
    np.save(movie_path, movie)

    # Now the frames.
    T = movie.shape[2]
    for t in range(T):
        scipy.misc.toimage(movie[:,:,t]).save(os.path.join(frames_dir_path,
                                                           '%03d.png' % t))

    # And an mp4 for completeness.
    def update_img(n):
        img.set_data(movie[:,:,n])
        return img

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img = ax.imshow(movie[:,:,0], cmap='gray', interpolation='nearest')
    a = mani.FuncAnimation(fig, update_img, T, repeat=False,
                           interval=1000.0/fps)
    writer = mani.writers['avconv'](fps=fps)

    mp4_path = movie_path.replace('npy', 'mp4')
    a.save(mp4_path, writer=writer,dpi=100)
    plt.close()
