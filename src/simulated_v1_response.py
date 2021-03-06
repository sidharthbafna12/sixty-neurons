""" simulated_v1_response.py
    To compute the output of a multilayer perceptron for a given video.

    Since the multilayer perceptron was trained to fit V1 responses to natural
    movies, this is a simulated mouse V1 response to the movie supplied.

    An MLP can also be created if its parameters are supplied.
"""

import numpy as np
from nnet_regression import MLPRegression
import theano

def transform(mlp, video):
    n_in = mlp.n_in
    ly, lx, T = video.shape
    n_lag = n_in / (ly * lx)
    
    padded_video = np.pad(video, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
    x = np.empty((T, n_in))
    for t in range(T):
        x[t,:] = padded_video[:,:,t:t+n_lag].flatten()
    
    X = theano.shared(x.astype(np.float32), name='X', borrow=True)
    pred = theano.function([], mlp.y_pred,
            givens={mlp.hidden_layer.input : X})

    return pred()

def create_mlp_from_params(Wh, bh, Wo, bo, rectify=False):
    n_in, n_hidden = Wh.get_value().shape
    n_hidden, n_out = Wo.get_value().shape

    x = theano.tensor.matrix('x')
    mlp = MLPRegression(input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                        Wh=Wh, bh=bh, Wo=Wo, bo=bo, rectify=rectify)

    return mlp
