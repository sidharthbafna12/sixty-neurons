""" linear_response_model.py
"""

import numpy as np

from nn_lib_tutorials.theano.src.linear_sgd import shared_dataset
from nn_lib_tutorials.theano.src.linear_sgd import LinearRegression
from src.io import load_responses, load_movies
from src.data_manip_utils import smooth_responses, train_test_split

import cPickle as pickle

import theano
import theano.tensor as T

import os
import sys
import timeit

def window_matrices(rsp, movs, n_lag):
    def movie_window_matrix(mov, n_lag):
        LY, LX, T = mov.shape
        p_mov = np.pad(mov, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
        mat = np.empty((T, LY*LX*n_lag))
        for t in range(T):
            mat[t,:] = p_mov[:,:,t:t+n_lag].flatten()
        return mat

    def response_window_matrix(rsp, n_lag):
        S,N,T,R = rsp.data.shape
        mat = np.empty((T*R*S,N))

        i = 0
        for s in range(S):
            for r in range(R):
                for t in range(T):
                    mat[i,:] = rsp.data[s,:,t,r]
                    i += 1
        return mat

    mov_mats = [movie_window_matrix(m, n_lag) for m in movs]
    rsp_mat = response_window_matrix(rsp, n_lag)
    n_trials = rsp.data.shape[3]
    
    mov_mat = np.vstack([np.vstack([m for i in range(n_trials)])
                         for m in mov_mats])
    
    return shared_dataset((rsp_mat, mov_mat))

class LinearRegressionRL(LinearRegression): # Rectified Linear unit
    def __init__(self, input, n_in, n_out):
        LinearRegression.__init__(self, input, n_in, n_out)
        self.y_pred = T.maximum(T.dot(input, self.W) + self.b, 0.0)

def fit_linear_model(learning_rate=1.0e-1, n_epochs=30, batch_size=600):
    exp_type = 'natural'
    movie_type = 'movie'
    spatial_downsample_factor = 4
    n_lag = 6
    saved_models_dir = './temp/linear-models'
    if not os.path.isdir(saved_models_dir):
        os.makedirs(saved_models_dir)

    responses = load_responses(exp_type)
    movies = load_movies(exp_type, movie_type,
                         downsample_factor=spatial_downsample_factor)
    
    for i, response in enumerate(responses):
        name = response.name
        print 'Mouse %s' % name

        print 'Splitting out training and test data...'
        tr_rsp, tst_rsp, tr_mov, tst_mov = train_test_split(response, movies,
                                                        'even', train_frac=0.8)

        print 'Splitting out training and validation data...'
        tr_rsp, val_rsp, tr_mov, val_mov = train_test_split(tr_rsp, tr_mov,
                                                        'even', train_frac=0.9)

        tr_rsp = smooth_responses(tr_rsp)
        val_rsp = smooth_responses(val_rsp)
        tst_rsp = smooth_responses(tst_rsp)

        train_set_x, train_set_y = window_matrices(tr_rsp, tr_mov, n_lag)
        valid_set_x, valid_set_y = window_matrices(val_rsp, val_mov, n_lag)
        test_set_x, test_set_y = window_matrices(tst_rsp, tst_mov, n_lag)

        n_in = train_set_x.get_value(borrow=True).shape[1] 
        n_out = train_set_y.get_value(borrow=True).shape[1]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.matrix('y')  # neural responses (actual)
        
        regression = LinearRegression(input=x, n_in=n_in, n_out=n_out)
        cost = regression.MSE(y)

        test_model = theano.function([index], cost,
                        givens={
                            x: test_set_x[index*batch_size:(index+1)*batch_size],
                            y: test_set_y[index*batch_size:(index+1)*batch_size]})
        validate_model = theano.function([index], cost,
                         givens={
                            x: valid_set_x[index*batch_size:(index+1)*batch_size],
                            y: valid_set_y[index*batch_size:(index+1)*batch_size]})

        g_W, g_b = T.grad(cost, [regression.W, regression.b])
        updates = [(regression.W, regression.W - learning_rate*g_W),
                   (regression.b, regression.b - learning_rate*g_b)]

        train_model = theano.function([index], cost, updates=updates,
                        givens={
                            x: train_set_x[index*batch_size:(index+1)*batch_size],
                            y: train_set_y[index*batch_size:(index+1)*batch_size]})

        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score
                            )
                        )

                        # save the best model
                        model_outpath = os.path.join(saved_models_dir,
                                                     'model_%s' % name)
                        with open(model_outpath, 'wb') as f:
                            pickle.dump(regression, f)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f,'
                'with test performance %f'
            )
            % (best_validation_loss, test_score)
        )
        print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))

if __name__ == "__main__":
    fit_linear_model()
