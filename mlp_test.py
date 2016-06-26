#!/usr/bin/env python
""" mlp_test.py
    Verifying that I can run Theano for a simple regression task.
    Also that my classes in src.nnet_regression are working correctly.
"""
from __future__ import print_function
from src.nnet_regression import MLPRegression, shared_dataset
from src.nnet_regression import RegressionModel

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import theano
import theano.tensor as T
import numpy as np

theano.config.openmp = True

def load_data(n_ex, n_features_in, n_features_out):
    print('Creating data...')
    data_input = np.random.random((n_ex, n_features_in))
    weights = (1.0/n_features_in) * np.ones((n_features_in, n_features_out))
    print('*\n')
    print(weights)
    print('\n*')
    data_output = np.dot(data_input, weights)\
                + 0.1 * np.random.random((n_ex, n_features_out)) + 1.0
    
    t1, t2 = int(0.6 * n_ex), int(0.8 * n_ex)
    train_input, train_output = data_input[:t1,:], data_output[:t1,:]
    val_input,val_output = data_input[t1:t2,:], data_output[t1:t2,:]
    test_input, test_output = data_input[t2:,:], data_output[t2:,:]
    print('Data created...')

    test_x, test_y = shared_dataset((test_input, test_output))
    train_x, train_y = shared_dataset((train_input, train_output))
    val_x, val_y = shared_dataset((val_input, val_output))

    rval = [(train_x, train_y), (val_x, val_y), (test_x, test_y)]
    return rval

def test_mlp(sizes, learning_rate=0.15, n_epochs=100, batch_size=600,
             L1_reg=1.0e-4, L2_reg=0.0e-4):
    n_ex, n_in, n_hidden, n_out = sizes
    datasets = load_data(n_ex, n_in, n_out)

    regmodel = RegressionModel('abc', n_hidden=n_hidden,
                               learning_rate=learning_rate,
                               n_epochs=n_epochs,
                               batch_size=batch_size,
                               L1_reg=L1_reg,
                               L2_reg=L2_reg)
    regmodel.setup_with_data(datasets)
    regmodel.train()

    print(map(lambda d: d.get_value(), regmodel.regression.params))
    print(regmodel.y_pred(), datasets[2][1].get_value())

    """
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

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
    
    regression = MLPRegression(input=x,n_in=n_in,n_hidden=n_hidden,n_out=n_out)

    model_predictions = theano.function([], regression.y_pred,
                                        givens={x: test_set_x})
    cost = regression.error(y) + L1_reg*regression.L1 + L2_reg*regression.L2_sq

    test_model = theano.function([index], cost,
                    givens={
                        x: test_set_x[index*batch_size:(index+1)*batch_size],
                        y: test_set_y[index*batch_size:(index+1)*batch_size]})
    validate_model = theano.function([index], cost,
                     givens={
                        x: valid_set_x[index*batch_size:(index+1)*batch_size],
                        y: valid_set_y[index*batch_size:(index+1)*batch_size]})

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in regression.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(regression.params, gparams)
    ]

    train_model = theano.function([index], cost, updates=updates,
                    givens={
                        x: train_set_x[index*batch_size:(index+1)*batch_size],
                        y: train_set_y[index*batch_size:(index+1)*batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
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
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f '
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, test_score))

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    print(map(lambda d: d.get_value(), regression.params))
    print(model_predictions(), test_set_y.get_value())
    """

if __name__ == "__main__":
    test_mlp((100000, 100, 10, 3), learning_rate=0.05, n_epochs=3000)
