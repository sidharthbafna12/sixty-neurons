"""
    Linear regression using Theano and stochastic gradient descent.
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import theano
import theano.tensor as T
import numpy as np

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(data_x, borrow=borrow)
    shared_y = theano.shared(data_y, borrow=borrow)
    return shared_x, shared_y

def load_data(n_ex, n_features_in, n_features_out):
    print('Creating data...')
    data_input = np.random.random((n_ex, n_features_in))
    weights = (1.0/n_features_in) * np.ones((n_features_in, n_features_out))
    print('*\n')
    print(weights)
    print('\n*')
    data_output = np.dot(data_input, weights)\
                + 0.01 * np.random.random((n_ex, n_features_out))
    
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

class LinearRegression:
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
                value=np.zeros((n_in,n_out), dtype=theano.config.floatX),
                name='W',
                borrow=True)
        self.b = theano.shared(
                value=np.zeros(n_out, dtype=theano.config.floatX),
                name='b',
                borrow=True)
        
        self.y_pred = T.dot(input, self.W) + self.b

        self.params = [self.W, self.b]
        self.input = input

    def MSE(self, y):
        return T.mean((y - self.y_pred) ** 2.0)


def regression_example(sizes, learning_rate=1.0e-2, n_epochs=1000, batch_size=600):
    n_ex, n_in, n_out = sizes
    datasets = load_data(n_ex, n_in, n_out)
    
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
                    with open('best_model.pkl', 'wb') as f:
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
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    print('Regression parameter W : %s' % str(regression.W.get_value()))
    print('Regression parameter b : %s' % str(regression.b.get_value()))

if __name__ == "__main__":
    regression_example((10000, 1000, 60), learning_rate=0.005, n_epochs=3000)
