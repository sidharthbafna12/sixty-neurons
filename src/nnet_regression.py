""" nnet_regression.py
    A Theano-based multilayer perceptron model to fit V1 responses to stimulus
    videos.
    
    Sliding windows are expected to be computed already. These classes expect a
    standard (n_samples, n_features)-shape kind of data representation.

    A stochastic gradient descent with minibatches is implemented in
    RegressionModel, or rather copied over from the Theano tutorials at
    deeplearning.net.
"""

import numpy as np
import theano
import theano.tensor as T

import timeit

def shared_dataset(data_xy, borrow=True):
    x, y = data_xy
    sh_x = theano.shared(x.astype(np.float32), borrow=borrow)
    sh_y = theano.shared(y.astype(np.float32), borrow=borrow)
    return sh_x, sh_y

class LinearLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,rectify=False):
        if W is None:
            W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W',
                    borrow=True)
        if b is None:
            b = theano.shared(
                    value=np.zeros(n_out, dtype=theano.config.floatX),
                    name='b',
                    borrow=True)
        
        self.W = W
        self.b = b
        
        if not rectify:
            self.y_pred = T.dot(input, self.W) + self.b
        else:
            self.y_pred = T.maximum(T.dot(input, self.W) + self.b, 0.0)

        self.params = [self.W, self.b]
        self.input = input

    def error(self, y):
        return T.mean((y - self.y_pred) ** 2.0)

# start-snippet-1 (stolen from deeplearning.net)
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLPRegression:
    def __init__(self, input, n_in, n_hidden, n_out, rectify=False,
                 Wh=None, bh=None, Wo=None, bo = None):
        self.hidden_layer = HiddenLayer(input=input, n_in=n_in, n_out=n_hidden,
                                        W=Wh, b=bh, activation=T.tanh)
        
        self.linear_layer = LinearLayer(input=self.hidden_layer.output,
                                        n_in=n_hidden, n_out=n_out,
                                        W=Wo, b=bo, rectify=rectify)

        self.L1 = (
            abs(self.hidden_layer.W).sum()
            + abs(self.linear_layer.W).sum()
        )

        self.L2_sq = (
                (self.hidden_layer.W ** 2).sum()
                + (self.linear_layer.W ** 2).sum()
        )

        self.y_pred = self.linear_layer.y_pred
        self.error = self.linear_layer.error

        self.params = self.hidden_layer.params + self.linear_layer.params
        self.input = input

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

class RegressionModel:
    def __init__(self, model_name, n_hidden,
            learning_rate=1.0e-2, n_epochs=1000, batch_size=500,
            L1_reg=1.0e-4, L2_reg=1.0e-4):
        self.name = model_name
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.n_hidden = n_hidden
        
    def setup_with_data(self, data):
        train, validation, test = data
        train_set_x, train_set_y = train
        valid_set_x, valid_set_y = validation
        test_set_x, test_set_y = test
        
        self.n_in = \
                train_set_x.get_value(borrow=True).shape[1]
        self.n_out = \
                train_set_y.get_value(borrow=True).shape[1]

        self.n_train_batches = \
                train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_valid_batches = \
                valid_set_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_test_batches = \
                test_set_x.get_value(borrow=True).shape[0] // self.batch_size

        print 'Building the model...'
        index = T.lscalar() # Index to minibatch

        x = T.matrix('x')
        y = T.matrix('y')

        self.regression = MLPRegression(input=x,
                                        n_in=self.n_in,
                                        n_hidden=self.n_hidden,
                                        n_out=self.n_out,
                                        rectify=True)

        cost = (self.regression.error(y) 
                + self.L1_reg * self.regression.L1
                + self.L2_reg * self.regression.L2_sq)

        # Set up the functions using the actual data. 
        self.y_pred = theano.function([], self.regression.y_pred,
                                      givens={x : test_set_x})

        self.test_model = \
            theano.function([index], cost,
                givens={
                x: test_set_x[index*self.batch_size:(index+1)*self.batch_size],
                y: test_set_y[index*self.batch_size:(index+1)*self.batch_size]})
        self.validate_model = \
            theano.function([index], cost,
            givens={
            x: valid_set_x[index*self.batch_size:(index+1)*self.batch_size],
            y: valid_set_y[index*self.batch_size:(index+1)*self.batch_size]})

        gparams = [T.grad(cost, p) for p in self.regression.params]
        updates = [(p, p - self.learning_rate * gp)
                   for p, gp in zip(self.regression.params, gparams)]

        self.train_model = \
            theano.function([index], cost, updates=updates,
            givens={
            x: train_set_x[index*self.batch_size:(index+1)*self.batch_size],
            y: train_set_y[index*self.batch_size:(index+1)*self.batch_size]})
    
    def train(self):
        print('... training')

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience // 2)
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

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):

                minibatch_avg_cost = self.train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i)
                                         for i in range(self.n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print 'epoch %i, minibatch %i/%i, validation error %f' \
                            % (epoch, minibatch_index + 1,
                               self.n_train_batches, this_validation_loss)

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
                        test_losses = [self.test_model(i)
                                       for i in range(self.n_test_batches)]
                        test_score = np.mean(test_losses)

                        print '     epoch %i, minibatch %i/%i, test error of '\
                              'best model %f' % (epoch, minibatch_index + 1,
                                                 self.n_train_batches,
                                                 test_score)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print 'Optimization complete. Best validation score of %f '\
              'obtained at iteration %i, with test performance %f' \
              % (best_validation_loss, best_iter + 1, test_score)

        print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)
        return test_score
