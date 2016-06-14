import numpy as np
import theano
import theano.tensor as T


def load_data(n_ex, n_features_in, n_features_out):
    print('Creating data...')
    data_input = np.random.random((n_ex, n_features_in))
    weights = (1.0/n_features_in) * np.ones((n_features_in, n_features_out))
    print '*\n', weights, '\n*'
    data_output = np.dot(data_input, weights)\
                + 0.01 * np.random.random((n_ex, n_features_out))
    
    t1, t2 = 0.6 * n_ex, 0.8 * n_ex
    train_input, train_output = data_input[:t1,:], data_output[:t1,:]
    val_input,val_output = data_input[t1:t2,:], data_output[t1:t2,:]
    test_input, test_output = data_input[t2:,:], data_output[t2:,:]

    print('Data created...')
    rval = [(train_input, train_output), (val_input, val_output),
            (test_input, test_output)]
    return rval

def linear_regression_test(learning_rate=0.13, n_epochs=1000, batch_size=600):
    n_ex, n_in, n_out = 10000, 100, 2
    datasets = load_data(n_ex, n_in, n_out)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    W = T.matrix('W')
    X = T.matrix('X')
    Y = T.matrix('Y')
    error = T.mean((Y - T.dot(X, W)) ** 2.0)
    
    err = theano.function([W, X, Y], error)
    grad_err = theano.function([W, X, Y], T.grad(error, W))
    
    weights = np.random.random((n_in, n_out))
    for i in range(n_epochs):
        err_val = err(weights, train_set_x, train_set_y)
        err_grad_val = grad_err(weights, train_set_x, train_set_y)
        weights -= (learning_rate * err_grad_val)
        print 'Iteration %d, error %f' % (i, err_val)
        # print weights, '\n'
    print weights
    print err(weights, test_set_x, test_set_y)

if __name__ == '__main__':
    linear_regression_test(0.03, 5000)
