#!/usr/bin/python

from pyfann import libfann as fann
import numpy as np
from matplotlib import pyplot as plt

X = np.random.random((500, 2))
Y = (np.dot(X, [1,1]) + 6.0).reshape((500,1))

X_tr = X[:400,:]
Y_tr = Y[:400,:]
X_te = X[400:,:]
Y_te = Y[400:,:]

train_data = fann.training_data()
test_data = fann.training_data()
train_data.set_train_data(X_tr, Y_tr)
test_data.set_train_data(X_te, Y_te)

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_output = 1
num_hidden = 4
desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

nn = fann.neural_net()
nn.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
nn.set_learning_rate(learning_rate)
nn.set_activation_function_output(fann.LINEAR)
nn.set_activation_function_hidden(fann.SIGMOID_SYMMETRIC_STEPWISE)

nn.train_on_data(train_data, max_iterations, iterations_between_reports,
                 desired_error)

pred = np.zeros(Y_te.shape)
for i in range(Y_te.shape[0]):
    pred[i,:] = nn.run(X_te[i,:])
