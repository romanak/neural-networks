#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Mon November 29 2021

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation.

@author: Roman Akchurin
"""

import numpy as np

class Network(object):
    def __init__(self, sizes, seed=None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [3, 4, 2] then it would be a three-layer network, with the
        first layer containing 3 neurons, the second layer 4 neurons,
        and the third layer 2 neurons. The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0 and variance 1. Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.rng = np.random.default_rng(seed)
        self.biases = [self.rng.standard_normal((j, 1)) for j in sizes[1:]]
        self.weights = [self.rng.standard_normal((j, k)) for j, k in \
            zip(sizes[1:], sizes[:-1])]

    def SGD(self, train_data, epochs, mini_batch_size, eta, \
            test_data=None):
        """Trains the neural network using mini_batch stochastic
        gradient descent. The ``train_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs."""
        n = len(train_data)
        for j in range(epochs):
            # divide train data into mini_batches randomly
            self.rng.shuffle(train_data)
            mini_batches = [train_data[m:m+mini_batch_size] \
                for m in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, mini_batch_size, eta)
            # evaluate metrics
            if test_data:
                correct = self.evaluate(test_data)
                print(f"Epoch {j} : {correct} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, mini_batch_size, eta):
        """Updates the network's weights and biases by applying
        gradient descent using backpropagation to a single mini_batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/mini_batch_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/mini_batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """Returns a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x for an example in the
        mini_batch. ``nabla_b`` and ``nabla_w`` are layer-by-layer
        lists of numpy arrays, similar to ``self.biases`` and
        ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        (zs, activations) = self.feedforward(x)
        # compute dC/dz, dC/db, and dC/dw for the output layer
        dCda = self.cost_derivative(activations[-1], y)
        sp = sigmoid_prime(zs[-1])
        delta = np.multiply(dCda, sp)
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose())
        # compute dC/dz, dC/db, and dC/dw for all the previous layers
        for l in range(2, self.num_layers):
            d = np.matmul(self.weights[-l+1].transpose(), delta)
            sp = sigmoid_prime(zs[-l])
            delta =  np.multiply(d, sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.matmul(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def feedforward(self, x):
        """Returns matrix of ``zs`` and ``activations`` of the mini_batch."""
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        # list to store all the weighted input z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return (zs, activations)

    def evaluate(self, test_data):
        """Returns the number of correctly identified
        examples."""
        test_results = []
        for x, y in test_data:
            (zs, activations) = self.feedforward(x)
            test_results.append((np.argmax(activations[-1]), y))
        correct = sum(int(x == y) for (x, y) in test_results)
        return correct

    def cost_derivative(self, output_activations, y):
        """Returns the vector of partial derivatives dC_x/da
        for the output activations."""
        return (output_activations - y)
    
def sigmoid(z):
    """Returns the sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Returns the derivative of the sigmoid function."""
    s = sigmoid(z)
    return s*(1-s)

if __name__ == "__main__":
    import mnist
    (train_data, valid_data, test_data) = mnist.load_processed_data()
    net = Network([784, 15, 10], seed=0)
    net.SGD(train_data, 10, 30, 3.0, test_data)
