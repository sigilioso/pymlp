# -*- coding: utf-8 -*-

#
# Author: Christian Felipe Álvarez <sigilioso@gmail.com>
#

"""
activation_functions.py

Defines functions to use in backpropagation algorithm.
If a function `f` is provided, `f_derivative` has also to be provided and
it  has to be `f`'s derivative.
If possible, `f_delta` and `f_delta_last_layer` should be provided in order
to optimize delta rule.
- `f_delta_last_layers` recibes `s` as the desired output and `y` as the output
which has been obtained by the network.
- `f_delta` recibes `a` as the activation value for a cell, W as the weights
vector containing weights from that cell to every cells in next layer and `D`
as the delta values vector for each cell in the next layer.
"""

import math

def sigmoid(x):
    return 1/(1 + math.e**(-1*x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1- sigmoid(x))

def sigmoid_delta(a, W, D):
    return (a * (1 - a)) * (sum(map(lambda w, d: w * d, W, D)))

def sigmod_delta_last_layer(s, y):
    return (s - y) * y * (1 - y)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def tanh_delta(a, W, D):
    return (1 - a ** 2) * (sum(map(lambda w, d: w * d, W, D)))

def tanh_delta_last_layer(s, y):
    return (s - y) * (1 - y ** 2)
