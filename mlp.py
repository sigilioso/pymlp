# -*- coding: utf-8 -*-

#
# Author: Christian Felipe Álvarez <sigilioso@gmail.com>
#

"""
mlp.py

MLP class definition.
"""

from collections import deque
import activation_functions


class MLP(object):
    """
    Represents a Multilayer Perceptron to execute Backpropagation Algorithm
    """

    # Weights: w[l][i][j] --> weight in layer 'l', from cell 'i' to cell 'j'
    w = []
    # Thresholds u[c][l] --> threshold for cell j in layer l
    u = []

    def __init__(self, layers_definition, activation_function = 'sigmoid'):
        """
        MLP constructor.
        `layers_definition`, a list of integers that represents the number of
        layers and how many cells are in each one. There has to be at least
        three layers: inputs, one layers and outputs. Example:
            layer_definition = [num_inputs, ..., num_cells_i, ..., num_outputs]
        `activation_function`, name of the activation function to use, it has to
        be defined in activation_functions module. It's derivative has also to
        be defined in activation_functions module, named 
        '<function>_derivative'.
        `error_function`, name of the error fuction to use, it has to be
        defined in error_functions module use, it has to be
        defined in error_functions module.
        """
        # Create weights and thresholds according to layers_definition
        for i in range(len(layers_definition)-1):
            self.w.append([])
            for j in range(layers_definition[i]):
                self.w[i].append([0]*layers_definition[i+1])
        for i in layers_definition:
            self.u.append([0]*i)
        # Store num_inputs, num_outputs and functions
        self.num_inputs, self.num_outputs = layers_definition[0], layers_definition[-1]
        self.layers = layers_definition
        self.af = getattr(activation_functions, activation_function)
        self.afd = getattr(activation_functions, 
                '%s_derivative' % activation_function)
        try :
            self.delta = getattr(activation_functions,
                    '%s_delta' % activation_function)
        except AttributeError:
            self.delta = None
        try :
            self.delta_last = getattr(activation_functions,
                    '%s_delta_last_layer' % activation_function)
        except AttributeError:
            self.delta_last = None


    def initialize_weights(self, difference = 0.5):
        """
        Initializes weights in a range (-difference, difference)
        """
        from random import random
        for c in range(len(self.w)):
            for i in range(len(self.w[c])):
                for j in range(len(self.w[c][i])):
                    self.w[c][i][j] = random() * difference - difference

    def initialize_thresholds(self, difference = 0.5):
        """
        Initializes thresholds in a range (-difference, difference)
        """
        from random import random
        for c in range(len(self.u)):
            for i in range(len(self.u[c])):
                self.u[c] = random() * difference - difference

    def calculate_activation(self, pattern):
        """
        Calculate the activation value for each cell.
        The activation value of cells in the last layer are the outputs.
        """
        a = [pattern[0]]
        for l in range(1, len(self.layers)): 
            a.append([self.af(sum([a[l-1][i] * self.w[l][i][j]\
                        for j in range(self.layers[l])]) + self.u[l][i])\
                    for i in range(self.layers[l-1])])
        return a

    def get_output(self, pattern):
        """
        Use the neural network and get the output for a `pattern`.
        """
        return self.calculate_activation(pattern)[-1]

    def backpropagation(self, pattern, learning_rate = 0.5):
        """
        Implents backpropagation method, learning a `pattern` with the learning
        rate specified.
        """
        #TODO Complete documentation describing backpropagation method

        # ---------------------------------------------------------------------
        # Calculate activation and get output from the network
        # ---------------------------------------------------------------------

        a = self.calculate_activation(pattern)

        # ---------------------------------------------------------------------
        # Calculate deltas
        # ---------------------------------------------------------------------

        # Last layer. Apply delta function (or rule) for the last layer.
        d = deque()
        if self.delta_last:
            d.appendleft([self.delta_last(pattern[1][i], a[-1][i]) \
                    for i in range(self.layers_definition[-1])])
        else :
            d.appendleft([(pattern[1][i] - a[-1][i]) *\
                        self.afd(sum([self.w[-2][i][j] * a[-2][i]\
                                for j in range(self.layers_definition[-2])])\
                            + self.u[-1][i])\
                    for i in range(self.layers_definition[-1])])

        # Rest of the layers.
        # Apply delta function (or rule) for each layer l greater than 0.
        # delta_i <-- activation_i, delta[l+1] and every weights from cell i to
        # cells in layer l+1
        if self.delta:
            for l in reversed(range(1, len(self.layers_definition)-1)):
                d.appendleft([self.delta(a[l][i], d[l+1],\
                        [self.w[l][i][j]\
                            for j in range(self.layers_definition[l+1])])\
                    for i in range(self.layers_definition[l])
                    ])
        else :
            for l in reversed(range(1, len(self.layers_definition)-1)):
                d.appendleft([self.afd(sum([self.w[l-1][i][j] * a[l-1][i]\
                            for j in range(self.layers_definition[l-1])])\
                        + self.u[-1][i]) *\
                        sum([map(lambda d, w: d * w,
                            d[l+1],
                            [self.w[l][i][j] \
                                    for j in range(self.layers_definition[l+1])]
                            )])
                        for i in range(self.layers_definition[l]) 
                        ])
        
        # ---------------------------------------------------------------------
        # Adjust weights and thresholds
        # ---------------------------------------------------------------------

        # Thresholds in the last layer
        self.u[-1] = map(lambda u, d: u + learning_rate * d, self.u[-1], d[-1])

        # Weights and the rest of thresholds
        for l in reversed(range(len(self.layers_definition)-1)):
            for i in range(self.layers_definition[l]):
                for j in range(self.layers_definition[l+1]):
                    self.w[l][i][j] += learning_rate * a[l][i] * d[l+1][j] 
                self.u[l][i] += learning_rate * d[l][i]



