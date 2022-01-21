""" Activations handling"""

import numpy as np

def compute_activation(layer_out, activation_type):
    """ Computes the activation based on the input activation type."""

    if activation_type == "relu":
        return _relu(layer_out)
    elif activation_type == "sigmoid":
        return _sigmoid(layer_out)
    elif activation_type == "softmax":
        return _softmax(layer_out)

def _relu(layer_out):
    """ Calculates the relu function"""

    return np.maximum(np.zeros(layer_out.shape), layer_out)

def _sigmoid(layer_out):
    """ Calculates the sigmoid function"""

    return 1 / (1 + np.exp(-layer_out))

def _softmax(layer_out):
    """ Calculates the softmax function"""
    
    return np.exp(layer_out) / np.exp(layer_out).sum(1, keepdims=True)