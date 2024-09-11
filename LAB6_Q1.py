import numpy as np

# Function for the summation unit
def summation_unit(inputs, weights):
    return np.dot(inputs, weights)

# Activation Functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def tanh_function(x):
    return np.tanh(x)

def relu_function(x):
    return max(0, x)

def leaky_relu_function(x, alpha=0.01):
    return x if x >= 0 else alpha * x

# Error Comparator Unit
def comparator_unit(predicted, actual):
    return predicted - actual
