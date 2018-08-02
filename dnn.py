import numpy as np

from dnn.layers import *


def affine(input_size, output_size, std):
    W = std * np.random.randn(input_size, output_size)
    b = np.zeros(output_size)
    return AffineLayer(W, b)


def convolution(filter_params, std):
    W = std * np.random.randn(
        filter_params['num'],
        filter_params['channels'],
        filter_params['height'],
        filter_params['width'])
    b = np.zeros(filter_params['num'])

    return Convolution(W, b, filter_params['stride'], filter_params['pad'])

    
def pooling(height, width, stride, pad=0):
    return Pooling(height, width, stride, pad)


def relu():
    return ReluLayer()
