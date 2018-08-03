from collections import OrderedDict

from dnn.dnn import *
from dnn.utils import *
from dnn.losses import *


class ConvolutionalNetwork:
    def __init__(
        self,
        input_dim, # {cannels, height, width}
        conv_filter_params, # {num, channels, height, width, pad, stride}
        output_size,
        hidden_size,
        std=0.01):

        self.input_dim = input_dim
        self.conv_filter_params = conv_filter_params
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.params = {}
        self.layers = OrderedDict()
        conv_1 = convolution('conv_1', conv_filter_params, std)
        self.layers[conv_1.name] = conv_1
        self.params[key_W(conv_1.name)] = conv_1.W
        self.params[key_b(conv_1.name)] = conv_1.b
        relu_1 = relu('relu_1')
        self.layers[relu_1.name] = relu_1
        pool_1 = pooling('pool_1', height=2, width=2, stride=2)
        self.layers[pool_1.name] = pool_1

        conv_h, conv_w = conv_1.output_size(input_dim['height'], input_dim['width'])
        pool_output_size = int(conv_filter_params['num'] * (conv_h / 2) * (conv_w / 2))
        affine_1 = affine('affine_1', pool_output_size, hidden_size, std)
        self.layers[affine_1.name] = affine_1
        self.params[key_W(affine_1.name)] = affine_1.W
        self.params[key_b(affine_1.name)] = affine_1.b
        relu_2 = relu('relu_2')
        self.layers[relu_2.name] = relu_2
        affine_2 = affine('affine_2', hidden_size, output_size, std)
        self.layers[affine_2.name] = affine_2
        self.params[key_W(affine_2.name)] = affine_2.W
        self.params[key_b(affine_2.name)] = affine_2.b
        self.output_layer = SoftmaxLayer('softmax', CrossEntropyError())

    def predict(self, x, is_train=False):
        for key, layer in self.layers.items():
            x = layer.forward(x, is_train)

        return x

    def loss(self, x, t, is_train=False):
        y = self.predict(x, is_train)
        y, loss = self.output_layer.forward(y, t)
        return loss
        
    def accuracy(self, x, t):
        y = self.predict(x, is_train=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        loss = self.loss(x, t, is_train=True)
        dy = 1
        dy = self.output_layer.backward(dy)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dy = layer.backward(dy)

        grads = {}
        for name, layer in self.layers.items():
            if key_W(name) in self.params and key_b(name) in self.params:
                grads[key_W(name)] = layer.dW
                grads[key_b(name)] = layer.db
            
        return grads
