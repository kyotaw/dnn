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
        self.conv_filter_params = input_dim
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.params = {}
        self.layers = OrderedDict()
        conv_1 = convolution(conv_filter_params, std)
        self.layers[key_conv(1)] = conv_1
        self.params[key_W(1)] = conv_1.W
        self.params[key_b(1)] = conv_1.b
        self.layers[key_activation(1)] = relu()
        self.layers[key_pool(1)] = pooling(height=2, width=2, stride=2)

        conv_h, conv_w = conv_1.output_size(input_dim['height'], input_dim['width'])
        pool_output_size = int(conv_filter_params['num'] * (conv_h / 2) * (conv_w / 2))
        affine_1 = affine(pool_output_size, hidden_size, std)
        self.layers[key_affine(1)] = affine_1
        self.params[key_W(2)] = affine_1.W
        self.params[key_b(2)] = affine_1.b
        self.layers['relu2'] = relu()
        affine_2 = affine(hidden_size, output_size, std)
        self.layers[key_affine(2)] = affine_2
        self.params[key_W(3)] = affine_2.W
        self.params[key_b(3)] = affine_2.b
        self.output_layer = SoftmaxLayer(CrossEntropyError())

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
        grads[key_W(1)], grads[key_b(1)] = self.layers[key_conv(1)].dW, self.layers[key_conv(1)].db
        grads[key_W(2)], grads[key_b(2)] = self.layers[key_affine(1)].dW, self.layers[key_affine(1)].db
        grads[key_W(3)], grads[key_b(3)] = self.layers[key_affine(2)].dW, self.layers[key_affine(2)].db

        return grads
