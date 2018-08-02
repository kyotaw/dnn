from collections import OrderedDict

from mine.backend.neuralnet.layers import *
from mine.backend.neuralnet.losses import *
from mine.backend.neuralnet.weight_initializers import *
from mine.backend.neuralnet.utils import *


weight_initializers = {
    'he': HeInitializer,
    'relu': HeInitializer,
}

activation_layers = {
    'relu': ReluLayer,
}


class AffineNetwork:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size_list,
        activation,
        weight_initializer,
        use_batch_norm=True):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.use_batch_norm = use_batch_norm
        self.params = weight_initializers[weight_initializer].init_weights(input_size, output_size, hidden_size_list, use_batch_norm=use_batch_norm)
        self.layers = self._init_hidden_layers(len(hidden_size_list), self.params, activation)
        self.output_layer = SoftmaxLayer(CrossEntropyError())


    def _init_hidden_layers(self, hidden_num, weights, activation):
        layers = OrderedDict()
        for i in range(1, hidden_num + 1):
            layers[key_affine(i)] = AffineLayer(weights[key_W(i)], weights[key_b(i)])
            if self.use_batch_norm:
                layers[(key_batch_norm(i))] = BatchNormalization(weights[key_gamma(i)], weights[key_beta(i)])
            layers[key_activation(i)] = activation_layers[activation]()

        i = hidden_num + 1
        layers[key_affine(hidden_num + 1)] = AffineLayer(weights[key_W(i)], weights[key_b(i)])

        return layers

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
        hidden_layer_num = len(self.hidden_size_list)
        for i in range(1, hidden_layer_num + 2):
            grads[key_W(i)] = self.layers[key_affine(i)].dW
            grads[key_b(i)] = self.layers[key_affine(i)].db

            if self.use_batch_norm and i != hidden_layer_num + 1:
                grads[key_gamma(i)] = self.layers[key_batch_norm(i)].dgamma
                grads[key_beta(i)] = self.layers[key_batch_norm(i)].dbeta

        return grads
