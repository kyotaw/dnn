from collections import OrderedDict

from dnn.dnn import *
from dnn.utils import *
from dnn.losses import *
from dnn.weight_initializers import he


def _make_conv_params(num, channels, height, width, stride, pad):
    return {
        'num': num,
        'channels': channels,
        'height': height,
        'width': width,
        'stride': stride,
        'pad': pad 
    }
    

def _make_pool_params(height, width, stride, pad):
    return {
        'height': height,
        'width': width,
        'stride': stride,
        'pad': pad
    }


class Vgg16:
    def __init__(self):
        self.params = {}
        self.layers = OrderedDict()

        # section1
        conv_params = _make_conv_params(64, 3, 3, 3, 1, 1)
        self._add_conv('conv_1_1', conv_params, 0.01)
        self._add_relu('relu_1_1')
        
        conv_params = _make_conv_params(64, 64, 3, 3, 1, 1)
        self._add_conv('conv_1_2', conv_params, 0.01)
        self._add_relu('relu_1_2')
        
        pool_params = _make_pool_params(2, 2, 2, 0)
        self._add_pooling('pool_1_1', pool_params)

        # section2
        conv_params = _make_conv_params(128, 64, 3, 3, 1, 1)
        self._add_conv('conv_2_1', conv_params, 0.01)
        self._add_relu('relu_2_1')

        conv_params = _make_conv_params(128, 128, 3, 3, 1, 1)
        self._add_conv('conv_2_2', conv_params, 0.01)
        self._add_relu('relu_2_2')

        self._add_pooling('pool_2_1', pool_params)

        # section3
        conv_params = _make_conv_params(256, 128, 3, 3, 1, 1)
        self._add_conv('conv_3_1', conv_params, 0.01)
        self._add_relu('relu_3_1')

        conv_params = _make_conv_params(256, 256, 3, 3, 1, 1)
        self._add_conv('conv_3_2', conv_params, 0.01)
        self._add_relu('relu_3_2')

        self._add_conv('conv_3_3', conv_params, 0.01)
        self._add_relu('relu_3_3')
        
        self._add_pooling('pool_3_1', pool_params)

        # section4
        conv_params = _make_conv_params(512, 256, 3, 3, 1, 1)
        self._add_conv('conv_4_1', conv_params, 0.01)
        self._add_relu('relu_4_1')
        
        conv_params = _make_conv_params(512, 512, 3, 3, 1, 1)
        self._add_conv('conv_4_2', conv_params, 0.01)
        self._add_relu('relu_4_2')

        self._add_conv('conv_4_3', conv_params, 0.01)
        self._add_relu('relu_4_3')

        self._add_pooling('pool_4_1', pool_params)

        # section5
        conv_params = _make_conv_params(512, 512, 3, 3, 1, 1)
        self._add_conv('conv_5_1', conv_params, 0.01)
        self._add_relu('relu_5_1')

        self._add_conv('conv_5_2', conv_params, 0.01)
        self._add_relu('relu_5_2')

        self._add_conv('conv_5_3', conv_params, 0.01)
        self._add_relu('relu_5_3')
    
        self._add_pooling('pool_5_1', pool_params)

        # section6
        self._add_affine('affine_6_1', 25088, 4096)
        self._add_affine('affine_6_2', 4096, 4096)
        self._add_affine('affine_6_3', 4096, 1000)

        # output
        self._add_output()
    
    def _add_conv(self, name, params, std):
        conv = convolution(name, params, 0.01)
        self.layers[name] = conv
        self.params[key_W(name)] = conv.W
        self.params[key_b(name)] = conv.b
   
    def _add_relu(self, name):
        rel = relu(name)
        self.layers[name] = rel

    def _add_pooling(self, name, params):
        pool = pooling(name, params['height'], params['width'], params['stride'], params['pad'])
        self.layers[name] = pool

    def _add_affine(self, name, input_size, output_size):
        std = he(input_size)
        aff = affine(name, input_size, output_size, std)
        self.layers[name] = aff
        self.params[key_W(name)] = aff.W
        self.params[key_b(name)] = aff.b

    def _add_output(self):
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
