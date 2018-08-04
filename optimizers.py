import numpy as np


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def _update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            step = self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            params[key] -= step

    def update(self, network):
        if self.h is None:
            self.h = {}
            for name, layer in network.layers.items():
                for key, val in layer.params.items():
                    self.h[key] = np.zeros_like(val)
               
        for name, layer in network.layers.items():
            for key, val in layer.grads.items():
                self.h[key] += val * val
                step = self.lr * val / (np.sqrt(self.h[key]) + 1e-7)
                layer.params[key] -= step
