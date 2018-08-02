import numpy as np


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            step = self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            params[key] -= step
