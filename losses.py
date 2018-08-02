import numpy as np


class CrossEntropyError:
    def calc(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y)) / batch_size
