import numpy as np

from mine.backend.neuralnet.utils import *


class HeInitializer:
    @staticmethod
    def init_weights(input_size, output_size, hidden_size_list, use_batch_norm=True):
        weights = {}
        all_size_list = [input_size] + hidden_size_list + [output_size]
        list_len = len(all_size_list)
        for i in range(1, list_len):
            std = np.sqrt(2.0 / all_size_list[i - 1])
            weights[key_W(i)] = std * np.random.randn(all_size_list[i - 1], all_size_list[i])
            weights[key_b(i)] = np.zeros(all_size_list[i])
            if use_batch_norm and (i != list_len - 1):
                weights[key_gamma(i)] = np.ones(all_size_list[i])
                weights[key_beta(i)] = np.zeros(all_size_list[i])

        return weights
