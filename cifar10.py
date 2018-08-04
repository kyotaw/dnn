import numpy as np
import _pickle


def _unpickle(f):
    fo = open(f, 'rb')
    d = _pickle.load(fo, encoding='latin1')
    fo.close()
    return d

def _one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_cifar(path, flatten=False, one_hot_label=True):
    meta = _unpickle(path + '/batches.meta')
    batch_1 = _unpickle(path + '/data_batch_1')
    label_1 = np.array(batch_1['labels'])
    data_1 = batch_1['data']
    #batch_2 = _unpickle(path + '/data_batch_2')
    #label_2 = np.array(batch_1['labels'])
    #data_2 = batch_1['data']
    #batch_3 = _unpickle(path + '/data_batch_3')
    #label_3 = np.array(batch_1['labels'])
    #data_3 = batch_1['data']
    #batch_4 = _unpickle(path + '/data_batch_4')
    #label_4 = np.array(batch_1['labels'])
    #data_4 = batch_1['data']
    #batch_5 = _unpickle(path + '/data_batch_5')
    #label_5 = np.array(batch_1['labels'])
    #data_5 = batch_1['data']
    test_batch = _unpickle(path + '/test_batch')
    label_test = np.array(test_batch['labels'])
    data_test = test_batch['data']

    if flatten == False:
        data_1 = data_1.reshape([10000, 3, 32, 32])
        #data_2 = data_2.reshape([10000, 3, 32, 32])
        #data_3 = data_3.reshape([10000, 3, 32, 32])
        #data_4 = data_4.reshape([10000, 3, 32, 32])
        #data_5 = data_5.reshape([10000, 3, 32, 32])
        data_test = data_test.reshape([10000, 3, 32, 32])

    if one_hot_label:
        label_1 = _one_hot_label(label_1)
        #label_2 = _one_hot_label(label_2)
        #label_3 = _one_hot_label(label_3)
        #label_4 = _one_hot_label(label_4)
        #label_5 = _one_hot_label(label_5)
        label_test = _one_hot_label(label_test)

    #return (data_1, data_2, data_3, data_4, data_5, data_test), (label_1, label_2, label_3, label_4, label_5, label_test)
    return (data_1, data_test), (label_1, label_test)
