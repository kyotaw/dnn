import math
import numpy as np

from mine.backend.core.log import StdLogger


class Trainer:
    def __init__(
        self,
        network,
        optimizer,
        x_train,
        t_train,
        x_test,
        t_test,
        batch_size,
        epochs):

        self.network = network
        self.optimizer = optimizer
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_size = x_train.shape[0]
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        logger = StdLogger('Trainer')
        logger.log_info('##### Train started #####')
        logger.log_info('Train data size: ' + str(self.train_size))
        logger.log_info('Batch size: ' + str(self.batch_size))
        logger.log_info('Epoch count: ' + str(self.epochs))

        for e in range(self.epochs):
            iteration = max(math.ceil(self.train_size / self.batch_size), 1)
            for i in range(iteration):
                self._train()
                logger.log_info('Epoch: ' + str(e) + '/' + str(self.epochs) + ', Iteration: ' + str(i) + '/' + str(iteration))
            
            train_acc, test_acc = self._evaluate()
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            logger.log_info('##### Epoch: ' + str(e) + ' cpmpleted #####')
            logger.log_info('Train accuracy: ' + str(train_acc))
            logger.log_info('Test accuracy: ' + str(test_acc))

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        logger.log_info('##### Train completed')
        logger.log_info('Final accuracy: ' + str(test_acc))

    def _train(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        self.network.train(x_batch, t_batch, self.optimizer)

    def _evaluate(self):
       train_acc = self.network.accuracy(self.x_train, self.t_train)
       test_acc = self.network.accuracy(self.x_test, self.t_test)
       return train_acc, test_acc
