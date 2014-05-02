from abc import ABCMeta, abstractmethod


class Trainer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, features, target):
        pass
