from abc import ABCMeta, abstractmethod


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def connect(self, prev_layer):
        pass

    @abstractmethod
    def apply(self, x):
        pass

