from abc import ABCMeta, abstractmethod


class Activation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, vector):
        pass

    @abstractmethod
    def apply_derivative(self, vector):
        pass