from abc import ABCMeta, abstractmethod


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, vector):
        pass

    @abstractmethod
    def calc_error(self, delta):
        pass

    @abstractmethod
    def calc_prev_error(self, delta):
        pass

    @abstractmethod
    def require_connect(self, prev_layer):
        pass
