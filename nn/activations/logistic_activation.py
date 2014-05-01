from nn.activations.activation import Activation
import math
import numpy as np


class LogisticActivation(Activation):
    def __init__(self):
        def f(x):
            return 1 / (1 + math.exp(-x))

        def df(x):
            fx = f(x)
            return fx * (1 - fx)

        self.vec_f = np.vectorize(f)
        self.vec_df = np.vectorize(df)

    def apply(self, vector):
        return self.vec_f(vector)

    def apply_derivative(self, vector):
        return self.vec_df(vector)
