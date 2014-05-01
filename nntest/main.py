import numpy as np

from nn.activations import IdActivation, LogisticActivation
from nn.layers import FullyConnectedLayer
from nn.network import NN


nn = NN(5)

nn.add_layer(FullyConnectedLayer(300, activation=IdActivation()))
nn.add_layer(FullyConnectedLayer(2, activation=LogisticActivation()))

out = nn.apply(np.random.random(5))
print(out)
