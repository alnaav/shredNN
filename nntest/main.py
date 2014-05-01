from nn.activations import IdActivation, LogisticActivation
from nn.layers import FullyConnectedLayer
from nn.network import NN
import numpy as np

nn = NN()

nn.add_layer(FullyConnectedLayer(5, 300, activation=IdActivation()))
nn.add_layer(FullyConnectedLayer(300, 2, activation=LogisticActivation()))

out = nn.apply(np.random.random(5))
print(out)
