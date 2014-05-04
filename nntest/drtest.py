import numpy as np
import pandas as p
from sklearn import metrics

from nn.activations import LogisticActivation
from nn.layers import FullyConnectedLayer
from nn.network import NN


data = p.read_csv(r'./data/train.csv')
train_set = np.asarray(data)

cv_factor = 1.0 / 10
samples_len = 10000 #train_set.shape[0]
train_len = samples_len * (1 - cv_factor)
cv_len = samples_len - train_len

print samples_len, train_len, cv_len

features = train_set[:train_len, 1:]
target = train_set[:train_len, 0]

cv_features = train_set[train_len:train_len + cv_len, 1:]
cv_target = train_set[train_len:train_len + cv_len, 0]

features = (features - 127.0) / 255.0
cv_features = (cv_features - 127.0) / 255.0

nn = NN(features.shape[1])

nn.add_layer(FullyConnectedLayer(25, activation=LogisticActivation()))
nn.add_layer(FullyConnectedLayer(10, activation=LogisticActivation()))

nn.train(features, target)
y = nn.apply(cv_features)

predicted = np.zeros(cv_target.shape)
for i in range(0, y.shape[0]):
    predicted[i] = np.argmax(y[i, :])

print "Classification report:\n{}\n".format(metrics.classification_report(cv_target, predicted))
print "Confusion matrix:\n{}".format(metrics.confusion_matrix(cv_target, predicted))
