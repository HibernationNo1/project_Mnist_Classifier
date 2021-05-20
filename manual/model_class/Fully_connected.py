import tensorflow as tf

from AffineFunction import Layer
from tensorflow.feras.layers import Activation

class Dense:
    def __init__(self, feature_dim, units):
        self._feature_dim = feature_dim

        self._affine= Layer(self._feature_dim, units)
        self._relu = Activation('relu')
        self._softmax = Activation('softmax') 

    def relu(self, x):
        z = self._affine.layer(x)
        pred = self._relu(z)
        return pred

    def softmax(self, x):
        z = self._affine.layer(x)
        pred = self._softmax(z)
        return pred

class Flatten:
    def __init__(self, x):
        self._re = x.shape[1]*x.shape[2]*x.shape[3]
        self._x = x.reshape([-1, self._re])
        return x