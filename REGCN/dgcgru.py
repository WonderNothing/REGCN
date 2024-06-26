import numpy as np
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from tensorflow.keras.layers import AbstractRNNCell
from utils import calculate_laplacian
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, GRUCell, CuDNNLSTM, BatchNormalization, RNN, TimeDistributed
from tensorflow.keras.constraints import MinMaxNorm, UnitNorm

class gcgru(AbstractRNNCell):

    # def __init__(self, num_units, adj1,adj2, num_gcn_nodes, s_index, **kwargs):
    def __init__(self, num_units, adj, num_gcn_nodes, s_index, **kwargs):
        super(gcgru, self).__init__(**kwargs)
        self.units = num_units
        self._gcn_nodes = num_gcn_nodes
        self.s_index = s_index

        self._adj = adj

    @ property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # weights
        self.wz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wz')
        self.wr = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wr')
        self.wh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wh')

        self.w0 = self.add_weight(shape=(1, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='w0')
        self.wa = self.add_weight(shape=(self._adj.shape[0],self._gcn_nodes,self._gcn_nodes),
                                  initializer='random_normal',
                                  trainable=True,
                                  constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                                  # constraint= UnitNorm(axis=0),
                                  name='wa')
        # us
        self.uz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wz')
        self.ur = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='ur')
        self.uh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='uh')

        # biases
        self.bz = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bz")
        self.br = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="br")
        self.bh = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bh")
        self.built = True

    def call(self, inputs, states):
        state = states[0]
        adj = self._adj
        integrated_inputs = tf.math.multiply(self.wa, adj)
        adj_max = tf.reduce_sum(integrated_inputs, axis=0)

        #GCN
        x = self.gc(inputs, adj_max)
        #GRU
        z = K.dot(x, self.wz) + K.dot(x, self.uz) + self.bz
        z = sigmoid(z)
        r = K.dot(x, self.wr) + K.dot(x, self.ur) + self.br
        r = sigmoid(r)
        h = K.dot(x, self.wh) + K.dot((r * state), self.uh) + self.bh
        h = tanh(h)

        output = z * state + (1 - z) * h
        return output, output

    def gc(self, inputs, adj):
        ax = K.dot(inputs, adj)
        ax = ax[:, self.s_index]
        ax = tf.expand_dims(ax, -1)
        return K.dot(ax, self.w0)


