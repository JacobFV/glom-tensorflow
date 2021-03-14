#  MIT License
#
#  Copyright (c) 2021 Jacob Valdez
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

from typing import List, Tuple, Mapping, Text, Union, Optional, Callable

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .backend.ops import geometric_weighted_mean, get_local_lateral, get_global_lateral

# anything that can directly initialize tf.zeros(.) is acceptable
Shape3D = Union[Tuple[int, int, int], tf.TensorShape, tf.Tensor]


class GLOMRNNCell(tfkl.AbstractRNNCell):

    HPARAM_DEFAULTS = dict(
        b_td=0.,
        b_lat=0.,
        b_bu=0.,
        sparsity=0.2,
    )

    def __init__(self,
                 input_layers: List[Text],
                 output_layers: List[Text],
                 layer_sizes: Mapping[Text, Shape3D],
                 connections: List[Tuple[Text, Text]],
                 hparams: Optional[Mapping[Text, object]] = None,
                 name: Optional[Text] = None):

        super(GLOMRNNCell, self).__init__(name=name)

        self.input_layers = input_layers
        self.output_layers = output_layers
        self.layer_sizes = layer_sizes
        self.connections = {(src, dst): None for src, dst in connections}
        self.hparams = GLOMRNNCell.HPARAM_DEFAULTS.copy().update(hparams)

        self.call_fns = {
            ('awake', True): self._call_awake_training,
            ('awake', False): self._call_awake_not_training,
            ('asleep', True): self._call_asleep_training,
            ('asleep', False): self._call_asleep_not_training,
        }

    def build(self, input_shape):
        # for name, shape in input_shape:
        #    assert self.input_layers[name] == shape, 'input shape does not match specified input_layers shape'

        for (src, dst), _ in self.connections.copy().items():
            self.connections[(src, dst)] = DenseND(
                input_shape=self.layer_sizes[src],
                output_shape=self.layer_sizes[dst],
                sparsity=self.hparams['sparsity'])

    def call(self, inputs: Tuple[dict, dict], training=None, mask=None):
        observations, layer_states_flat = inputs
        if training is None: training = False
        layer_states = tf.nest.pack_sequence_as(self.layer_sizes, layer_states_flat)

        # update layer_states with input
        for k, v in observations.items():
            layer_states[k] = v

        # run appropriate function
        call_fn = self.call_fns[(self.mode, training)]
        layer_states = call_fn(layer_states)

        # return output values
        return {k: layer_states[k]['x'] for k in self.output_layers}, tf.nest.flatten(layer_states)

    def _call_awake_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
        -> Mapping[Text, Mapping[Text, tf.Tensor]]:
        pass

    def _call_awake_not_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
        -> Mapping[Text, Mapping[Text, tf.Tensor]]:
        pass

    def _call_asleep_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
        -> Mapping[Text, Mapping[Text, tf.Tensor]]:
        pass

    def _call_asleep_not_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
        -> Mapping[Text, Mapping[Text, tf.Tensor]]:
        pass

    @property
    def state_size(self):
        return tf.nest.flatten(self.layer_sizes)


class DenseND(keras.Model):

    def __init__(self, input_shape, output_shape, sparsity=0.1, activation='relu', name=None):
        super(DenseND, self).__init__(name=name)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.sparsity = sparsity
        self.activation = activation

    def build(self, input_shape):
        assert input_shape[-3:] == self._input_shape  # allow for a batch dimesnion
        locs = self._input_shape[:-1]
        d1 = self._input_shape[-1]
        d2 = (self._input_shape[-1] + self._output_shape[-1]) / 2
        d3 = self.output_shape[-1]

        self.W1 = self.add_weight(name=self.name+'_W1', shape=locs+(d1, d2))
        self.b1 = self.add_weight(name=self.name+'_b1', shape=locs+(d2,))
        self.W2 = self.add_weight(name=self.name+'_W2', shape=locs+(d2, d3))
        self.b2 = self.add_weight(name=self.name+'_b2', shape=locs+(d3,))

        if isinstance(self.activation, str):
            self.activation = tfkl.Activation(self.activation)
        assert isinstance(self.activation, Callable), 'activation must be callable or valid keras activation'

    def call(self, inputs, training=None, mask=None):

        x1 = inputs
        x2 = self.activation(tf.einsum('...a,...ab->...b', x1, self.W1) + self.b1)
        x3 = self.activation(tf.einsum('...a,...ab->...b', x2, self.W2) + self.b2)

        # some recent arxiv paper suggested this over x**2 penality. Not sure if it works
        self.add_loss(tf.reduce_sum(tf.exp(x3)/tf.reduce_sum(x3, axis=-1)))
        self.add_loss((tf.reduce_mean(x3) - self.sparsity)**2)

        return x3