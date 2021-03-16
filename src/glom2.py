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
import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .utils import is_tensor_type
from .backend.ops import geometric_weighted_mean, get_local_lateral, get_global_lateral

# anything that can directly initialize tf.zeros(.) is acceptable
Shape3D = Union[Tuple[int, int, int], tf.TensorShape, tf.Tensor]


class GLOMCell(tfkl.AbstractRNNCell):
    HPARAM_DEFAULTS = dict(
        a_td=0.,  # top down weight multiply
        a_lat=0.,  # lateral weight multiply
        a_bu=0.,  # bottom up weight multiply
        b_td=0.,  # top down weight shift
        b_lat=0.,  # lateral weight shift
        b_bu=0.,  # bottom up weight shift
        sparsity=0.2,  # activation sparsity
        lr_awake=0.005,  # learning rate when awake
        lr_asleep=0.02,  # learning rate when asleep
        epsilon_control=1e-3,  # prevents ln(0) for td attn weight
        window_size=(5, 5),  # x_loc window
        roll_over=True,  # x_loc connect edges
        global_sparsity=0.1,  # x_global sparsity
    )

    CONN_KS = dict(
        fn='fn',  # if None (default) a neural network is made and stored by GLOMCell
        inputs='inputs',  # list, tuple, or single string
        outputs='outputs',  # list, tuple, or single string
        type='type',  # 'bu' | 'td' | 'lat'
    )

    def __init__(self,
                 input_layers: List[Text],
                 output_layers: List[Text],
                 layer_sizes: Mapping[Text, Shape3D],
                 connections: List[Mapping],
                 asleep_optimizer: Optional[tf.optimizers.Optimizer] = None,
                 awake_optimizer: Optional[tf.optimizers.Optimizer] = None,
                 hparams: Optional[Mapping[Text, object]] = None,
                 name: Optional[Text] = None):

        super(GLOMCell, self).__init__(name=name)

        self._mode = 'awake'  # 'awake' | 'asleep'

        self.input_layers = input_layers
        self.output_layers = output_layers
        self.layer_sizes = layer_sizes
        self.connections = connections
        self.hparams: dict = GLOMCell.HPARAM_DEFAULTS.copy().update(hparams)

        if awake_optimizer is None:
            awake_optimizer = tf.optimizers.SGD(self.hparams['lr_awake'])
        if asleep_optimizer is None:
            asleep_optimizer = tf.optimizers.SGD(self.hparams['lr_asleep'])

        self.optimizers = dict(awake_optimizer=awake_optimizer,
                               asleep_optimizer=asleep_optimizer)

        # clean self.connections
        for i in range(len(self.connections)):
            # allow unit (non list) connections
            if not isinstance(connections[i]['inputs'], (list, tuple)):
                self.connections[i]['inputs'] = [connections[i]['inputs']]
            if not isinstance(connections[i]['outputs'], (list, tuple)):
                self.connections[i]['outputs'] = [connections[i]['outputs']]
            if 'fn' not in connections[i]:
                connections[i]['fn'] = None

            # for output in self.connections[i]['outputs']:
            #    self.layer_params[output]['nun_inputs'] += 1

        self.call_fns = {
            ('awake', True): self._call_awake_training,
            ('awake', False): self._call_awake_not_training,
            ('asleep', True): self._call_asleep_training,
            ('asleep', False): self._call_asleep_not_training,
        }

    def build(self, input_shape):
        for i, connection in enumerate(self.connections):
            if connection['fn'] is None:
                self.connections[i]['fn'] = DenseND(
                    input_shape=self.layer_sizes[connection['inputs'][0]],
                    output_shape=self.layer_sizes[connection['outputs'][0]],
                    sparsity=self.hparams['sparsity'])

    def call(self, inputs: Tuple[dict, dict], training=None, mask=None):
        observations, layer_states_flat = inputs
        if training is None: training = False
        layer_states = tf.nest.pack_sequence_as(self.layer_sizes, layer_states_flat)

        # randomize order to simulate true asychronous updating
        connections = self.connections
        random.shuffle(connections)

        # run appropriate function
        call_fn = self.call_fns[(self._mode, training)]
        new_layer_states, grads_and_vars = call_fn(layer_states)

        # maybe apply gradients
        if training:
            optimizer = self.optimizers[self._mode]
            optimizer.apply_gradients(grads_and_vars)

        # return output values
        return {k: new_layer_states[k]['x'] for k in self.output_layers}, \
               tf.nest.flatten(new_layer_states)

    def _call_awake_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
            -> Tuple[Mapping[Text, Mapping[Text, tf.Tensor]], List[Tuple[tf.Tensor, tf.Variable]]]:

        new_errs = {layer: list() for layer in self.layer_sizes.keys()}
        layer_targets = {layer: list() for layer in self.layer_sizes.keys()}
        new_layer_states = dict()
        grads_and_vars = []

        # compute targets for all layers and backpropagate errors
        for connection in self.connections:
            # get inputs
            input_vals = [layer_states[layer]['x'] for layer in connection['inputs']]

            # forward propagation
            with tf.GradientTape() as tape:
                # I think I have to explicitly watch input_xs
                # since they are `Tensors` and not-necesarily trainale
                tape.watch(input_vals)
                output_vals = connection['fn'](input_vals)
                ssl_loss = sum(connection['fn'].losses)

            output_vals = tf.nest.flatten(output_vals)  # ensure is list. x -> [x]

            # difference-based saliency
            ws = [(val - layer_states[layer]['x']) ** 2
                  for layer, val in zip(connection['outputs'], output_vals)]

            # apply hyper-parameters
            conn_type = connection['type']  # 'bu' or 'td'. Must match dictionary key exactly
            ws = [self.hparams[f'a_{conn_type}'] * w + self.hparams[f'b_{conn_type}'] for w in ws]

            # assign output vals
            for layer, w, output in zip(connection['outputs'], ws, output_vals):
                layer_targets[layer].append((w, output))

            # backpropagate errors
            input_grads, weight_grads = tape.gradient(
                target=output_vals,
                sources=(input_vals, connection['fn'].trainable_weights),
                output_gradients=[layer_states[layer]['e'] for layer in connection['outputs']])

            #### TODO: include ssl_loss in the gradient calculation

            # backpropagate errors top down
            for layer, input_grad in zip(connection['inputs'], input_grads):
                new_errs[layer].append(input_grad)

            # store parameter gradients
            grads_and_vars.extend([(grads, var) for grads, var
                                   in zip(weight_grads, connection['fn'].trainable_weights)])

        # compute lateral self-attention
        for layer in layer_states.keys():
            x = layer_states[layer]['x']
            tf.assert_rank(x, 4, 'inputs should be four dimensional [B, X, Y, D]')

            x_local = get_local_lateral(x=x, window_size=self.hparams['window_size'],
                                        roll_over=self.hparams['roll_over'])
            x_global = get_global_lateral(x=x, global_sparsity=self.hparams['global_sparsity'])
            x_neighbor = tf.concat([x_local, x_global], axis=-2)

            # compute similarity scores; assuming values are normal, divide by sqrt(num depth dimensions)
            similarity = tf.einsum('...d,...id->...i', x, x_neighbor) / (similarity.shape[-1]**0.5)
            layer_targets[layer].append((similarity, x))

            # NOTE There are no trainable parameters so this spatial contrastive objective does nothing!
            # sim_local = tf.einsum('...d,...id', x, x_local)
            # sim_global = tf.einsum('...d,...id', x, x_global)
            # sim_cat = tf.concat([sim_local, sim_global], axis=-2)
            # self.add_loss(tf.reduce_mean(sim_global) - tf.reduce_mean(sim_local))

        # apply targets
        for layer, targets in layer_targets.items():
            new_layer_states[layer]['x'] = geometric_weighted_mean(
                xs=[x for x, w in targets], ws=[w for x, w in targets])

        # update errors
        for layer, state in new_layer_states.keys():
            new_errs[layer].append(new_layer_states[layer]['x'] - layer_states[layer]['x'])
            new_layer_states[layer]['e'] = sum(new_errs[layer]) / len(new_errs[layer])
            new_layer_states[layer]['e_norm'] = tf.reduce_sum(new_layer_states[layer]['e'] ** 2, axis=-1)

        return new_layer_states, None

    def _call_awake_not_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
            -> Tuple[Mapping[Text, Mapping[Text, tf.Tensor]], List[Tuple[tf.Tensor, tf.Variable]]]:
        pass

    def _call_asleep_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
            -> Tuple[Mapping[Text, Mapping[Text, tf.Tensor]], List[Tuple[tf.Tensor, tf.Variable]]]:
        pass

    def _call_asleep_not_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
            -> Tuple[Mapping[Text, Mapping[Text, tf.Tensor]], List[Tuple[tf.Tensor, tf.Variable]]]:
        pass

    @property
    def state_size(self):
        initial_state_sizes = {layer: {
            'x': size,
            'e': size,
            'e_norm': size[:-1],
        } for layer, size in self.layer_sizes.items()}
        return tf.nest.flatten(initial_state_sizes)

    @property
    def output_size(self):
        return [self.layer_sizes[layer] for layer in self.output_layers]

    @property
    def get_mode(self):
        return self._mode

    def set_mode(self, mode):
        self._mode = mode


class ConcatTransformSplit(tfkl.Layer):

    def __init__(self,
                 transform_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 concat_axis: Optional[int] = -2,
                 num_or_size_splits: Optional[List[int]] = None,
                 split_axis: Optional[int] = -2,
                 name: Optional[Text] = None):
        super(ConcatTransformSplit, self).__init__(name=name)

        if transform_fn is None:
            transform_fn = (lambda x: x)

        self.transform_fn = transform_fn
        self.concat_axis = concat_axis
        self.num_or_size_splits = num_or_size_splits
        self.split_axis = split_axis

        if self.split_sizes is None:
            self.do_split = False
        else:
            self.do_split = True

    def build(self, input_shape):
        if self.do_split:
            self.split_layer = tfkl.Lambda(lambda x: tf.split(x,
                                                              num_or_size_splits=self.num_or_size_splits,
                                                              axis=self.split_axis))

    def call(self, inputs, **kwargs):
        inputs = tf.nest.flatten(inputs)  # ensure inputs is a `list`
        x_cat = tfkl.concatenate(inputs, axis=self.concat_axis)
        x_transformed = self.transform_fn(x_cat)
        if self.do_split:
            return self.split_layer(x_transformed)
        else:
            return x_transformed

    def map_inputs(self, inputs):
        return self.call(inputs)


class DenseND(tfkl.Layer):

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

        self.W1 = self.add_weight(name=self.name + '_W1', shape=locs + (d1, d2))
        self.b1 = self.add_weight(name=self.name + '_b1', shape=locs + (d2,))
        self.W2 = self.add_weight(name=self.name + '_W2', shape=locs + (d2, d3))
        self.b2 = self.add_weight(name=self.name + '_b2', shape=locs + (d3,))

        if isinstance(self.activation, str):
            self.activation = tfkl.Activation(self.activation)
        assert isinstance(self.activation, Callable), 'activation must be callable or valid keras activation'

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nest.flatten(inputs)  # ensure inputs is a `list`
        x1 = inputs[0]

        x2 = self.activation(tf.einsum('...a,...ab->...b', x1, self.W1) + self.b1)
        x3 = self.activation(tf.einsum('...a,...ab->...b', x2, self.W2) + self.b2)

        # some recent arxiv paper suggested this over x**2 penality. Not sure if it works
        self.add_loss(tf.reduce_sum(tf.exp(x3) / tf.reduce_sum(x3, axis=-1)))
        self.add_loss((tf.reduce_mean(x3) - self.sparsity) ** 2)

        return x3
