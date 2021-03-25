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


class GLOM(keras.Model):
    HPARAM_DEFAULTS = dict(
        a_td=1.,  # top down weight multiply
        a_lat=1.,  # lateral weight multiply
        a_bu=1.,  # bottom up weight multiply
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
        connection_activation='relu',  # for interlayer influence
        # run_steps=4,  # number of steps to run before returning output
        clip_value_min=1e-2,
        clip_value_max=10.,
        sparsity_L1_penalty=0.1,

    )

    CONN_KS = dict(
        fn='fn',  # if None (default) a neural network is made and stored by GLOMRNNCell
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

        super(GLOM, self).__init__(name=name)

        self._mode = 'awake'  # 'awake' | 'asleep'

        self.input_layers = input_layers
        self.output_layers = output_layers
        self.layer_sizes = {k: tf.TensorShape(v) for k, v in layer_sizes.items()}
        self.connections = connections
        if hparams is None:
            hparams = dict()
        self.hparams: dict = GLOM.HPARAM_DEFAULTS.copy()
        self.hparams.update(hparams)

        if awake_optimizer is None:
            awake_optimizer = tf.optimizers.SGD(self.hparams['lr_awake'])
        if asleep_optimizer is None:
            asleep_optimizer = tf.optimizers.SGD(self.hparams['lr_asleep'])

        self.optimizers = dict(awake=awake_optimizer,
                               asleep=asleep_optimizer)

        # clean self.connections
        for i in range(len(self.connections)):
            # allow unit (non list) connections
            if not isinstance(connections[i]['inputs'], (list, tuple)):
                self.connections[i]['inputs'] = [connections[i]['inputs']]
            if not isinstance(connections[i]['outputs'], (list, tuple)):
                self.connections[i]['outputs'] = [connections[i]['outputs']]
            if 'fn' not in connections[i]:
                connections[i]['fn'] = None

        self.call_fns = {
            ('awake', True): self._call_awake_training,
            ('awake', False): self._call_awake_not_training,
            ('asleep', True): self._call_asleep_training,
            ('asleep', False): self._call_asleep_not_training,
        }

    def build(self, input_shape):
        for i, connection in enumerate(self.connections):
            if connection['fn'] is None:
                self.connections[i]['fn'] = ManyToManyDense(
                    input_layer_shapes=[(layer, self.layer_sizes[layer])
                                        for layer in connection['inputs']],
                    output_layer_shapes=[(layer, self.layer_sizes[layer])
                                         for layer in connection['outputs']],
                    activation=self.hparams['connection_activation'],
                    sparsity=self.hparams['sparsity'],
                    concat_axis=-2, split_axis=-2,
                    name=f'{".".join(connection["inputs"])}-{".".join(connection["outputs"])}'
                )

    def call(self, layer_states: Mapping[Text, Mapping], training=None, mask=None):
        training = True  # QT workaround "call() got multiple values for argument 'training'"
        if training is None:
            training = False

        grads_and_vars = list()
        call_fn = self.call_fns[(self._mode, training)]

        # run appropriate function
        layer_states, new_grads_and_vars = call_fn(layer_states)
        grads_and_vars.extend(new_grads_and_vars)

        # maybe apply gradients
        if training:
            # THIS IS NOT NEEDED SINCE TF.OPTIMIZER CONSOLIDATES GRADIENTS AUTOMATICALLY
            ## consolidate grads and vars
            # vars = set(v for g, v in grads_and_vars)
            # grads_and_vars_dict = {v: [] for v in vars}
            # for g, v in grads_and_vars:
            #    grads_and_vars_dict[g].append(v)
            # for g, _ in grads_and_vars:
            #    grads_and_vars_dict[g] = tf.concat(grads_and_vars_dict[g], axis=0)

            optimizer = self.optimizers[self._mode]
            optimizer.apply_gradients(grads_and_vars)

        layer_states = {
            layer: dict(
                x=tf.clip_by_value(state['x'],
                                   clip_value_min=self.hparams['clip_value_min'],
                                   clip_value_max=self.hparams['clip_value_max']),
                e=tf.clip_by_value(state['e'],
                                   clip_value_min=self.hparams['clip_value_min'],
                                   clip_value_max=self.hparams['clip_value_max']),
                e_norm=tf.clip_by_value(state['e_norm'],
                                        clip_value_min=self.hparams['clip_value_min'],
                                        clip_value_max=self.hparams['clip_value_max'])
            ) for layer, state in layer_states.items()
        }

        # return next layer states  # output values
        return layer_states  # {k: layer_states[k]['x'] for k in self.output_layers}

    def _call_awake_training(self, layer_states: Mapping[Text, Mapping[Text, tf.Tensor]]) \
            -> Tuple[Mapping[Text, Mapping[Text, tf.Tensor]], List[Tuple[tf.Tensor, tf.Variable]]]:

        new_errs = {layer: list() for layer in self.layer_sizes.keys()}
        layer_targets = {layer: list() for layer in self.layer_sizes.keys()}
        new_layer_states = {layer: dict(x=None, e=None, e_norm=None)
                            for layer in self.layer_sizes.keys()}
        grads_and_vars = []

        # compute targets for all layers and backpropagate errors
        for connection in self.connections:
            # get inputs
            input_vals = [layer_states[layer]['x'] for layer in connection['inputs']]

            # forward propagation
            with tf.GradientTape() as tape:
                tape.watch(input_vals)
                output_vals = connection['fn'](input_vals)
                ssl_loss = sum(connection['fn'].losses)

            output_vals = tf.nest.flatten(output_vals)  # ensure is list. x -> [x]

            # TODO If I want contrastive spatial representation, add this to the gradients targets
            # get x_local, x_global from output_val for each output_val in output_vals
            # sim_local = tf.einsum('...d,...id', x, x_local)
            # sim_global = tf.einsum('...d,...id', x, x_global)
            # sim_cat = tf.concat([sim_local, sim_global], axis=-2)
            # self.add_loss(tf.reduce_mean(sim_global) - tf.reduce_mean(sim_local))

            # difference-based saliency
            ws = [tf.norm(val - layer_states[layer]['x'], ord=2, axis=-1)
                  for layer, val in zip(connection['outputs'], output_vals)]

            # apply hyper-parameters
            conn_type = connection['type']  # 'bu' or 'td'. Must match dictionary key exactly
            ws = [self.hparams[f'a_{conn_type}'] * w + self.hparams[f'b_{conn_type}'] for w in ws]

            # assign output vals
            for layer, w, output in zip(connection['outputs'], ws, output_vals):
                layer_targets[layer].append((w[..., None], output[..., None]))

            # backpropagate errors
            input_grads, weight_grads = tape.gradient(
                target=(output_vals, ssl_loss),
                sources=(input_vals, connection['fn'].trainable_weights),
                output_gradients=[layer_states[layer]['e'] for layer in connection['outputs']])

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
            x_neighbor = tf.einsum('...id->...di', x_neighbor)

            # compute similarity scores
            similarity = tf.einsum('...d,...di->...i', x, x_neighbor)
            # divide by x**2 to make self similarity = 1
            self_similarity = tf.einsum('...d,...d->...', x, x)
            similarity = similarity / (self_similarity[..., None] + 1e-2)
            similarity = self.hparams['a_lat'] * similarity + self.hparams['b_lat']
            layer_targets[layer].append((similarity, x_neighbor))

        # apply targets
        for layer, targets in layer_targets.items():
            new_layer_states[layer]['x'] = geometric_weighted_mean(
                xs=[x for w, x in targets], ws=[w for w, x in targets])

        # update errors
        for layer in new_layer_states.keys():
            with tf.name_scope(f'{layer}_update'):
                new_errs[layer].append(new_layer_states[layer]['x'] - layer_states[layer]['x'])
                new_layer_states[layer]['e'] = sum(new_errs[layer]) / len(new_errs[layer])
                new_layer_states[layer]['e_norm'] = tf.norm(new_layer_states[layer]['e'], ord=2, axis=-1) / \
                    new_layer_states[layer]['e'].shape[-1]  # assuming binary RV's

        return new_layer_states, grads_and_vars

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
        return initial_state_sizes  # tf.nest.flatten(initial_state_sizes)

    def get_initial_state(self, batch_size=1):
        return tf.nest.map_structure(lambda shape: tf.random.uniform(shape=[batch_size] + shape, minval=0, maxval=0.1),
                                     self.state_size)

    # @property
    # def output_size(self):
    #     return [self.layer_sizes[layer] for layer in self.output_layers]

    @property
    def get_mode(self):
        return self._mode

    def set_mode(self, mode):
        self._mode = mode


class ManyToManyDense(tfkl.Layer):

    def __init__(self,
                 input_layer_shapes: List[Tuple[Text, Tuple]],
                 output_layer_shapes: List[Tuple[Text, Tuple]],
                 activation: Union[Text, Callable] = 'relu',
                 sparsity: float = 0.1,
                 concat_axis: int = -2,
                 split_axis: int = -2,
                 name: Text = None):
        super(ManyToManyDense, self).__init__(name=name)

        self.input_layer_shapes = input_layer_shapes
        self.output_layer_shapes = output_layer_shapes
        self.activation = activation
        self.sparsity = sparsity
        self.concat_axis = concat_axis
        self.split_axis = split_axis

    def build(self, input_shape):
        del input_shape

        combined_input_len = sum(input_layer_shape[1][self.concat_axis]
                                 for input_layer_shape in self.input_layer_shapes)
        input_shape = list(self.input_layer_shapes[0][1])
        input_shape[self.concat_axis] = combined_input_len

        combined_output_len = sum(output_layer_shape[1][self.split_axis]
                                  for output_layer_shape in self.output_layer_shapes)
        output_shape = list(self.output_layer_shapes[0][1])
        output_shape[self.split_axis] = combined_output_len

        self.denseND_layer = _DenseND(
            input_shape=input_shape,
            output_shape=output_shape,
            sparsity=self.sparsity,
            activation=self.activation,
            name=f'{self.name}_denseND'
        )

        self.concat_transform_split = _ConcatTransformSplit(
            transform_fn=self.denseND_layer,
            concat_axis=self.concat_axis,
            num_or_size_splits=[output_layer_shape[1][self.split_axis]
                                for output_layer_shape in self.output_layer_shapes],
            split_axis=self.split_axis,
            name=f'{self.name}_concatsplit'
        )

    def call(self, inputs, **kwargs):
        return self.concat_transform_split(inputs)


class _ConcatTransformSplit(tfkl.Layer):

    def __init__(self,
                 transform_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 concat_axis: Optional[int] = -2,
                 num_or_size_splits: Optional[List[int]] = None,
                 split_axis: Optional[int] = -2,
                 name: Optional[Text] = None):
        super(_ConcatTransformSplit, self).__init__(name=name)

        if transform_fn is None:
            transform_fn = (lambda x: x)

        self.transform_fn = transform_fn
        self.concat_axis = concat_axis
        self.num_or_size_splits = num_or_size_splits
        self.split_axis = split_axis

        if self.num_or_size_splits is None:
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
        if len(inputs) > 1:
            x_cat = tfkl.concatenate(inputs, axis=self.concat_axis)
        else:
            x_cat = inputs[0]
        x_transformed = self.transform_fn(x_cat)
        if self.do_split:
            return self.split_layer(x_transformed)
        else:
            return x_transformed

    def map_inputs(self, inputs):
        return self.call(inputs)


class _DenseND(tfkl.Layer):

    def __init__(self, input_shape, output_shape, sparsity=0.1, activation='relu', name=None):
        super(_DenseND, self).__init__(name=name)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.sparsity = sparsity
        self.activation = activation

    def build(self, input_shape):
        input_shape = tf.nest.flatten(input_shape)[0]
        input_shape_actual = (input_shape[-3:])
        assert input_shape_actual == self._input_shape
        # assert input_shape[-3:] == self._input_shape  # allow for a batch dimesnion
        locs = self._input_shape[:-1]
        d1 = self._input_shape[-1]
        d2 = (self._input_shape[-1] + self._output_shape[-1]) // 2
        d3 = self._output_shape[-1]

        self.W1 = self.add_weight(name=self.name + '_W1', shape=locs + [d1, d2],
                                  initializer=keras.initializers.random_normal(mean=0.0, stddev=(2 / (d1 + d2)) ** 0.5))
        self.b1 = self.add_weight(name=self.name + '_b1', shape=locs + [d2], initializer='zeros')
        self.W2 = self.add_weight(name=self.name + '_W2', shape=locs + [d2, d3],
                                  initializer=keras.initializers.random_normal(mean=0.0, stddev=(2 / (d2 + d3)) ** 0.5))
        self.b2 = self.add_weight(name=self.name + '_b2', shape=locs + [d3], initializer='zeros')

        if isinstance(self.activation, (str, Text)):
            self.activation = tfkl.Activation(self.activation)
        assert isinstance(self.activation, Callable), 'activation must be callable or valid keras activation'

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nest.flatten(inputs)  # ensure inputs is a `list`
        x1 = inputs[0]

        x2 = self.activation(tf.einsum('...a,...ab->...b', x1, self.W1) + self.b1)
        x3 = self.activation(tf.einsum('...a,...ab->...b', x2, self.W2) + self.b2)

        # some recent arxiv paper suggested this over x**2 penality. Not sure if it works
        # sparse_coding_loss = tf.reduce_sum(tf.exp(x3) / (tf.reduce_sum(x3, axis=-1)[..., None] + 1e-2))
        # if tf.reduce_any(tf.math.is_nan(sparse_coding_loss)) or tf.reduce_any(tf.math.is_inf(sparse_coding_loss)):
        #     raise ValueError('Bad number')
        # self.add_loss(sparse_coding_loss)
        self.add_loss(0.05 * tf.norm(x3, ord=1, axis=-1))
        # self.add_loss(tf.reduce_sum(tf.exp(x3) / tf.reduce_sum(x3, axis=-1)))
        self.add_loss(0.05 * (tf.reduce_mean(x3) - self.sparsity) ** 2)

        return x3
