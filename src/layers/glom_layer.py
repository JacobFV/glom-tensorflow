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

# MIT License
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#
from typing import Text, Mapping, Optional, Tuple, List, Union

import tensorflow as tf
import tensorflow.keras as keras

from ..backend.ops import geometric_weighted_mean, get_lateral
from .multidirectional_layer import MultiDirectionalLayer


class GLOMLayer(MultiDirectionalLayer):
    hparam_defaults = {
        'window_size': (7, 7),
        'global_sparsity': 0.05,
        'normalize_beta': 0.5,
    }

    forward_param_defaults = {
        'mode': 'awake',  # 'awake'|'asleep'
        'training': False
    }

    def __init__(self,
                 f_bu: keras.Model,
                 f_td: keras.Model,
                 hparams: Optional[Mapping] = None,
                 name: Optional[Text] = None):

        super(GLOMLayer, self).__init__(name=name)

        self.f_bu = f_bu
        self.f_td = f_td

        if hparams is None:
            hparams = dict()
        self.hparams = GLOMLayer.hparam_defaults.copy().update(hparams)

    def build(self, bu_shapes: List[tf.TensorShape], **kwargs):
        assert len(bu_shapes) == 1, 'GLOMLayer can only accept one incoming layer'
        self.shape = bu_shapes[0][0:2]

    def get_initial_state_shape(self, **kwargs) -> Mapping[Text, tf.TensorShape]:
        """Build internal layer weights with shape information. Shapes do not include batch dimension.

        :param kwargs: dictionary containing:
            * 'mode': 'awake' or 'asleep'
            * 'training': True or False
        :return: dictionary of `TensorShape`'s.
        """
        mode = kwargs['mode']
        training = kwargs['training']
        initial_state_shape = dict()

        if mode == 'awake' and training:
            initial_state_shape = dict(
                x=self.shape,
                sigma2_g=self.shape[0:1],
                # TODO
            )
        elif mode == 'awake' and not training:
            return dict()
        elif mode == 'asleep' and training:
            return dict()
        elif mode == 'asleep' and not training:
            return dict()
        else:
            raise AttributeError('`mode` not \'awake\' or \'asleep\'')
        return initial_state_shape

    def forward(self,
                prev_bu_vars: List[Mapping[Text, object]],
                prev_self_vars: Mapping[Text, object],
                prev_td_vars: List[Mapping[Text, object]],
                **kwargs) \
            -> Tuple[List[Mapping[Text, object]],
                     Mapping[Text, object],
                     List[Mapping[Text, object]],
                     List[Tuple[tf.Tensor, tf.Variable]]]:
        """Forward pass for multidirectional layer. Called by `GLOMRNNCell`.

        :param prev_bu_vars: bottom up vars output by layer(s) below at previous timestep
        :param prev_self_vars: vars output by self at previous timestep
        :param prev_td_vars: top down vars output by layer(s) above at previous timestep
        :param kwargs:
        :return: (next_bu_vars, next_self_vars, next_td_vars, grads_and_vars) tuple.
            `next_bu_vars` are supplied to the layer(s) above and `next_td_vars` are
            supplied to the layer(s) below. `grads_and_vars` are processed by the optimizer
            and may be an empty list.
        """
        params: dict = self.forward_param_defaults.copy().update(kwargs)
        call_fns = {
            ('awake', True): self._forward_awake_training,
            ('awake', False): self._forward_awake_not_training,
            ('asleep', True): self._forward_asleep_training,
            ('asleep', False): self._forward_asleep_not_training,
        }
        call_fn = call_fns[(params['mode'], params['training'])]
        return call_fn(prev_bu_vars, prev_self_vars, prev_td_vars)

    def _forward_awake_training(self,
                                prev_bu_vars: List[Mapping[Text, object]],
                                prev_self_vars: Mapping[Text, object],
                                prev_td_vars: List[Mapping[Text, object]],
                                **kwargs) \
            -> Tuple[List[Mapping[Text, object]],
                     Mapping[Text, object],
                     List[Mapping[Text, object]],
                     List[Tuple[tf.Tensor, tf.Variable]]]:
        """Forward pass for multidirectional layer. Called by `GLOMRNNCell`.

        :param prev_bu_vars: bottom up vars output by layer(s) below at previous timestep
        :param prev_self_vars: vars output by self at previous timestep
        :param prev_td_vars: top down vars output by layer(s) above at previous timestep
        :param kwargs:
        :return: (next_bu_vars, next_self_vars, next_td_vars, grads_and_vars) tuple.
            `next_bu_vars` are supplied to the layer(s) above and `next_td_vars` are
            supplied to the layer(s) below. `grads_and_vars` are processed by the optimizer
            and may be an empty list.
        """

    def _forward_awake_not_training(self,
                                    prev_bu_vars: List[Mapping[Text, object]],
                                    prev_self_vars: Mapping[Text, object],
                                    prev_td_vars: List[Mapping[Text, object]],
                                    **kwargs) \
            -> Tuple[List[Mapping[Text, object]],
                     Mapping[Text, object],
                     List[Mapping[Text, object]],
                     List[Tuple[tf.Tensor, tf.Variable]]]:
        """Forward pass for multidirectional layer. Called by `GLOMRNNCell`.

        :param prev_bu_vars: bottom up vars output by layer(s) below at previous timestep
        :param prev_self_vars: vars output by self at previous timestep
        :param prev_td_vars: top down vars output by layer(s) above at previous timestep
        :param kwargs:
        :return: (next_bu_vars, next_self_vars, next_td_vars, grads_and_vars) tuple.
            `next_bu_vars` are supplied to the layer(s) above and `next_td_vars` are
            supplied to the layer(s) below. `grads_and_vars` are processed by the optimizer
            and may be an empty list.
        """

    def _forward_asleep_training(self,
                                 prev_bu_vars: List[Mapping[Text, object]],
                                 prev_self_vars: Mapping[Text, object],
                                 prev_td_vars: List[Mapping[Text, object]],
                                 **kwargs) \
            -> Tuple[List[Mapping[Text, object]],
                     Mapping[Text, object],
                     List[Mapping[Text, object]],
                     List[Tuple[tf.Tensor, tf.Variable]]]:
        """Forward pass for multidirectional layer. Called by `GLOMRNNCell`.

        :param prev_bu_vars: bottom up vars output by layer(s) below at previous timestep
        :param prev_self_vars: vars output by self at previous timestep
        :param prev_td_vars: top down vars output by layer(s) above at previous timestep
        :param kwargs:
        :return: (next_bu_vars, next_self_vars, next_td_vars, grads_and_vars) tuple.
            `next_bu_vars` are supplied to the layer(s) above and `next_td_vars` are
            supplied to the layer(s) below. `grads_and_vars` are processed by the optimizer
            and may be an empty list.
        """

    def _forward_asleep_not_training(self,
                                     prev_bu_vars: List[Mapping[Text, object]],
                                     prev_self_vars: Mapping[Text, object],
                                     prev_td_vars: List[Mapping[Text, object]],
                                     **kwargs) \
            -> Tuple[List[Mapping[Text, object]],
                     Mapping[Text, object],
                     List[Mapping[Text, object]],
                     List[Tuple[tf.Tensor, tf.Variable]]]:
        """Forward pass for multidirectional layer. Called by `GLOMRNNCell`.

        :param prev_bu_vars: bottom up vars output by layer(s) below at previous timestep
        :param prev_self_vars: vars output by self at previous timestep
        :param prev_td_vars: top down vars output by layer(s) above at previous timestep
        :param kwargs:
        :return: (next_bu_vars, next_self_vars, next_td_vars) tuple.
            NOTE: `next_bu_vars` are supplied to the layer(s) above and `next_td_vars` are
            supplied to the layer(s) below.
        """

        ###def forward(self, bu_vars, self_vars, td_vars, **kwargs):

        params: dict = self.forward_param_defaults.copy().update(kwargs)

        x_below_prev = bu_vars['x']
        g_bu_prev = bu_vars['g_bu']
        e_below_norm_prev = bu_vars['e_norm']

        x_prev = self_vars['x']
        sigma2_g_prev = self_vars['sigma2_g']

        x_above_prev = bu_vars['x']
        g_td_prev = bu_vars['g_td']
        e_above_norm_prev = bu_vars['e_norm']

        # TODO: I need to use these variables apporpriately
        td_ret_vars = dict()
        self_ret_vars = dict()
        bu_ret_vars = dict()
        grads_and_vars = list()

        if params['mode'] == 'awake':

            ###################### DAYTIME - TRAINING ######################
            if params['training']:
                with tf.GradientTape() as tape:
                    tape.watch([self.f_bu.trainable_variables,
                                self.f_td.trainable_variables,
                                x_below_prev, x_above_prev])
                    x_bu = self.f_bu(x_below_prev, training=True)
                    x_td = self.f_td(x_above_prev, training=True)

                    with tape.stop_recording():
                        x_lat = get_lateral(x=x_prev, window_size=self.hparams['window_size'],
                                            global_sparsity=self.hparams['global_sparsity'])

                    x = geometric_weighted_mean(
                        xs=[x_bu[..., None], x_td[..., None],
                            tf.transpose(x_lat, perm=(0, 1, 2, 4, 3))],  # [B, X, Y, i, D] -> [B, X, Y, D, i]
                        ws=[e_below_norm_prev[..., None, None],  # [B, X, Y] -> [B, X, Y, D=1, i=1] (broadcast over D,i)
                            1. / (e_above_norm_prev[..., None, None] + 1e-2),
                            # same as above. should approach equalibrium.
                            tf.einsum('bxyid,bxyd->bxyi', x_lat, x_prev) / (self._depth ** 0.5)[..., None, :]
                            ])  # lateral weights: [B, X, Y, i] -> [B, X, Y, D=1, i]

                new_g = x - x_prev

                with self.hparams['normalize_beta'] as beta:
                    # I do not want continually shifting representations
                    mu_g = 0  # mu_e = beta * mu_e_prev + (1 - beta) * new_e
                    sigma2_g = beta * sigma2_g_prev + (1 - beta) * (mu_g - new_g) ** 2

                g = new_g / (sigma2_g ** 0.5)
                # I do not want continually shifting representations
                # e = (new_e - mu_e)**2 / (sigma2_e ** 0.5)
                g_norm = tf.norm(g, ord=1, axis=-1) / (g.shape[-1] ** 0.5)

                # train for similarity
                (
                    d_f_bu, d_f_td, g_td, g_bu
                ) = tape.gradient(target=x,
                                  sources=[self.f_bu.trainable_variables,
                                           self.f_td.trainable_variables,
                                           x_below_prev, x_above_prev],
                                  output_gradients=g + g_td_prev + g_bu_prev)

                # assign grads_and_vars
                grads_and_vars = [(d_f_bu, self.f_bu.trainable_variables),
                                  (d_f_td, self.f_td.trainable_variables)]

            ###################### DAYTIME - NOT TRAINING ######################
            else:  # not training

                x_bu = self.f_bu(x_below_prev, training=False)
                x_td = self.f_td(x_above_prev, training=False)

                x_lat = get_lateral(x=x_prev, window_size=self.hparams['window_size'],
                                    global_sparsity=self.hparams['global_sparsity'])

                x = geometric_weighted_mean(
                    xs=[x_bu[..., None], x_td[..., None],
                        tf.transpose(x_lat, perm=(0, 1, 2, 4, 3))],  # [B, X, Y, i, D] -> [B, X, Y, D, i]
                    ws=[e_below_norm_prev[..., None, None],  # [B, X, Y] -> [B, X, Y, D=1, i=1] (broadcast over D,i)
                        1. / (e_above_norm_prev[..., None, None] + 1e-2),  # same as above. should approach equalibrium.
                        tf.einsum('bxyid,bxyd->bxyi', x_lat, x_prev) / (self._depth ** 0.5)[..., None, :]
                        ])  # lateral weights: [B, X, Y, i] -> [B, X, Y, D=1, i]

                new_g = x - x_prev

                with self.hparams['normalize_beta'] as beta:
                    # I do not want continually shifting representations
                    mu_g = 0  # mu_e = beta * mu_e_prev + (1 - beta) * new_e
                    sigma2_g = beta * sigma2_g_prev + (1 - beta) * (mu_g - new_g) ** 2

                g = new_g / (sigma2_g ** 0.5)
                # I do not want continually shifting representations
                # e = (new_e - mu_e)**2 / (sigma2_e ** 0.5)
                g_norm = tf.norm(g, ord=1, axis=-1) / (g.shape[-1] ** 0.5)

        else:  # params['mode'] == 'asleep'
            # no lateral. only top down (but how will the bottom up layers learn)?

            ###################### NIGHTTIME - TRAINING ######################
            if params['training']:
                with tf.GradientTape() as tape:
                    tape.watch([self.f_bu.trainable_variables,
                                self.f_td.trainable_variables,
                                x_below_prev, x_above_prev])
                    x_bu = self.f_bu(x_below_prev, training=True)
                    x_td = self.f_td(x_above_prev, training=True)

                new_g_bu_out = 1 / (x_prev - x_bu)
                new_g_td_out = 1 / (x_prev - x_td)

                with self.hparams['normalize_beta'] as beta:
                    sigma2_g = ((beta * sigma2_g_prev + (1 - beta) * new_g_bu_out ** 2) +
                                (beta * sigma2_g_prev + (1 - beta) * new_g_td_out ** 2)) / 2

                g_td_out = new_g_td_out / (sigma2_g ** 0.5)
                g_bu_out = new_g_td_out / (sigma2_g ** 0.5)

                # train for similarity
                (
                    d_f_bu, d_f_td, g_td, g_bu
                ) = tape.gradient(target=[x_bu, x_td],
                                  sources=[self.f_bu.trainable_variables,
                                           self.f_td.trainable_variables,
                                           x_below_prev, x_above_prev],
                                  output_gradients=[g_bu_out + g_td_prev + g_bu_prev,
                                                    g_td_out + g_td_prev + g_bu_prev])

                # assign grads_and_vars
                grads_and_vars = [(d_f_bu, self.f_bu.trainable_variables),
                                  (d_f_td, self.f_td.trainable_variables)]

                td_ret_vars['g_td'] = g_td
                self_ret_vars['x'] = x_td
                bu_ret_vars['g_bu'] = g_bu

            ###################### NIGHTTIME - NOT TRAINING ######################
            else:  # not training
                x = self.f_td(x_above_prev, training=False)

                self_ret_vars['x'] = x

        # return x, g, g_norm, g_td, g_bu, sigma2_g, grads_and_vars
        return td_ret_vars, self_ret_vars, bu_ret_vars, grads_and_vars
