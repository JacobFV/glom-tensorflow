# MIT License
#
# Copyright (c) 2021 Jacob Valdez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Text, Mapping, List, Optional, Union, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .ops import inject_gradient, geometric_weighted_mean, get_lateral


class GLOMLayer(tfkl.Layer):
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

        super(GLOMLayer, self).__init__(trainable=True, name=name)

        self.f_bu = f_bu
        self.f_td = f_td

        if hparams is None: hparams = dict()
        self.hparams = GLOMLayer.hparam_defaults.copy().update(hparams)

        self._optimizer = keras.optimizers.Adam(1e-3)

    def build(self, input_shape):
        (
            x_prev_below_shape, x_prev_shape, x_prev_above_shape,
            e_prev_below_shape, e_prev_shape, e_prev_above_shape,
            e_below_norm_prev_shape, e_norm_prev_shape, e_above_norm_prev_shape,
            g_prev_below_shape, g_prev_above_shape,
            mu_e_prev_shape, sigma2_e_prev_shape
        ) = input_shape

        # x_prev_shape: [B, X, Y, D]
        self._size = x_prev_shape[1:3]
        self._depth = x_prev_shape[3]

    def call(self, inputs, **kwargs):
        (
            x_prev_below, x_prev, x_prev_above,
            e_prev_below, e_prev, e_prev_above,
            e_below_norm_prev, e_norm_prev, e_above_norm_prev,
            g_prev_below, g_prev_above,
            mu_e_prev, sigma2_e_prev
        ) = inputs

        params: dict = self.forward_param_defaults.copy().update(kwargs)

        if params['training']:
            tape = tf.GradientTape()
            tape.__enter__()
            tape.watch([self.f_bu.trainable_variables,
                        self.f_td.trainable_variables,
                        x_prev_below, x_prev_above])

        if params['mode'] == 'awake':

            x_bu = self.f_bu(x_prev_below)
            x_td = self.f_td(x_prev_above)
            x_lat = get_lateral(x=x_prev, window_size=self.hparams['window_size'],
                                global_sparsity=self.hparams['global_sparsity'])
            x_lat = tf.stop_gradient(x_lat)

            # TODO: use e_norm instead of e
            x = geometric_weighted_mean(
                xs=[x_bu[..., None], x_td[..., None],
                    tf.transpose(x_lat, perm=(0, 1, 2, 4, 3))],  # [B, X, Y, i, D] -> [B, X, Y, D, i]
                ws=[e_below_norm_prev[..., None, None],  # [B, X, Y] -> [B, X, Y, D=1, i=1] (broadcast over D,i)
                    1. / (e_above_norm_prev[..., None, None] + 1e-2),  # same as above. should approach equalibrium.
                    tf.einsum('bxyid,bxyd->bxyi', x_lat, x_prev) / (self._depth ** 0.5)[..., None, :]
                    ]) # lateral weights: [B, X, Y, i] -> [B, X, Y, D=1, i]

            new_e = x - x_prev

            with self.hparams['normalize_beta'] as beta:
                mu_e = beta * mu_e_prev + (1 - beta) * new_e
                sigma2_e = beta * sigma2_e_prev + (1 - beta) * (mu_e - new_e) ** 2

            e = (new_e - mu_e)**2 / (sigma2_e ** 0.5)
            e_norm = tf.norm(e, ord=1) / e.shape[-1]

            if params['training']:
                # train for similarity
                tape.__exit__()

                (
                    d_f_bu, d_f_td, g_td, g_bu
                ) = tape.gradient(target=e,
                                  sources=[self.f_bu.trainable_variables,
                                           self.f_td.trainable_variables,
                                           x_prev_below, x_prev_above],
                                  output_gradients=g_prev_above+g_prev_below)

                # assign grads_and_vars
                grads_and_vars = [(d_f_bu, self.f_bu.trainable_variables),
                                  (d_f_td, self.f_td.trainable_variables)]

        else:  # params['mode'] == 'asleep'
            # no lateral. only top down (but how will the bottom up layers learn)?

            if params['training']:
                # train for difference

                grads_and_vars = []

            return

        return x, e, e_norm, g_td, g_bu, mu_e, sigma2_e, grads_and_vars
