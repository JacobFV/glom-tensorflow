from typing import Text, Mapping, List, Optional, Union, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .ops import geometric_weighted_mean, get_lateral


class GLOMLayer(tfkl.Layer):

    hparam_defaults = {
        'window_size': (7, 7),
        'global_sparsity': 0.05,
        'normalize_beta': 0.9
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
            mu_e_prev, sigma2_e_prev
        ) = inputs

        train_ops = []

        params: dict = self.forward_param_defaults.copy().update(kwargs)

        if params['mode'] == 'awake':

            x_lat = get_lateral(x=x_prev, window_size=self.hparams['window_size'],
                                global_sparsity=self.hparams['global_sparsity'])
            x_bu = self.f_bu(x_prev_below)
            x_td = self.f_td(x_prev_above)
            x = geometric_weighted_mean(
                xs=[x_bu[..., None], x_td[..., None],
                    tf.transpose(x_lat, perm=(0,1,2,4,3))],  # [B, X, Y, i, D] -> [B, X, Y, D, i]
                ws=[e_prev_below[..., None],
                    1. / (e_prev_above[..., None] + 1e-2),  # not sure if this will work
                    tf.einsum('bxyid,bxyd->bxyi', x_lat, x_prev) / (self._depth ** 0.5)])

            new_e = x - x_prev

            with self.hparams['normalize_beta'] as beta:
                mu_e = beta * mu_e_prev + (1-beta) * new_e
                sigma2_e = beta * sigma2_e_prev + (1-beta) * (mu_e-new_e)**2

            e = (new_e - mu_e) / (sigma2_e ** 0.5)
            e_norm = tf.norm(e, ord=1)

            if params['training'] == True:
                # train for similarity

                ######
                # I need to manually backpropagate the errors
                # or somehow otherwise retrieve the gradients of the very
                # bottom layer to pass on to e_below
                train_ops = [
                    self._optimizer.minimize(
                        keras.losses.mse(y_true=tf.stop_gradient(x_bu-e), y_pred=x_bu),
                        var_list=self.f_bu.trainable_weights),
                    # same thing for f_td
                ]

        elif params['mode'] == 'asleep':
            # no lateral. only top down (but how will the bottom up layers learn)?

            if params['training'] == True:
                # train for difference

                ######
                # I need to manually backpropagate the errors
                # or somehow otherwise retrieve the gradients of the very
                # bottom layer to pass on to e_below
                train_ops = [
                    self._optimizer.minimize(
                        keras.losses.mse(y_true=tf.stop_gradient(x_bu-e), y_pred=x_bu),
                        var_list=self.f_bu.trainable_weights),
                    # same thing for f_td
                ]

        return x, e, mu_e, sigma2_e, e_norm, train_ops
