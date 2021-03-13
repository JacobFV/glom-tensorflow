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

from typing import Tuple, Mapping, List, Text, Optional, Callable

import tensorflow as tf

from .multidirectional_layer import MultiDirectionalLayer


def RoutingLayer(lower_concat_axis: int,
                 upper_split_axis: Optional[int] = None,
                 upper_split_indeces: Optional[List[int]] = None,
                 known_shape: Optional[Tuple[int, int, int]] = None,
                 transformation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None):
        """Concatentate multiple below layers and split into multiple above layers.

        This is a convenience initializer for `RoutingLayerComponent` which represents
        a single above layer input. While this does mean `transformation_fn` is executed
        multiple time in python, the backend tensorflow graph optimizes use to prevent
        inneficency.

        Example:
        >>> x1 = PeripheralLayer((32, 16))  # [32, 16]
        >>> x2 = PeripheralLayer((32, 16))  # [32, 16]
        >>> x = RoutingLayer(2)([x1, x2])  # [32, 32]
        >>> y1, y2 = RoutingLayer(1, 1, [12])  # [12, 32], [20, 32]

        :param lower_concat_axis: Indicates which axis to concatenate below layers on.
                Valid options: 1 or 2.
        :param upper_split_axis: Indicates which axis upper layers split apart on.
                Valid options: 1 or 2.
        :param upper_split_indeces: Indeces of splits
        :param known_shape: useful when making cyclic connections.
        :param transformation_fn: Tensor[X,Y,D] -> Tensor[X,Y,D] transformation function.
                Useful to enable non rectangular routing. For example, `transformation_fn`
                could skew and randomly slice and scatter the tensor locations.
        :return: `RoutingLayerComponent` or tuple of `RoutingLayerComponent` depending on
                whether upper_split is enabled.
        """

        if upper_split_axis is None:
            return RoutingLayerComponent(
                lower_concat_axis=lower_concat_axis,
                known_shape=known_shape,
                transformation_fn=transformation_fn
            )
        else:
            return [
                RoutingLayerComponent(
                    lower_concat_axis=lower_concat_axis,
                    upper_split_index=i,
                    upper_split_axis=upper_split_axis,
                    upper_split_indeces=upper_split_indeces,
                    known_shape=known_shape,
                    transformation_fn=transformation_fn
                )
                for i in range(1+len(upper_split_indeces))
            ]

class RoutingLayerComponent(MultiDirectionalLayer):

    def __init__(self,
                 lower_concat_axis: int,
                 upper_split_index: Optional[int] = None,
                 upper_split_axis: Optional[int] = None,
                 upper_split_indeces: Optional[List[int]] = None,
                 known_shape: Optional[Tuple[int, int, int]] = None,
                 transformation_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None):
        """Private single output component from routing layer.

        :param lower_concat_axis: Indicates which axis to concatenate below layers on.
                Valid options: 1 or 2.
        :param upper_split_index: Index of upper split that this layer passes to above layer.
        :param upper_split_axis: Indicates which axis upper layers split apart on.
                Valid options: 1 or 2.
        :param upper_split_indeces: Indeces of splits
        :param known_shape: useful when making cyclic connections.
        :param transformation_fn: Tensor[X,Y,D] -> Tensor[X,Y,D] transformation function.
                Useful to enable non rectangular routing. For example, `transformation_fn`
                could skew and randomly slice and scatter the tensor locations.
        """
        self.lower_concat_axis = lower_concat_axis
        self.lower_nonconcat_axis = 1 if lower_concat_axis == 2 else 2
        self.lower_split_indeces = []
        super(RoutingLayerComponent, self).__init__(shape=known_shape)

    def build(self, bu_shapes: List[tf.TensorShape], **kwargs):


#    def build(self, bu_shape: tf.Tensor, self_shape: tf.Tensor, td_shape: tf.Tensor, **kwargs):
        split_indeces = []
        for bu_shape_i in bu_shape:
            split_indeces.append(bu_shape_i[self.lower_concat_axis])
        self.lower_split_indeces = tf.cumsum(split_indeces)

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
        bu_ret_vars = {k: None for k, _ in prev_td_vars[0].items()}
        td_ret_vars = {k: None for k, _ in prev_bu_vars.items()}

        # bottom up: concatenate all x's

        # top down: split vars that correspond to the concatenation of dimensions
        for k, td_var in td_vars.items():
            # only split the var if it matches self.lower_split_indeces[-1] on self.lower_concat_axis
            if td_var.shape[self.lower_concat_axis] == self.lower_split_indeces[-1]:

            else:
