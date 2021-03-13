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

from typing import Optional, Text, Mapping, Tuple, List, Union

import tensorflow as tf

from ..utils.make_unique import make_name_unique


class MultiDirectionalLayer():

    def __init__(self,
                 shape: Optional[tf.TensorShape] = None,
                 name: Optional[Text] = None):
        self.name = make_name_unique(name)
        self.below_layers = []
        self.above_layers = []
        self.shape = shape  # shape including depth dimension

    def __call__(self, layer):
        if isinstance(layer, MultiDirectionalLayer):
            self.below_layers = [layer]
        elif isinstance(layer, (tuple, list)):
            layers = layer
            self.below_layers = layers
        for layer in self.below_layers:
            layer.above_layers.append(self)
        return self

    def build(self, bu_shapes: List[tf.TensorShape], **kwargs):
        """Build internal layer weights with shape information. Shapes do not include batch dimension.

        :param bu_shapes: shapes of lower layers
        :param kwargs:
        :return:
        """
        raise NotImplementedError('subclasses should impliment `build` method')

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
        raise NotImplementedError('subclasses should impliment `forward` method')

    def get_initial_state_shape(self, **kwargs) -> Mapping[Text, tf.TensorShape]:
        """Get the size of initial state for this `MultiDirectionalLayer` not including batch dimensions.

        :param kwargs: dictionary containing subclass specific data
        """
        raise NotImplementedError('subclasses should impliment `initial_state` method')