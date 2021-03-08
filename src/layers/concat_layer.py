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

from .multidirectional_layer import MultiDirectionalLayer

class ConcatLayer(MultiDirectionalLayer):

    def __init__(self, concat_axis: int, **kwargs):
        """Public constructor for ConcatLayer.
        
        :param concat_axis: Valid options: 1 or 2. Axis to concatentate below layers along.
        :param kwargs: superclass __init__ parameters.
        """
        self.concat_axis = concat_axis
        self.nonconcat_axis = 1 if concat_axis == 2 else 2
        self.split_indeces = []
        super(ConcatLayer, self).__init__(**kwargs)

    def __call__(self, layer):
        try:
            layer = list(layer)
        finally:
            pass
        assert isinstance(layer, list), 'must call with list-like of `MultiDirectionalLayer` to build graph'
        self.below_layers = layer

    def build(self, bu_var_shapes, self_var_shapes, td_var_shapes, **kwargs):
        x_prev_shape, sigma2_e_prev_shape = self_var_shapes
        x_prev_above_shape, g_prev_td_shape, e_above_norm_prev_shape = td_var_shapes

        for bu_var_shapes_i in bu_var_shapes:
            x_prev_below_shape, g_prev_bu_shape, e_below_norm_prev_shape = bu_var_shapes_i
            assert x_prev_below_shape[self.nonconcat_axis] == x_prev_shape[self.nonconcat_axis]
            self.split_indeces.append(x_prev_below_shape[self.concat_axis])

        assert x_prev_shape[self.concat_axis] == sum(self.split_indeces)


    def forward(self, bu_vars, self_vars, td_vars, **kwargs):
        x_prev_below, g_prev_bu, e_below_norm_prev = bu_vars
        x_prev, sigma2_e_prev = self_vars
        x_prev_above, g_prev_td, e_above_norm_prev = td_vars

