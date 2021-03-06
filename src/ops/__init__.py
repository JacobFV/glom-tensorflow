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

from typing import List

import tensorflow as tf

def geometric_weighted_mean(xs: List[tf.Tensor], ws: List[tf.Tensor]) -> tf.Tensor:
    """Computes the geometric weighted mean of values arranged
    along the final dimension of a list of `Tensor`s.

    Uses logorithms to speed up computations:
    GWM(x,w) = (ProdSum x^w)^(1/Sum w)
             = (e^ln(ProdSum x^w))^(1/Sum w)
             = (e^(Sum ln(x^w)))^(1/Sum w)
             = (e^(Sum w*ln(x)))^(1/Sum w)
             = e^( (Sum w*ln(x)) / Sum w)

    :param xs: values to compute geometric mean over.
            List[Tensor] [..., N_i] where N_i can be different across tensors
            It can even be 1. However it must match the number of values in `ws`.
    :param ws: corresponding nonnegative weights for values (ws >= 0; sum ws[..., :] > 0)
            List[Tensor] [..., N_i] where N_i can be different across tensors
            It can even be 1. However it must match the number of values in `xs`.
    :return: geometric weighted mean Tensor: [...]
    """

    xs = tf.concat(xs, axis=-1)
    ws = tf.concat(ws, axis=-1)

    return tf.exp(tf.reduce_sum(ws * tf.math.log(xs), axis=-1) /
                  tf.reduce_sum(ws, axis=-1))

def get_lateral(x, window_size, global_sparsity) -> tf.Tensor:
    """gets local and global lateral elements in a grid

    :param x: Tensor: [B, X, Y, D]
    :param window_size: tuple: [h, w]
    :param global_sparsity: percentage (0.-1.) of global slices to take
    :return: per-location receptive field: [B, X, Y, i, D]
    """

    # local elements


    # global elements


    pass