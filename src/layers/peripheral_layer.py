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

from typing import Optional, Tuple

import tensorflow as tf

from .. import backend
from .multidirectional_layer import MultiDirectionalLayer

class PeripheralLayer(MultiDirectionalLayer):
    """Interal layer type for inputs and outputs."""

    def __init__(self,
                 shape: Tuple[int, int],
                 allow_control: Optional[bool] = True,
                 controllable_mask: Optional[tf.Tensor] = None):

        self.shape = shape

        if controllable_mask is None:
            controllable_mask = tf.cast(allow_control, backend.config.get_dtype())

        self.controllable_mask = controllable_mask

        super(PeripheralLayer, self).__init__()