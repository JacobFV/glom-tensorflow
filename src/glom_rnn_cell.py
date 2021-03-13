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

from typing import Mapping, Text, List, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .layers import MultiDirectionalLayer, PeripheralLayer

class GLOMRNNCell(tfkl.AbstractRNNCell):

    def build(self, input_shape):

        ## recursively climb the layer graph and call build on layers whose children
        ## all have specified shapes

    def call(self, inputs, states):
        # rename all dict keys from '...' to '...(_above|_below|)_prev'
        pass

    @property
    def state_size(self):
        pass

    @property
    def output_size(self):
        pass

    def _rename_keys_layerwise(self, ) -> Mapping[Text, Mapping[Text, object]]: