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

import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl

from .layers import MultiDirectionalLayer, PeripheralLayer
from .glom_rnn_cell import GLOMRNNCell


def GLOM(layers):

    # convert any tfkl.Input layers into uncontrollable PeripheralLayers
    layers = [PeripheralLayer(layer.input_shape) if isinstance(layer, (tfkl.Input, tfkl.InputLayer))
              else layer for layer in layers]

    for layer in layers:
        assert isinstance(layer, MultiDirectionalLayer), 'all layers should be `MultiDirectionalLayer`s'

    # initialize single recurrent cell
    rnn_cell = tfkl.RNN(GLOMRNNCell(layers=layers))

    # initialize input layers for each PeripheralLayer
    inputs = [tfkl.Input(layer.shape) for layer in layers
              if isinstance(layer, PeripheralLayer)]

    # convert to keras model
    outputs = rnn_cell(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
