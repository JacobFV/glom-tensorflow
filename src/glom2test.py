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

import tensorflow as tf

from .glom2 import GLOMCell


glom_cell = GLOMCell(
    input_layers=['L0'],
    output_layers=['L0'],
    layer_sizes=dict(L0=(28, 28, 16), L1=(28, 28, 32), L2=(28, 28, 64)),
    connections=[
        dict(inputs=['L0'], outputs=['L1'], type='bu'),
        dict(inputs=['L1'], outputs=['L2'], type='bu'),
        dict(inputs=['L2'], outputs=['L1'], type='td'),
        dict(inputs=['L1'], outputs=['L0'], type='td'),
    ],
    name='glom_cell'
)

glom_cell.set_mode('awake')
glom_rnn = tf.keras.layers.RNN(glom_cell)

# convert static images to sequences
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def convertImageToOneHot(image, N=16, T=16):
    image = image[None, ...]
    image = image / 255.0
    image = image * N
    image = tf.cast(image, tf.int64)
    image = tf.one_hot(image, depth=N)
    image = tf.cast(image, tf.float32)
    image = tf.tile(image[:, None, :, :, :], multiples=[1, T, 1, 1, 1])
    return image


for train_image in train_images:
    train_image = convertImageToOneHot(train_image, N=16, T=32)
    glom_rnn(train_image[None, ...])


print('done')
