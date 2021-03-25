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

from ..glom import GLOM

import os
os.system("rm -r /tmp/tfdbg2_logdir")
"""from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "/tmp/tfdbg2_logdir"])
url = tb.launch()
print(url)"""

tf.debugging.experimental.enable_dump_debug_info(
    dump_root="/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)

glom = GLOM(
    input_layers=['L0'],
    output_layers=['L0'],
    layer_sizes=dict(L0=(28, 28, 16), L1=(28, 28, 32), L2=(28, 28, 64)),
    connections=[
        dict(inputs=['L0'], outputs=['L1'], type='bu'),
        dict(inputs=['L1'], outputs=['L2'], type='bu'),
        dict(inputs=['L2'], outputs=['L1'], type='td'),
        dict(inputs=['L1'], outputs=['L0'], type='td'),
    ],
    name='glom'
)

glom.set_mode('awake')
layer_states = glom.get_initial_state()

# convert static images to sequences
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def convertImageToOneHot(image, N=16):
    image = image[None, ...]
    image = image / 255.0
    image = image * N
    image = tf.cast(image, tf.int64)
    image = tf.one_hot(image, depth=N)
    image = tf.cast(image, tf.float32)
    return image


# @tf.function
def train_step(layer_states):
    layer_states = glom(layer_states)
    return layer_states

for i, train_image in enumerate(train_images[:5]):
    print(f'batch {i}')
    train_image = convertImageToOneHot(train_image, N=16)
    layer_states['L0']['x'] = train_image

    layer_states = train_step(layer_states)
