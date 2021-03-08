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

from typing import Optional, Text

from ..utils.make_unique import make_name_unique


class MultiDirectionalLayer():

    def __init__(self, name: Optional[Text] = None):
        self.name = make_name_unique(name)
        self.below_layers = []

    def __call__(self, layer):
        assert isinstance(layer, MultiDirectionalLayer), 'must call with another `MultiDirectionalLayer` to build graph'
        self.below_layers = [layer]

    def build(self, bu_var_shapes, self_var_shapes, td_var_shapes, **kwargs):
        raise NotImplementedError('subclasses should impliment `build` method')

    def forward(self, bu_vars, self_vars, td_vars, **kwargs):
        raise NotImplementedError('subclasses should impliment `forward` method')