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

_UNIQUE_NAME_COUNTER = {}


def make_name_unique(name):
    """Sequentially suffixes names. Non-idempotent method to ensure no name collisions.
    Example:
        >>> make_name_unique('Node')
        'Node1'
        >>> make_name_unique('Node')
        'Node2'
        >>> make_name_unique('Node')
        'Node3'
        >>> make_name_unique('Node1')
        'Node11'
    Args:
        name: Node instance name to make unique.
    Returns: unique name
    """
    if name in _UNIQUE_NAME_COUNTER:
        _UNIQUE_NAME_COUNTER[name] += 1
    else:
        _UNIQUE_NAME_COUNTER[name] = 1
    return name + str(_UNIQUE_NAME_COUNTER[name])
