#!/usr/bin/env python
# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events, mimicking dat_events_tools.py.
In particular :
    -> defines functions to read events from binary .npy files using numpy
    -> defines functions to write events to binary .dat files using numpy (TODO later)
"""

from __future__ import print_function
import numpy as np


def stream_td_data(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle
    args :
        - file_handle: file object
        - buffer: pre-allocated buffer to fill with events
        - dtype:  expected fields
        - ev_count: number of events
    """
    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat['t'])
    for name, _ in dtype:
        buffer[name][:count] = dat[name]


def parse_header(fhandle):
    """
    Parses the header of a .npy file
    Args:
        - f file handle to a .npy file
    return :
        - int position of the file cursor after the header
        - int type of event
        - int size of event in bytes
        - size (height, width) tuple of int or (None, None)
    """
    version = np.lib.format.read_magic(fhandle)
    shape, fortran, dtype = np.lib.format._read_array_header(fhandle, version)
    assert not fortran, "Fortran order arrays not supported"
    # Get the number of elements in one 'row' by taking
    # a product over all other dimensions.
    if len(shape) == 0:
        count = 1
    else:
        count = np.multiply.reduce(shape, dtype=np.int64)
    ev_size = dtype.itemsize
    assert ev_size != 0
    start = fhandle.tell()
    # turn numpy.dtype into an iterable list
    ev_type = [(x, str(dtype.fields[x][0])) for x in dtype.names]
    # filter name to have only t and not ts
    ev_type = [(name if name != "ts" else "t", desc) for name, desc in ev_type]
    ev_type = [(name if name != "confidence" else "class_confidence", desc) for name, desc in ev_type]
    size = (None, None)
    size = (None, None)

    return start, ev_type, ev_size, size
