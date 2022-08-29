# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Define same filtering that we apply in:
"Learning to detect objects on a 1 Megapixel Event Camera" by Etienne Perot et al.

Namely, we apply 2 different filters:
1. skip all boxes before 0.5s (before we assume it is unlikely you have sufficient historic)
2. filter all boxes whose diagonal <= min_box_diag**2 and whose side <= min_box_side
"""

from __future__ import print_function
import numpy as np


def filter_boxes(boxes, skip_ts=int(5e5), min_box_diag=60, min_box_side=20):
    """Filters boxes according to the paper rule. 

    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0

    Args:
        boxes (np.ndarray): structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence'] 
        (example BBOX_DTYPE is provided in src/box_loading.py)

    Returns:
        boxes: filtered boxes
    """
    ts = boxes['t'] 
    width = boxes['w']
    height = boxes['h']
    diag_square = width**2+height**2
    mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)
    return boxes[mask]

