"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import cv2
import argparse

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader


def play_files_parallel(td_files, labels=None, delta_t=50000, skip=0):
    """
    play simultaneously files and their boxes in a rectangular format
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    # use the naming pattern to find the corresponding box file
    box_videos = [PSEELoader(td_file.replace('_td.dat', '_bbox.npy')) for td_file in td_files]

    height, width = videos[0].get_size()

    # optionally skip n minutes in all videos
    for v in videos + box_videos:
        v.seek_time(skip)

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)

    # while all videos have something to read
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]
        box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos]

        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)

            vis.draw_bboxes(im, boxes)

        # display the result
        cv2.imshow('out', frame)
        cv2.waitKey(1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+",
                        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=0, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=20000, type=int, help="load files by delta_t in microseconds")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, skip=ARGS.skip, delta_t=ARGS.delta_t)
