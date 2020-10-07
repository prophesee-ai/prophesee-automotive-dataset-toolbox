import glob
import numpy as np
import os
import argparse
from src.metrics.coco_eval import evaluate_detection
from src.io.box_filtering import filter_boxes
from src.io.box_loading import reformat_boxes


        

def evaluate_folders(dt_folder, gt_folder, camera):
    dt_file_paths = sorted(glob.glob(dt_folder+'/*.npy'))
    gt_file_paths = sorted(glob.glob(gt_folder+'/*.npy'))
    assert len(dt_file_paths) == len(gt_file_paths)
    print("There are {} GT bboxes and {} PRED bboxes".format(len(gt_file_paths), len(dt_file_paths)))
    result_boxes_list = [np.load(p) for p in dt_file_paths]
    gt_boxes_list = [np.load(p) for p in gt_file_paths]

    result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
    gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]

    min_box_diag = 60 if camera == 'GEN4' else 30
    min_box_side = 20 if camera == 'GEN1' else 10

    filter_boxes_fn = lambda x:filter_boxes(x, int(1e5), min_box_diag, min_box_side)

    gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
    result_boxes_list = map(filter_boxes_fn, result_boxes_list)
    evaluate_detection(gt_boxes_list, result_boxes_list)



def main():
    parser = argparse.ArgumentParser(prog='psee_evaluator.py')
    parser.add_argument('gt_folder', type=str, help='GT folder containing .npy files')
    parser.add_argument('dt_folder', type=str, help='RESULT folder containing .npy files')
    parser.add_argument('--camera', type=str, default='GEN4', help='GEN1 (QVGA) or GEN4 (720p)')
    opt = parser.parse_args()
    evaluate_folders(opt.dt_folder, opt.gt_folder, opt.camera)

if __name__ == '__main__':
    main()
