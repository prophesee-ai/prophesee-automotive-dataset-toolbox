import glob
import numpy as np
import os
import argparse
from src.metrics.coco_eval import evaluate_detection
from src.io.box_filtering import filter_boxes


BBOX_DTYPE = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})


def reformat_boxes(boxes):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if 't' not in boxes.dtype.names or 'class_confidence' not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=BBOX_DTYPE) 
        for name in boxes.dtype.names:
            if name == 'ts':
                new['t'] = boxes[name]
            elif name == 'confidence':
                new['class_confidence'] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes
        

def evaluate_folders(dt_folder, gt_folder):
    dt_file_paths = sorted(glob.glob(dt_folder+'/*'))
    gt_file_paths = sorted(glob.glob(gt_folder+'/*'))
    assert len(dt_file_paths) == len(gt_file_paths)
    print("There are {} GT bboxes and {} PRED bboxes".format(len(gt_file_paths), len(dt_file_paths)))
    result_boxes_list = [np.load(p) for p in dt_file_paths]
    gt_boxes_list = [np.load(p) for p in gt_file_paths]

    result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
    gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]


    gt_boxes_list = map(filter_boxes, gt_boxes_list)
    result_boxes_list = map(filter_boxes, result_boxes_list)
    evaluate_detection(gt_boxes_list, result_boxes_list)



def main():
    parser = argparse.ArgumentParser(prog='psee_evaluator.py')
    parser.add_argument('gt_folder', type=str, help='GT folder containing .npy files')
    parser.add_argument('dt_folder', type=str, help='RESULT folder containing .npy files')
    opt = parser.parse_args()
    evaluate_folders(opt.dt_folder, opt.gt_folder)

if __name__ == '__main__':
    main()
