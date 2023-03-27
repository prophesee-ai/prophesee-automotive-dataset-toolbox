# Prophesee's Automotive Dataset Toolbox

[![Prophesee Automotive Dataset](media/GEN1-Automotive-detection-dataset-thumbnail.jpg)](https://www.youtube.com/watch?v=VJ7VSUqFvVE&feature=youtu.be) 

This repository contains a set of Python scripts to evaluate the Automotive Datasets provided by Prophesee.

## Requirements

The scripts can be launched with Python 2.x or Python 3.x:
* `io` requires [NumPy](https://numpy.org/) 
* `visualize` requires also [OpenCV](https://opencv.org/) with python bindings.

You can install all the dependencies using [pip](https://pip.pypa.io/en/stable/):
```
pip install numpy
pip install opencv-python
```

## Get the data

### 1 Megapixel Automotive Detection Dataset

Go to the [dataset presentation page](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) and download the dataset.
The dataset is split between train, test and val folders. 
Files consist of 60 seconds recordings that were cut from longer recording sessions. Cuts from a single recording session are all in the same training split.

Bounding box annotations for 7 classes (pedestrians, two-wheelers, cars, trucks, buses, traffic signs, traffic lights) are obtained in a semi automated way.
For more details, please refer to [our NeurIPS paper](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf).
Also note that, as explained in the paper, the official evaluation code considers only 3 classes for mAP computation (pedestrians, two-wheelers, cars).

### GEN1 Automotive Detection Dataset 

Go to the [dataset presentation page](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) and download the dataset.
(200G compressed and 750G uncompressed !). 
The dataset is split into 10 archive files that can be independently used (2 for testing and validation sets each and six for training set).
Each archive contains up to 500 files and their annotations.

Please notice that compared to the 1Mpx dataset, the Gen1 dataset contains only annotations for pedestrians, and cars.
Moreover, in contrast with the 1Mpx dataset, these labels were manually annotated.
For more details, please refer to [our article](https://arxiv.org/abs/2001.08499)

## Visualization

To view a few files and their annotation just use
    `python3 dataset_visualization.py file_1_td.dat file_2_td.dat ... file_n_dat`
And it will display those events video in a grid. You can use it with any number of files,
but a large number of them will make the display slow!

## Reading files in python

There is a convenience class to read files that works both for the event .dat files and their annotations.
A small tutorial can be found [here](tutorial.ipynb)


## Running a baseline

Now you can start by running a baseline either by looking into [the last results in event-based literature](https://github.com/uzh-rpg/event-based_vision_resources) or by leveraging [the e2vid project](https://github.com/uzh-rpg/rpg_e2vid) of the University of Zurich's Robotic and Perception Group to run a frame-based detection algorithm!


## Evaluation using the COCO API

### DISCLAIMER: New Dataset! 

To account for the [1 Megapixel Automotive Detection Dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) described in our NeurIPS article: ["Learning to Detect Objects with a 1 Megapixel Event Camera"](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf) by Etienne Perot, Pierre de Tournemire, Davide Nitti, Jonathan Masci and Amos Sironi, the format has slightly changed. 
Essentially `ts` has been renamed `t` in events and box events, alongside `confidence` is now `class_confidence`
Also now, for comparison with our result inside this paper, you need to filter too small boxes and boxes appearing before 0.5s inside each recording. We provide such function
as following example will show.


If you install the [API from COCO](https://github.com/cocodataset/cocoapi) you can use the provided helper function in `metrics` to get mean average precision metrics.
This is a usage example if you saved your detection results in the same format as the Ground Truth:
```python
import numpy as np
from src.metrics.coco_eval import evaluate_detection
from src.io.box_loading import reformat_boxes

RESULT_FILE_PATHS = ["file1_results_bbox.npy", "file2_results_bbox.npy"]
GT_FILE_PATHS = ["file1_bbox.npy", "file2_bbox.npy"]

result_boxes_list = [np.load(p) for p in RESULT_FILE_PATHS]
gt_boxes_list = [np.load(p) for p in GT_FILE_PATHS]

# For backward-compatibility
result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]

# For fair comparison with paper results
gt_boxes_list = map(filter_boxes, gt_boxes_list)
result_boxes_list = map(filter_boxes, result_boxes_list)

evaluate_detection(gt_boxes_list, result_boxes_list)
```
We provide a complete evaluator at `src/psee_evaluator.py`. Note that box filtering uses a diagonal threshold different for the 1 megapixel camera and for the qvga one (60 and 30).


## Credit
When using those tools in an academic context, please cite the article ["Learning to Detect Objects with a 1 Megapixel Event Camera" by Etienne Perot, Pierre de Tournemire, Davide Nitti, Jonathan Masci, Amos Sironi, In 34th Conference on Neural Information Processing Systems (NeurIPS) 2020](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf)


## Contacts
The code is open to contributions, so do not hesitate to ask questions, propose pull requests or create bug reports.
For any other information or inquiries, contact us [here](https://www.prophesee.ai/contact-us/)
