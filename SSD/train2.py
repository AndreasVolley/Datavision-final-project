import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
import click
import torch
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops.config import instantiate
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True


def print_config(cfg):
    container = OmegaConf.to_container(cfg)
    pp = pprint.PrettyPrinter(indent=2, compact=False)
    print("--------------------Config file below--------------------")
    pp.pprint(container)
    print("--------------------End of config file--------------------")


from detectron2.data.datasets import register_coco_instances
#register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
#register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = mpimg.imread("Photos/photo2.jpg")
print(im.shape)

cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

#######################################################################################################################

