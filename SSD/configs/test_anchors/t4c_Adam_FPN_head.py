from .t4c_Adam_FPN import (
    train, optimizer, schedulers,
    loss_objective,
    model, 
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform, 
    gpu_transform,
    label_map,
    anchors)

from tops.config import LazyCall as L
import torch
from ssd.modeling.retinaNet_shallow import RetinaNet
from ssd.modeling.backbones.fpn_shallow import FPN

train.imshape = (128, 1024)

model = L(RetinaNet)(
    feature_extractor=backbone,
    anchors=anchors,
    loss_objective=loss_objective,
    num_classes = 9,
    anchor_prob_initialization = True,
    flag = "deepHeadsConfig",
)