# from SSD.ssd.modeling.backbones.fpn import FPN
from .t2c_fpn import (
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

train.imshape = (128, 1024)

from tops.config import LazyCall as L
import torch
import numpy as np
from ssd.modeling.focalLoss import FocalLoss

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=torch.tensor([10, *[1000 for i in range(model.num_classes-1)]]))

