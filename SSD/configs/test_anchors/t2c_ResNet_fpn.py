# from SSD.ssd.modeling.backbones.fpn import FPN
from .t2b_data_augmentation import (
    train, optimizer, schedulers,
    loss_objective,
    model, 
    # backbone,
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
from ssd.modeling.backbones.fpn_shallow import FPN

backbone = L(FPN)(
    output_channels = [64, 64, 64, 64, 64, 64], 
)