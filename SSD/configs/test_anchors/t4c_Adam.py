from .t2d_deepFPN_AB_deepHead_aspect import (
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

optimizer = L(torch.optim.Adam)(
    lr=4e-4,
    #weight_decay=0.0005
)
