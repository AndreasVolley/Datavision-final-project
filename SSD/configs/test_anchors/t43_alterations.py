from .t2d_anchor_boxes import (
    train, optimizer, schedulers,
    loss_objective,
    # model, 
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

#train.imshape = (128, 1024)

#anchors.aspect_ratios=[[4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5]]

optimizer = L(torch.optim.Adam)(
    lr=4e-4, weight_decay=0.0005
)

