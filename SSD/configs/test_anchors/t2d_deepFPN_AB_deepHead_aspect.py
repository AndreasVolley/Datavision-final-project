from .t2d_deepFPN_AB_deepHead import (
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
from ssd.modeling.retinaNet_shallow import RetinaNet
from ssd.modeling.backbones.fpn_shallow import FPN

train.imshape = (128, 1024)

anchors.aspect_ratios=[[4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5]]



