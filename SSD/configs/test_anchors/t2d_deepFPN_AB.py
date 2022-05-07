from .t2d_deepFPN import (
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

# anchors.min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]]
anchors.min_sizes=[[14, 14], [30, 30], [43, 43], [60, 60], [86, 86], [128, 128], [128, 400]]
# anchors.min_sizes=[[10, 10], [24, 24], [40, 40], [50, 50], [86, 86], [128, 128], [128, 400]]



