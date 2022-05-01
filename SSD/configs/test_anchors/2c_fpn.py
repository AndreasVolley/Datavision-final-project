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
from ssd.modeling import AnchorBoxes
from ssd.modeling.backbones.fpn import FPN
from ssd.modeling.focalLoss import FocalLoss

backbone = L(FPN)(
    output_channels = [64, 128, 256, 512, 64, 64],
    #image_channels=3,
    #output_feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
)

loss_objective = L(FocalLoss)(anchors=anchors, alpha=[0.01, *[1 for i in range(model.num_classes-1)]])

