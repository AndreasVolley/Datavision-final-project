from SSD.ssd.modeling.backbones.fpn import FPN
from .t2b_data_augmentation import (
    train, optimizer, schedulers,
    loss_objective,
    model, 
    #backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform, 
    gpu_transform,
    label_map,
    anchors)

from tops.config import LazyCall as L
#from ssd.modeling.backbones import FPN
backbone = L(FPN)(type="resnet34", pretrained=True)