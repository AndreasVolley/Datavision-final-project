from .t2d_deepFPN_AB import (
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

train.imshape = (128, 1024)

model = L(RetinaNet)(
    feature_extractor=backbone,
    anchors=anchors,
    loss_objective=loss_objective,
    num_classes = 9,
    anchor_prob_initialization = True,
    flag = "deepHeads",
)




