from .t2c_fpn_focal import (
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
from ssd.modeling.retinaNet import RetinaNet

train.imshape = (128, 1024)

anchors.aspect_ratios=[[4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5], [4, 1.5]]

# min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
# anchors.min_sizes=[[10, 10], [28, 28], [40, 40], [50, 50], [86, 86], [128, 128], [128, 400]]

model = L(RetinaNet)(
    feature_extractor=backbone,
    anchors=anchors,
    loss_objective=loss_objective,
    num_classes = 9,
    anchor_prob_initialization = True,
)




