import torchvision
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors,RandomHorizontalFlip, RandomSampleCrop,)

from .base import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

train.imshape = (128, 1024)
train.epochs = 50

data_train.dataset.transform  = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(Resize)(imshape=train.imshape),
    L(RandomHorizontalFlip)(), 
    L(GroundTruthBoxesToAnchors)(anchors=anchors, iou_threshold=0.5),
])

