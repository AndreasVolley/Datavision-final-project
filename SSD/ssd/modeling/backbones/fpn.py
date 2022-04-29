import torchvision
import torch
from torch import nn
from typing import Tuple, List


class FPN(torch.nn.Module):
    def __init__(self,output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        model = torchvision.models.resnet18(pretrained=True)

        img = torch.zeros((1, 3, 128, 1024))
        print("the model:", model)
        x = model.conv1(img)
        print("x:", x)

    def forward(self, x):
         out1 = self.model.layer1()
         out2 = self.model.layer2(out1)
         out3 = self.model.layer3(out2)
         out4 = self.model.layer4(out3)
         out5 = self.model.layer5(out4)
         out6 = self.model.layer6(out5)
        

         out_features = [out1,out2,out3,out4,out5,out6]
         print("self.output_feature_shape", self.output_feature_shape)
        

         for idx, feature in enumerate(out_features):
             out_channel = self.out_channels[idx]
             h, w = self.output_feature_shape[idx]
             expected_shape = (out_channel, h, w)
             print("feature.shape[1:]", feature.shape[1:])
             assert feature.shape[1:] == expected_shape, \
                 f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
         assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
         return tuple(out_features)


