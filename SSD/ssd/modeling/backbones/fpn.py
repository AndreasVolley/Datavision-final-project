import torchvision
import torch
from torch import nn
from typing import Tuple, List
from collections import OrderedDict


class FPN(torch.nn.Module):
    def __init__(self,output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # self.oldmodel = torchvision.models.resnet18(pretrained=True)
        # self.model = torch.nn.Sequential(*(list(self.oldmodel.children())[4:-2]))
        
        ## Add feature extractors
        self.feature_extractorP2 = torch.nn.Sequential(*(list(self.oldmodel.children())[0:5]))
        self.feature_extractorP3 = torch.nn.Sequential(*(list(self.oldmodel.children())[5]))
        self.feature_extractorP4 = torch.nn.Sequential(*(list(self.oldmodel.children())[6]))
        self.feature_extractorP5 = torch.nn.Sequential(*(list(self.oldmodel.children())[7]))
        self.feature_extractorP6 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=True
            )
        )
        self.feature_extractorP7 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=True
            )
        )
        
        self.feature_extractorFPN = torchvision.ops.FeaturePyramidNetwork([128, 256, 512, 1028, 64, 64], 128)

        img = torch.zeros((1, 3, 128, 1024))
        print("the model:", self.model)
        x = self.model.conv1(img)
        
        print("x:", x)

    def forward(self, x):

        # Only use P3-P7?

        P2 = self.feature_extractorP2(x)
        P3 = self.feature_extractorP3(P2)
        P4 = self.feature_extractorP4(P3)
        P5 = self.feature_extractorP5(P4)
        P6 = self.feature_extractorP6(P5)
        P7 = self.feature_extractorP7(P6)
        
        FeatureMaps = OrderedDict()
        FeatureMaps['P2'] = P2
        FeatureMaps['P3'] = P3
        FeatureMaps['P4'] = P4
        FeatureMaps['P5'] = P5
        FeatureMaps['P6'] = P6
        FeatureMaps['P7'] = P7
        
        FPNout = self.feature_extractorFPN(FeatureMaps)

        # Wrote P2-P5 below just to remove errors. Do not use. Use orderedDict Above^
        out_features = [P2,P3,P4,P5]
        
        
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


