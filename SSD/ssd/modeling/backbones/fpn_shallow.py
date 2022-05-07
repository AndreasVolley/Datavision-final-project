from turtle import forward

from pyrsistent import freeze
import torchvision
import torch
from torch import nn
from typing import Tuple, List
from collections import OrderedDict
from torch.autograd import Variable

class FPN(torch.nn.Module):
    def __init__(self, output_channels: List[int],
                 flag,
            #image_channels: int,
            #output_feature_sizes: List[Tuple[int]]
            ):
        super().__init__()
        self.out_channels = output_channels
        self.flag = flag
        # self.output_feature_shape = output_feature_sizes

        self.oldModel = torchvision.models.resnet34(pretrained=True)
        
        if self.flag == "fpn":
            chan = 64
        elif self.flag == "fpnDeep":
            chan = 256
                
        ## Add feature extractors
        ###############################################################################################################
        self.feature_extractorP2 = torch.nn.Sequential(                                     # 32x256
            self.oldModel.conv1,
            self.oldModel.bn1,
            self.oldModel.relu,
            self.oldModel.maxpool,
            self.oldModel.layer1,
        )
        self.feature_extractorP3 = torch.nn.Sequential(self.oldModel.layer2)                # 16x128
        self.feature_extractorP4 = torch.nn.Sequential(self.oldModel.layer3)                # 8x64
        self.feature_extractorP5 = torch.nn.Sequential(self.oldModel.layer4)                # 4x32
        self.feature_extractorP6 = torch.nn.Sequential(                                     # 2x16
            nn.Conv2d(
                in_channels=512,
                out_channels=chan,  
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.feature_extractorP7 = torch.nn.Sequential(                                    # 1x8
            nn.Conv2d(
                in_channels=chan,
                out_channels=chan,
                kernel_size=3,
                stride=2,
                padding=1
            ),    
        )

        # FPN
        self.feature_extractorFPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, chan, chan], chan)

    def forward(self, x):

        ## Custom Backbone
        P2 = self.feature_extractorP2(x)
        P3 = self.feature_extractorP3(P2)
        P4 = self.feature_extractorP4(P3)
        P5 = self.feature_extractorP5(P4)
        P6 = self.feature_extractorP6(P5)
        P7 = self.feature_extractorP7(P6)
        
        # FPN
        FeatureMaps = OrderedDict()
        FeatureMaps['P2'] = P2
        FeatureMaps['P3'] = P3
        FeatureMaps['P4'] = P4
        FeatureMaps['P5'] = P5
        FeatureMaps['P6'] = P6
        FeatureMaps['P7'] = P7
        
        outFeatures = []
        FPNout = self.feature_extractorFPN(FeatureMaps)

        for _, v in FPNout.items():
            outFeatures.append(v)
        
        return tuple(outFeatures)