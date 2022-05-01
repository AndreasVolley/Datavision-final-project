import torchvision
import torch
from torch import nn
from typing import Tuple, List
from collections import OrderedDict


class FPN(torch.nn.Module):
    def __init__(self, output_channels: List[int],
            #image_channels: int,
            #output_feature_sizes: List[Tuple[int]]
            ):
        super().__init__()
        self.out_channels = output_channels
        # self.output_feature_shape = output_feature_sizes

        self.oldModel = torchvision.models.resnet34(pretrained=True)
        # self.model = torch.nn.Sequential(*(list(self.oldmodel.children())[4:-2]))
        
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
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractorP7 = torch.nn.Sequential(                                     # 1x8
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        
        # Extract features from P2-P7
        self.feature_extractorFPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 64, 64], 64)

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
        
        outFeatures = []
        FPNout = self.feature_extractorFPN(FeatureMaps)

        for k, v in FPNout.items():
            # print(k, v.shape)
            outFeatures.append(v)
        
        # outFeatures = [P2, P3, P4, P5, P6, P7]
        
        # print("hei: ", outFeatures.shape)
        
        # for idx, feature in enumerate(outFeatures):
        #     out_channel = self.out_channels[idx]
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     print("feature.shape[1:]", feature.shape[1:])
        #     assert feature.shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        # assert len(outFeatures) == len(self.output_feature_shape),\
        #    f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(outFeatures)}"
        
        return tuple(outFeatures)
        
        outFeatures = []
        for _, v in FPNout.items():
            outFeatures.append(v)

        # Wrote P2-P5 below just to remove errors. Do not use. Use orderedDict Above^
        # out_features = [P2,P3,P4,P5] 
        
        print("self.output_feature_shape", outFeatures.shape)
        
        # for idx, feature in enumerate(out_features):
        #     out_channel = self.out_channels[idx]
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     print("feature.shape[1:]", feature.shape[1:])
        #     assert feature.shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #    f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        
        return tuple(outFeatures)


