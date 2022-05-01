# import torch
# from typing import Tuple, List


# class BasicModel(torch.nn.Module):
#     """
#     This is a basic backbone for SSD.
#     The feature extractor outputs a list of 6 feature maps, with the sizes:
#     [shape(-1, output_channels[0], 38, 38),
#      shape(-1, output_channels[1], 19, 19),
#      shape(-1, output_channels[2], 10, 10),
#      shape(-1, output_channels[3], 5, 5),
#      shape(-1, output_channels[3], 3, 3),
#      shape(-1, output_channels[4], 1, 1)]
#     """
#     def __init__(self,
#             output_channels: List[int],
#             image_channels: int,
#             output_feature_sizes: List[Tuple[int]]):
#         super().__init__()
#         self.out_channels = output_channels
#         self.output_feature_shape = output_feature_sizes

#     def forward(self, x):
#         """
#         The forward functiom should output features with shape:
#             [shape(-1, output_channels[0], 38, 38),
#             shape(-1, output_channels[1], 19, 19),
#             shape(-1, output_channels[2], 10, 10),
#             shape(-1, output_channels[3], 5, 5),
#             shape(-1, output_channels[3], 3, 3),
#             shape(-1, output_channels[4], 1, 1)]
#         We have added assertion tests to check this, iteration through out_features,
#         where out_features[0] should have the shape:
#             shape(-1, output_channels[0], 38, 38),
#         """
#         out_features = []
#         for idx, feature in enumerate(out_features):
#             out_channel = self.out_channels[idx]
#             h, w = self.output_feature_shape[idx]
#             expected_shape = (out_channel, h, w)
#             assert feature.shape[1:] == expected_shape, \
#                 f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
#         assert len(out_features) == len(self.output_feature_shape),\
#             f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
#         return tuple(out_features)

    
#This is my basic:


import torch
from torch import nn
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # num_filters1 = 32
        # num_filters2 = num_filters1*2 #64
        # num_filters3 = num_filters2*2 #128
        # num_filters4 = num_filters3*2 #256
        kernal_conv = 3
        kernal_maxPool = 2
        stride_maxPool = 2
        pad = 1
        drop_prop = 0.02
        
        self.feature_extractor_A = nn.Sequential(     
        )
        self.feature_extractor_B = nn.Sequential()
        self.feature_extractor_C = nn.Sequential()

        self.feature_extractor1 = nn.Sequential(
           
           #Added extra conv layers
            # nn.Conv2d(
            #     in_channels=image_channels,
            #     out_channels=32,
            #     kernel_size=kernal_conv,
            #     stride=1,
            #     padding=pad
            # ),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=kernal_conv,
            #     stride=1,
            #     padding=pad
            # ),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=64,
            #     out_channels=128,
            #     kernel_size=kernal_conv,
            #     stride=1,
            #     padding=pad
            # ),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            #End extra conv layers

            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size= kernal_maxPool, stride = stride_maxPool),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),
            #nn.MaxPool2d(kernel_size= kernal_maxPool, stride = stride_maxPool),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)
            

        self.feature_extractor2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            #Added
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[1],
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)


        self.feature_extractor3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=256,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            #Added
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[2],
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)


        self.feature_extractor4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            #Added
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[3],
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)
        

        self.feature_extractor5 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            #Added
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernal_conv,
                stride=1,
                padding=pad
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[4],
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)

        
        self.feature_extractor6 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=128,
                kernel_size=kernal_conv,
                stride=2,
                padding=pad
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            #Added
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=pad
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[5],
                kernel_size=2,
                stride=1,
                padding=0
            ),
            nn.Dropout2d(p=drop_prop),
            nn.GELU(),)

            

        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        out1 = self.feature_extractor1(x)
        out2 = self.feature_extractor2(out1)
        out3 = self.feature_extractor3(out2)
        out4 = self.feature_extractor4(out3)
        out5 = self.feature_extractor5(out4)
        out6 = self.feature_extractor6(out5)

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
