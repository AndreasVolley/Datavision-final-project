import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class RetinaNet(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            anchor_prob_initialization):
        super().__init__()
        """
            Implements the SSD network.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        # self.regression_heads = []
        # self.classification_heads = []
        self.anchor_prob_initialization = anchor_prob_initialization
        
        
        self.regression_heads = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.Dropout2d(p=0.05),
            nn.Conv2d(256, 6 * 4, kernel_size=3, stride=1, padding=1),
        )
        self.classification_heads = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.GELU(),   
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.Dropout2d(p=0.05),
            nn.Conv2d(256, 6 * self.num_classes, kernel_size=3, stride=1, padding=1),
        )
        
        
        
        
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    ## Change this
    def _init_weights(self):
        layers = [*self.regression_heads, *self.classification_heads]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)    
        
        if self.anchor_prob_initialization:
            self.classification_heads[0].bias.data.fill_(0)
            self.classification_heads[2].bias.data.fill_(0)
            self.classification_heads[4].bias.data.fill_(0)
            self.classification_heads[6].bias.data.fill_(0)
            
            self.regression_heads[0].bias.data.fill_(0)
            self.regression_heads[2].bias.data.fill_(0)
            self.regression_heads[4].bias.data.fill_(0)
            self.regression_heads[6].bias.data.fill_(0)

            print("True")            
            ## Change 65520 to self.anchor_encoder.num_anchors
            p = 0.99
            backgroundClass = torch.log(torch.tensor(p * (9 -1) / (1 - p)))
            self.classification_heads[-1].bias.data[:65520] = backgroundClass

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for _, x in enumerate(features):
            bbox_delta = self.regression_heads(x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads(x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]