from resnet import resnet50,resnet34
import torch 
import torch.nn as nn
import torch.nn.functional as F
from fpn import FPN50
from retina import RetinaNet


class BoundingBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50()
        self.classifier2 = nn.Conv2d(108, 18, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Conv2d(2048, 10, kernel_size=3, padding=1, bias=False)
        self.input_shape = (800,800)
        self.relu = nn.ReLU(inplace=True) 
        self.bn1 = nn.BatchNorm2d(16, momentum=0.01)
        self.classifier1 = nn.Conv2d(3, 18, kernel_size=3, padding=1, bias=False)
        self.regressor = nn.Conv2d(10, 4*4, kernel_size=3, padding=1, bias=False)
        self.pred = nn.Conv2d(10, 4*9, kernel_size=3, padding=1, bias=False)
    
    
    def forward(self, x):
        features = []
        for im in x:
          feature_list = []
          for i in im:
            #feat = i.view(-1,3,256,306)
            feat = self.classifier1(i.view(-1,3,256,306))
            feature_list.append(feat)

          feat = torch.cat(feature_list,1)
          #feat = torch.stack(feature_list, dim=0).sum(dim=0)
          features.append(feat)

        x = torch.cat(features)
        x = F.relu(self.classifier2(x))
        x = F.relu(self.encoder(x))
        x = F.relu(self.classifier(x))
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        pred_x = self.pred(x)
        box_x = self.regressor(x)

        return pred_x, box_x


