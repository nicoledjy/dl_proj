from resnet import resnet18
import torch 
import torch.nn as nn
import torch.nn.functional as F


class BoundingBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18()
        self.classifier = nn.Conv2d(512, 10, kernel_size=3, padding=1, bias=False)
        self.input_shape = (800,800)

        self.regressor = nn.Conv2d(10, 4*4, kernel_size=3, padding=1, bias=False)
        self.pred = nn.Conv2d(10, 4*9, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)

        pred_x = self.pred(x)
        box_x = self.regressor(x)
        return pred_x, box_x

