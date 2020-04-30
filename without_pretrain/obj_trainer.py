import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from anchors import get_bbox_gt, Transform_coor


matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


# define base trainer 
class Trainer:
    def __init__(self, model, optimizer, scheduler, trainloader, valloader, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.scheduler = scheduler
        self.device = device

    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError


# construct anchor boxes
def get_anchor_boxes(scaleX=[100, 70, 50, 20], scaleY=[25, 20, 15, 5]):
    widths = torch.tensor(scaleX)
    heights = torch.tensor(scaleY)
    ref_boxes = []
    for x in range(800):
        for y in range(800):
            x_r = widths + x
            y_r = heights + y
            x_l = torch.tensor([x, x, x, x])
            y_l = torch.tensor([y, y, y, y])
            x_r = x_r.unsqueeze(0)
            y_r = y_r.unsqueeze(0)
            x_l = x_l.unsqueeze(0)
            y_l = y_l.unsqueeze(0)
            ref_box = torch.cat((x_l, y_l, x_r, y_r))
            ref_box = ref_box.permute((1,0))
            ref_boxes.append(ref_box)

    anchor_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.double)
    
    return anchor_boxes


def my_collate_fn(batch):
    images = []
    gt_boxes = get_anchor_boxes()
    img_h = 256
    img_w = 306
    map_sz = 800
    class_target = []
    box_target = []

    for x in batch:
        images.append(x[0])  
        gt_classes, gt_offsets = get_bbox_gt(x[1]['bounding_box'], x[1]['category'], gt_boxes, map_sz)
        class_target.append(gt_classes)
        box_target.append(gt_offsets)

    samples = torch.stack(images)
    samples = samples.view(len(batch), -1, img_h, img_w).double()
    class_target = torch.stack(class_target)
    box_target = torch.stack(box_target)

    return samples, class_target, box_target


# object detection trainer
class ObjDetTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, trainloader, valloader, device="cpu", scaleX=[100, 70, 50, 20], scaleY=[25, 20, 15, 5]):
        super().__init__(model, optimizer, scheduler, trainloader, valloader, device)
        self.scaleX = scaleX
        self.scaleY = scaleY
        self.map_sz = 800
        self.img_h = 256
        self.img_w = 306
        self.batch_sz = 2
        self.anchor_boxes = self.get_anchor_boxes()
        #self.model.load_state_dict(torch.load('/content/drive/My Drive/self_dl/pre_train_subsample/epoch_4', map_location=self.device))
        self.model = self.model.to(self.device)

    def get_anchor_boxes(self):
        widths = torch.tensor(self.scaleX)
        heights = torch.tensor(self.scaleY)
        ref_boxes = []
        for x in range(self.map_sz):
            for y in range(self.map_sz):
                x_r = widths + x
                y_r = heights + y
                x_l = torch.tensor([x, x, x, x])
                y_l = torch.tensor([y, y, y, y])
                x_r = x_r.unsqueeze(0)
                y_r = y_r.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                y_l = y_l.unsqueeze(0)
                ref_box = torch.cat((x_l, y_l, x_r, y_r))
                ref_box = ref_box.permute((1,0))
                ref_boxes.append(ref_box)
    
        anchor_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.double).to(self.device)
        
        return anchor_boxes

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def bbox_loss(self, box_targets, class_targets, out_bbox):
        inds = (class_targets != 0)
        box_targets = box_targets[inds]
        out_bbox = out_bbox[inds]
        loss_bbox = F.smooth_l1_loss(out_bbox, box_targets)

        return loss_bbox

    def train(self, epoch, save=False):
        print('Started train')
        for ep in range(epoch):
            self.model.train()
            print('Started epoch', ep)
            #total_loss = 0
            train_losses = []
            for i, (samples, class_target, box_target) in enumerate(self.trainloader):
                out_pred, out_bbox = self.model(samples.double().to(device))
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)
                out_pred = out_pred.view(self.batch_sz, 9, -1)
                # transfer coordinates
                #coor = Transform_coor(out_bbox, box_target, class_target, nms_threshold=0.1, plot=True)

                # train only for classification for now
                loss = self.bbox_loss(box_target.to(device), class_target.to(device), out_bbox)
                train_losses.append(loss.item())
          
                if loss.item() != 0:
                    self.step(loss)

                if i % 10  == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, i * len(samples), len(self.trainloader.dataset), 10. * i / len(self.trainloader), loss.item()))

                torch.cuda.empty_cache()
 
            print("\nAverage Train Epoch Loss: ", np.mean(train_losses))
            
            self.validate(ep)

        if save:
            self.validate(ep, True, False)

    
    def validate(self, epoch, save=False, visualize=False):
        self.model.eval()
        val_losses = []
        print('Started validation')
        with torch.no_grad():
            for i, (samples, class_target, box_target) in enumerate(self.trainloader):
                out_pred, out_bbox = self.model(samples.to(device))
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)
                out_pred = out_pred.view(self.batch_sz, 9, -1)
                # transfer coordinates
                #coor = Transform_coor(out_bbox, box_target, class_target, nms_threshold=0.1, plot=True)

                # train only for classification for now
                loss = self.bbox_loss(box_target.to(device), class_target.to(device), out_bbox)
                val_losses.append(loss.item())

                torch.cuda.empty_cache()

                if i % 10 == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(ep, i * len(samples), len(valloader.dataset), 5. * i / len(valloader), np.mean(val_losses)))

        print("Average Validation Epoch Loss: ", np.mean(val_losses))

        if save:
            torch.save(self.model.state_dict(), '/scratch/jd4138/dl_proj/without_pretrain/classification.pth')

        if visualize:
            Transform_coor(out_bbox, gt_offsets, class_target, nms_threshold=0.1, plot=True)



