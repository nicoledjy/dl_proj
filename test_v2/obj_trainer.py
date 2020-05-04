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
from anchors import get_bbox_gt, batched_coor_threat_updated
from helper import compute_ats_bounding_boxes


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
        self.model.load_state_dict(torch.load('/scratch/jd4138/bbox_no_pretrain_50.pt', map_location=self.device))
        self.model = self.model.to(self.device)
        self.best_val_loss = 100

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


    def get_targets(self, target, sample):
        batched_preds = []
        batched_offsets = []
        for t, s in zip(target, sample):
            bboxes = t['bounding_box'].to(self.device)
            gt_classes, gt_offsets = get_bbox_gt(bboxes, t['category'].to(self.device), self.anchor_boxes.to(self.device), self.map_sz, self.device)
            batched_preds.append(gt_classes)
            batched_offsets.append(gt_offsets)

        class_targets = torch.stack(batched_preds)
        box_targets = torch.stack(batched_offsets)

        return class_targets, box_targets


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
            train_losses = []
            for i, (sample, target, road_image, extra) in enumerate(self.trainloader):
                samples = torch.stack(sample).to(self.device).double()
                class_target, box_target = self.get_targets(target, sample)
                out_pred, out_bbox = self.model(samples)
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)

                loss = self.bbox_loss(box_target, class_target, out_bbox)
                train_losses.append(loss.item())
          
                if loss.item() != 0:
                  self.step(loss)

                if i % 10  == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, i * len(samples), len(self.trainloader.dataset), 10. * i / len(self.trainloader), loss.item()))
                  #self.validate(ep) 
                    
                torch.cuda.empty_cache()
 
            print("\nAverage Train Epoch Loss: ", np.mean(train_losses))
            self.validate(ep,True,False)


    
    def validate(self, epoch, save=False, visualize=False):
        self.model.eval()
        val_losses = []
        threat_scores = []
        print('Started validation')
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.valloader):
                samples = torch.stack(sample).to(self.device).double()
                class_target, box_target = self.get_targets(target, sample)
                out_pred, out_bbox = self.model(samples)
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)

                target1 = target[0]['bounding_box'].numpy()
                target2 = target[1]['bounding_box'].numpy()
                target = [target1, target2]

                final_coor, batched_threat_sum = batched_coor_threat_updated(i, out_bbox, self.anchor_boxes, target, class_target, self.batch_sz, nms_threshold=0.1, plot=False)
                threat_scores.append(batched_threat_sum.item()/2)

                loss = self.bbox_loss(box_target, class_target, out_bbox)
                val_losses.append(loss.item())

                torch.cuda.empty_cache()

                if i % 20 == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(epoch, i * len(samples), len(self.valloader.dataset), 5. * i / len(self.valloader), np.mean(val_losses)))
                
        print("Average Validation Epoch Loss: ", np.mean(val_losses))
        print("Average Threat Score: ", np.mean(threat_scores))

        if save and np.mean(val_losses) < self.best_val_loss:
            self.best_val_loss = np.mean(val_losses)
            torch.save(self.model.state_dict(), '/scratch/jd4138/bbox_no_pretrain_resume_50.pt')

        # if visualize:
        #     Transform_coor(out_bbox, gt_offsets, class_target, nms_threshold=0.1, plot=True)



