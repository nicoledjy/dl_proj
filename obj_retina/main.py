import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
from torch.autograd import Variable
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200
from loss import FocalLoss
from retinanet import RetinaNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models

from data_helper import UnlabeledDataset, LabeledDataset
from helper import draw_box, collate_fn
from collections import OrderedDict
from resnet import resnet50
from obj_model import BoundingBox
from obj_trainer import ObjDetTrainer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='object detection')
  # need to add parser for image folder and annotation csv
  parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                      help='input batch size for training (default: 2)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                      help='learning rate (default: 1e-4)')
  parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay constant (default: 1e-4)')
  args = parser.parse_args()

  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)


  image_folder = '/content/drive/My Drive/self_dl/student_data/data'
  annotation_csv = '/content/drive/My Drive/self_dl/student_data/data/annotation.csv'


  train_index = np.arange(106,124)
  val_index = np.arange(124,134)


  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  torch.cuda.set_device(0)
  

  epochs = args.epochs
  lr = args.lr
  weight_decay_const = args.weight_decay
  batch_sz = args.batch_size

  transform = torchvision.transforms.ToTensor()


  labeled_trainset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=train_index,
                                    transform=transform,
                                    extra_info=True
                                   )
  trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=collate_fn)


  labeled_valset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=val_index,
                                    transform=transform,
                                    extra_info=True
                                   )

  valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=collate_fn)


  def train_obj():

      # Model
      net = RetinaNet()
      net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
      net.cuda()

      criterion = FocalLoss()
      optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

      scheduler_obj = torch.optim.lr_scheduler.LambdaLR(
              optimizer,
              lr_lambda=lambda x: (1 - x /( len(trainloader) * epochs)) ** 0.9)
      
      obj_trainer = ObjDetTrainer(net, criterion, optimizer, scheduler_obj, trainloader, valloader, device)
      obj_trainer.train(epochs, True)

  train_obj()






