import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.models as models

from data_helper import UnlabeledDataset, LabeledDataset
from collections import OrderedDict
from resnet import resnet50
from obj_model import BoundingBox
from obj_trainer import ObjDetTrainer, my_collate_fn
from pretrain.models import pirl_resnet
from pretrain.network_helpers import copy_weights_between_models

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
# image_folder = '../data'
# annotation_csv = '../data/annotation.csv'
image_folder = '/content/drive/My Drive/self_dl/student_data/data'
annotation_csv = '/content/drive/My Drive/self_dl/student_data/data/annotation.csv'
pirl_file_path = '/content/drive/My Drive/self_dl/pre_train_subsample/epoch_1s'
pretrained_res = True





train_index = np.arange(106,124)
val_index = np.arange(124,134)


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


epochs = 10
batch_sz = 2

transform = torchvision.transforms.ToTensor()

labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_index,
                                  transform=transform,
                                  extra_info=True
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=my_collate_fn)

labeled_valset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=val_index,
                                  transform=transform,
                                  extra_info=True
                                 )

valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=my_collate_fn)




def train_obj():
    epochs = 10
    model = BoundingBox('res50')
    if pretrained_res:
      pirl_model = pirl_resnet('res50')
      pirl_model.load_state_dict(torch.load(pirl_file_path, map_location=device))
      weight_copy_success = copy_weights_between_models(model, pirl_model)

      if not weight_copy_success:
        print ('Weight copy between SSL and classification net failed. Pls check !!')
        exit()

    # # Freeze all layers except fully connected in classification net
    # for name, param in model.named_parameters():
    #     if name[:7] == 'resnet_':
    #         param.requires_grad = False

    model.double().to(device)
    param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(param_list, lr=1e-4, weight_decay=1e-4)
    scheduler_obj = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: (1 - x /( len(trainloader) * epochs)) ** 0.9)
    

    obj_trainer = ObjDetTrainer(model, optimizer, scheduler_obj, trainloader, valloader, device)
    obj_trainer.train(epochs, True)

train_obj()
