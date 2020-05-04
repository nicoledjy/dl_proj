"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from hrnet import get_seg_model, get_config

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'DHL'
    team_number = 10
    round_number = 2
    team_member = ['Jiayi Du', 'Meiyi He', 'Ziyu Lei']
    contact_email = 'mh5275@nyu.edu'

    def __init__(self, model_file='put_your_model_file(or files)_name_here'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.batch_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model1 = get_seg_model(get_config()).to(self.device)
        self.model1.load_state_dict(torch.load('../PT_FILES/HRNET_RM_pretrain03.pt', map_location=self.device))
        pass

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        samples = samples.view(self.batch_size, -1, 256, 306)
        pred_map = self.model1(samples)
        out_map = (pred_map > 0.5).float()
        return out_map
