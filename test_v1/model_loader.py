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

from BoundingBox import *
from hrnet import get_seg_model, get_config


# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'DHL'
    round_number = 1
    team_member = ['Jiayi Du', 'Ziyu Lei', 'Meiyi He']
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file_name_here'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        
        self.batch_size = 1
        
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        #### model 1: predict Binary RoadMap ####
        self.model1 = get_seg_model(get_config()).to(self.device)
        self.model1.load_state_dict(torch.load('HRNET_RM_labeled_data01.pt', map_location=self.device))
        # TODO: self.model1.load_state_dict(torch.load('classification.pth', map_location=self.device))
        
        #### model 2: predict Bounding Boxes ####
        self.model2 = BoundingBox().to(self.device)
        # TODO: self.model2.load_state_dict(torch.load('classification.pth', map_location=self.device))
        pass

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        #return torch.rand(1, 15, 2, 4) * 10
        #samples = torch.stack(samples)
        samples = samples.view(self.batch_size, -1, 256, 306)
        
        # TODO: transform BBOX in the model and return BEV coordinates
        pred_bbox = self.model2(samples.to(self.device))
        
        #return pred_bbox.view(self.batch_size, -1, 4)
        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        # return torch.rand(1, 800, 800) > 0.5
        #samples = torch.stack(samples).to(self.device)
        samples = samples.view(self.batch_size, -1, 256, 306)
        pred_map = self.model1(samples)
        out_map = (pred_map > 0.5).float()

        return out_map
    
    
    def combine_images(self, batch):
        '''
        Given a single sample in size [batch_size, 6, 3, 256, 306], 
        combine them into 1 single image, keep only 1 channel, then resize to 800x800: [batch_size, 800, 800]
        
        Front_left | Front | Front_right
        -------------------------------- 
        Back_left  | Back  | Back_right
        
        and then rotate the entire image so that front is facing right
        (since ego car is always face right)
        
        '''
        imgs = []
        for sample in batch:
            # concatenating images
            front = torch.cat( (torch.tensor( sample[0] ), torch.tensor( sample[1] ), torch.tensor( sample[2] )), 2 )
            back = torch.cat( (torch.tensor( sample[3] ), torch.tensor( sample[4] ), torch.tensor( sample[5] )), 2 )

            # flip by 2 to make it face right
            curr_image = torch.cat( (front, back), 1).transpose(2,1).flip(2)

            # resize the image into 800x800 since binary road map is of size 800x800
            trans = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), 
                                        torchvision.transforms.Resize((800,800)),
                                        torchvision.transforms.ToTensor()
                                       ])
            
            # apply transformation
            combined = trans(curr_image)
            # append to the return list
            imgs.append(combined.squeeze(0))
            
            # apply transform function to make it grayscale
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5],
                    std=[0.5])
            ])
        
        return transform(imgs[0])
    
    
