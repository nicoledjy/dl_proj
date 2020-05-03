import os.path
import pandas as pd
import numpy as np
import math
import scipy.ndimage as ndimage
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from skimage import measure




### batched version ###
def box_to_label(target, device):
	'''
	Given a batch of target, return a batch of label map with size 800*800
	'''
	batched_labels = []
	for t in target:
		bboxes = t['bounding_box'].to(device)
		output = np.zeros((800,800))

		for i in range(len(bboxes)):
			class_label = 1 # always 1 for object
			this_bbox = bboxes[i]
			flx, frx, blx, brx = this_bbox[0]
			fly, fry, bly, bry = this_bbox[1]
			fx = math.floor(10*((flx + frx)/2) + 400)
			bx = math.floor(10*((blx + brx)/2) + 400)
			fy = math.floor(10*((fly + bly)/2) + 400)
			by = math.floor(10*((fry + bry)/2) + 400)

			# fill in 1's for object's position
			output[fy:by, fx:bx] = class_label
			output[by:fy, bx:fx] = class_label
		
		# append to batched labels
		batched_labels.append(torch.from_numpy(output))
	return torch.stack(batched_labels)


### batched version ###
def label_to_box(label, device):
	'''
	Given a batch of predicted label, 
	return a list of bbox coordinates for each one in the batch
	'''
	label = label.data.cpu().numpy()
	batched_coor = []
	for l in label:
		
		test_label = measure.label(l)
		output = test_label.copy()
		bboxes = []

		props = measure.regionprops(test_label)
		for prop in props:
			fy,fx,by,bx = prop.bbox
			fy, fx, by, bx = [min(fy,799), min(fx,799), min(by, 799), min(bx, 799)]
			flx, frx, blx, brx, fly, bly, fry, bry = (fx, fx, bx, bx, fy, fy, by, by)

			output[fy:by, fx-1:fx+1] = 50
			output[fy:by, bx-1:bx+1] = 50
			output[fy-1:fy+1, fx:bx] = 50
			output[by-1:by+1, fx:bx] = 50

			this_bbox = np.array([[flx, frx, blx, brx], [fly, fry, bly, bry]])

			this_bbox = (this_bbox - 400)/10
			bboxes.append(torch.as_tensor(this_bbox))
		#print('bboxes {}'.format(bboxes))
		if not bboxes:
			bboxes = torch.zeros(1,2,4)
			batched_coor.append(bboxes)
		else:
			batched_coor.append(torch.stack(bboxes))
	return batched_coor


#####################################################################################################################

# def bbox_to_label(target_object):
#     categories = target_object[0]['category']
#     bboxes = target_object[0]['bounding_box']

#     output = np.zeros((800,800))
#     #print(len(categories))

#     for i in range(len(bboxes)):
#         class_label = 1 #categories[i]
#         this_bbox = bboxes[i]
#         flx, frx, blx, brx = this_bbox[0]
#         fly, fry, bly, bry = this_bbox[1]
#         fx = math.floor(10*((flx + frx)/2) + 400)
#         bx = math.floor(10*((blx + brx)/2) + 400)
#         fy = math.floor(10*((fly + bly)/2) + 400)
#         by = math.floor(10*((fry + bry)/2) + 400)

#         #output[fx:bx, fy:by] = class_label
#         #output[bx:fx, by:fy] = class_label

#         output[fy:by, fx:bx] = class_label
#         output[by:fy, bx:fx] = class_label

#     return output

def get_bboxes_from_output(model_output): #v2
    test_label = measure.label(model_output)
    output = test_label.copy()
    bboxes = []

    props = measure.regionprops(test_label)

    for prop in props:
        fy,fx,by,bx = prop.bbox
        fy, fx, by, bx = [min(fy,799), min(fx,799), min(by, 799), min(bx, 799)]
        flx, frx, blx, brx, fly, bly, fry, bry = (fx, fx, bx, bx, fy, fy, by, by)

        output[fy:by, fx-1:fx+1] = 50
        output[fy:by, bx-1:bx+1] = 50
        output[fy-1:fy+1, fx:bx] = 50
        output[by-1:by+1, fx:bx] = 50

        this_bbox = np.array([[flx, frx, blx, brx], [fly, fry, bly, bry]])
        this_bbox = (this_bbox - 400)/10
        bboxes.append(this_bbox)

    return torch.tensor(bboxes)

# def bbox_to_label_bionary(target_object):
#     categories = target_object[0]['category']
#     bboxes = target_object[0]['bounding_box']

#     output = np.zeros((800,800))

#     for i in range(len(bboxes)):
#         #class_label = categories[i]
#         this_bbox = bboxes[i]
#         flx, frx, blx, brx = this_bbox[0]
#         fly, fry, bly, bry = this_bbox[1]
#         fx = math.floor(10*((flx + frx)/2) + 400)
#         bx = math.floor(10*((blx + brx)/2) + 400)
#         fy = math.floor(10*((fly + bly)/2) + 400)
#         by = math.floor(10*((fry + bry)/2) + 400)

#         output[fy:by, fx:bx] = 1
#         output[by:fy, bx:fx] = 1

#     return output

def frankenstein(image_object):
    this_image = image_object[0]
    front = torch.cat((this_image[0], this_image[1], this_image[2]), 2)
    back = torch.cat((this_image[5], this_image[4], this_image[3]), 2)
    all_images = torch.cat((front, back), 1)
    all_images = all_images.unsqueeze(0)

    return all_images
