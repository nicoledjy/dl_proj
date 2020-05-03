
## this file will be used to train road_map using HR Net and save .pt file

import os
import random
import argparse

from collections import OrderedDict
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
from helper import collate_fn, draw_box, compute_ts_road_map, compute_ats_bounding_boxes
from hrnet import get_seg_model, get_config
from bbox_helperfunc import *


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training HRNet for predicting object bbox')
	parser.add_argument('--batch-size', type=int, default=2, metavar='N',
						help='input batch size for training (default: 2)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
						help='learning rate (default: 1e-4)')
	parser.add_argument('--weight-decay', type=float, default=1e-4,
						help='Weight decay constant (default: 1e-4)')
	parser.add_argument('--data-dir', type=str, default='../data',
						help='data directory')
	parser.add_argument('--out-file', type=str, default='test.pt',
						help='output model file name, will be stored to current directory')

	args = parser.parse_args()

	image_folder = '/scratch/mh5275/data'
	annotation_csv = '/scratch/mh5275/data/annotation.csv'
	epochs = args.epochs
	batch_size = args.batch_size
	lr = args.lr
	weight_decay = args.weight_decay
	outfile = args.out_file

	print('========================================================================================')
	print('Current Parameters:\n epochs = {},\n batch_size = {},\n lr = {},\n weight_decay = {},\n output model state_dict will stored at: {} '\
		.format(epochs, batch_size, lr, weight_decay, outfile))
	print('========================================================================================')

	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# The scenes from 106 - 133 are labeled
	# You should devide the labeled_scene_index into two subsets (training and validation)
	labeled_scene_index = np.arange(106, 134)

	# define data split
	train_index = np.arange(106,128)
	val_index = np.arange(128,134)

	transform = torchvision.transforms.ToTensor()

	# load data
	labeled_trainset = LabeledDataset(
		image_folder=image_folder,
		annotation_file=annotation_csv,
		scene_index=train_index,
		transform=transform,
		extra_info=False
		)

	labeled_valset = LabeledDataset(
		image_folder=image_folder,
		annotation_file=annotation_csv,
		scene_index=val_index,
		transform=transform,
		extra_info=False
		)

	trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
	valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

	# define model, loss function, optimizer
	model = get_seg_model(get_config()).to(device)
	for param in model.parameters():
		param.require_grad = True

	criterion = torch.nn.SmoothL1Loss()
	#param_list = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(
		[{'params': filter(lambda p: p.requires_grad, model.parameters()),
		'lr': 0.0001}],
		lr=0.0001,
		momentum=0.9,
		weight_decay=0.0001,
		nesterov=False,
		)

	best_val_loss = 100
	best_TS = -1

	epochs = 10
	for epoch in range(epochs):

		#### train logic ####
		model.train()
		train_losses = []

		
		#threat_scores = []
		for i, (sample, target, road_img) in enumerate(trainloader):

			sample = torch.stack(sample).to(device)
			batch_size = sample.shape[0]
			sample = sample.view(batch_size, -1, 256, 306) # size: ([batchsize, 18, 256, 306])
			
			label = box_to_label(target, device).float().to(device)
			optimizer.zero_grad()
			pred_label = model(sample)
			
			loss = criterion(pred_label, label)
			train_losses.append(loss.item())

			if loss.item() != 0:
				loss.backward()
				optimizer.step()
			
			if i % 50 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
					epoch, i * len(sample), len(trainloader.dataset), 10. * i / len(trainloader), loss.item()))
				
		print("\nAverage Train Epoch Loss: ", np.mean(train_losses))

		#### validation logic ####
		model.eval()
		val_losses = []
		threat_scores = []
		for i, (sample, target, road_img) in enumerate(valloader):
			sample = torch.stack(sample).to(device)
			batch_size = sample.shape[0]
			sample = sample.view(batch_size, -1, 256, 306) # size: ([batchsize, 18, 256, 306])
			label = box_to_label(target, device).float().to(device)

			with torch.no_grad():
				pred_label = model(sample)
				loss = criterion(pred_label, label)
				val_losses.append(loss.item())
				#out_label = (pred_label > 0.5).double()
				#pred_bboxes = label_to_box(out_label, device)
				pred_bboxes = label_to_box(pred_label, device)
				
				for i in range(batch_size):
					threat_scores.append( compute_ats_bounding_boxes( pred_bboxes[i], target[i]['bounding_box'] ).item() )

			if i % 50 == 0:
				print('Val Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(\
					epoch, i * len(sample), len(valloader.dataset), 5. * i / len(valloader), np.mean(val_losses)))

		print("Average Validation Epoch Loss: ", np.mean(val_losses))
		print("Average Threat Score: ", np.mean(threat_scores))

		if np.mean(val_losses) < best_val_loss:
			best_val_loss = np.mean(val_losses)
			
		if np.mean(threat_scores) > best_TS:
			best_TS = np.mean(threat_scores) 
			print('== Saving model at epoch {} with best AVG Threat Score {} =='.format(epoch, best_TS))
			print('== Current Validation Loss is {} =='.format(np.mean(val_losses)))
			torch.save(model.state_dict(), outfile)


            
