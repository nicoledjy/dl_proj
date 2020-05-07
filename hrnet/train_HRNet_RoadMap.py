
## this file will be used to train road_map using HR Net and save .pt file

import os
import random

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
from helper import collate_fn, draw_box, compute_ts_road_map
from hrnet import get_seg_model, get_config


if __name__ == '__main__':

	image_folder = '../data'
	annotation_csv = '../data/annotation.csv'
	
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# The scenes from 106 - 133 are labeled
	# You should devide the labeled_scene_index into two subsets (training and validation)
	labeled_scene_index = np.arange(106, 134)

	train_index = np.arange(106,108)
	val_index = np.arange(128,130)

	transform = torchvision.transforms.ToTensor()

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

	model = get_seg_model(get_config()).to(device)

	# for param in model.parameters():
	# 	param.requires_grad = True

	criterion = torch.nn.BCELoss()
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

	epochs = 10
	for epoch in range(epochs):

		#### train logic ####
		model.train()
		train_losses = []

		for i, (sample, target, road_img) in enumerate(trainloader):
			

			sample = torch.stack(sample).to(device)
			batch_size = sample.shape[0]
			sample = sample.view(batch_size, -1, 256, 306) # size: ([3, 18, 256, 306])
			road_img = torch.stack(road_img).float().to(device)

			optimizer.zero_grad()
			pred_map = model(sample)
			
			loss = criterion(pred_map, road_img)
			train_losses.append(loss.item())
			loss.backward()
			optimizer.step()
			
			if i % 100 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, i * len(sample), len(trainloader.dataset),
					100. * i / len(trainloader), loss.item()))
		print("\n Average Train Epoch Loss for epoch {} is {} ", epoch+1, np.mean(train_losses))

		#### validation logic ####
		model.eval()
		val_losses = []
		threat_score = []
		for i, (sample, target, road_img) in enumerate(valloader):
			sample = torch.stack(sample).to(device)
			batch_size = sample.shape[0]
			sample = sample.view(batch_size, -1, 256, 306) # size: ([3, 18, 256, 306])
			road_img = torch.stack(road_img).float().to(device)

			with torch.no_grad():
				pred_map = model(sample)				
				loss = criterion(pred_map, road_img)
				val_losses.append(loss.item())

				out_map = (pred_map > 0.5).float()
				threat_score.append(compute_ts_road_map(out_map, road_img).item())

			print("Validation Epoch: {}, Average Validation Epoch Loss: {}".format(epoch, np.mean(val_losses)))
			print("Average Threat Score: {} ".format(np.mean(threat_score)))

			if np.mean(val_losses) < best_val_loss:
				best_val_loss = np.mean(val_losses)
				torch.save(model.state_dict(), 'test.pt')

