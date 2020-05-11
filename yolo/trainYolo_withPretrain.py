

## this file will be used to train bbox using yolo and save .pt file

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
from helper import collate_fn, draw_box, compute_ts_road_map, compute_ats_bounding_boxes


from yolo import *


if __name__ == '__main__':

	image_folder = '/scratch/mh5275/data'
	annotation_csv = '/scratch/mh5275/data/annotation.csv'
	
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);

	epochs = 30
	batch_size = 2
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# The scenes from 106 - 133 are labeled
	# You should devide the labeled_scene_index into two subsets (training and validation)
	labeled_scene_index = np.arange(106, 134)

	train_index = np.arange(106,128)
	val_index = np.arange(128,134)



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

	model = Darknet(num_classes = 10, encoder_features = 6).to(device)
	model.init_weights('/scratch/mh5275/AE_pretrain_new_01.pt')

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
	best_threat = 0

	for ep in range(epochs):

		#### train logic ####
		model.train()
		train_losses = []

		for i, (sample, target, road_img) in enumerate(trainloader):

			sample = torch.stack(sample).to(device)
			optimizer.zero_grad()
			pred_map, loss = model(sample.to(device), process_target(target).to(device))
			train_losses.append(loss.item())
			if loss.item() != 0:
				loss.backward()
				optimizer.step()

			if i % 50  == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, i * len(sample), len(trainloader.dataset), 10. * i / len(trainloader), loss.item()))

		print("\nAverage Train Epoch Loss: ", np.mean(train_losses))

		#### validation logic ####
		model.eval()
		val_losses = []
		threat_score = []
		for i, (sample, target, road_img) in enumerate(valloader):
			
			sample = torch.stack(sample).to(device)
			
			with torch.no_grad():
				pred_map, loss = model(sample.to(device), process_target(target).to(device))
				predicted_bounding_boxes = model.get_bounding_boxes(sample.to(device))
				val_losses.append(loss.item())

				for j in range(batch_size):
					score = compute_ats_bounding_boxes( predicted_bounding_boxes[0][j].to(device), target[j]['bounding_box'].to(device) )
					threat_score.append(score.item())

			if i % 50  == 0:
				print('Val Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(ep, i * len(sample), len(valloader.dataset), 5. * i / len(valloader), np.mean(val_losses)))

		print("Average Validation Epoch Loss: ", np.mean(val_losses))
		print("Average Threat Score: {} ".format(np.mean(threat_score)))
		if np.mean(val_losses) < best_val_loss:
			best_val_loss = np.mean(val_losses)
			torch.save(model.state_dict(), 'best_val_yolo02_withPre.pt')

		if np.mean(threat_score) > best_threat:
			best_threat = np.mean(threat_score)
			torch.save(model.state_dict(), 'best_threat_yolo02_withPre.pt')




