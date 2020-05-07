import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import os
import random

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box


class Autoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.enc1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
		self.enc2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)

		self.dec1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3)
		self.dec2 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3)

	def forward(self, x):
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		return x


if __name__ == '__main__':
	image_folder = '/scratch/mh5275/data'
	annotation_csv = '/scratch/mh5275/data/annotation.csv'
	
	NUM_EPOCHS = 50
	LEARNING_RATE = 1e-3
	BATCH_SIZE = 5

	print(' == Parameters: NUM_EPOCHS {}, LEARNING_RATE {}, BATCH_SIZE {} == '.format(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE))
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);

	unlabeled_scene_index = np.arange(106)
	
	# split trian/val datasets
	train_index = unlabeled_scene_index[:-25]
	val_index = unlabeled_scene_index[-25:]

	transform = torchvision.transforms.ToTensor()

	# train loader
	unlabeled_trainset = UnlabeledDataset(
		image_folder=image_folder,
		scene_index=train_index, 
		first_dim='image', 
		transform=transform)

	train_loader = torch.utils.data.DataLoader(
		unlabeled_trainset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2)

	# val loader
	unlabeled_valset = UnlabeledDataset(
		image_folder=image_folder,
		scene_index=val_index, 
		first_dim='image',
		transform=transform)

	val_loader = torch.utils.data.DataLoader(
		unlabeled_valset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Autoencoder().to(device)
	criterion = nn.MSELoss()
	
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=LEARNING_RATE,
		)

	# Training and testing the VAE
	epochs = NUM_EPOCHS
	best_val_loss = 100

	for epoch in range(0, epochs + 1):
		# Training
		model.train()
		train_loss = 0
		for i, (sample, _) in enumerate(train_loader):
			sample = sample.to(device)
			
			# ===================forward=====================
			x_hat = model(sample)
			loss = criterion(sample, x_hat)
			train_loss += loss.item()
			
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# ===================log========================
			# if i % 1000  == 0:
			# 	avg_loss = float(train_loss / (i+1))
			# 	print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

		avg_loss = float(train_loss / len(train_loader))
		print('Trained Epoch {} | Average Train Loss: {}'.format(epoch, avg_loss))

		# Testing
		with torch.no_grad():
			model.eval()
			val_loss = 0

			for i, (sample, _) in enumerate(val_loader):
				sample = sample.to(device)

				# ===================forward=====================
				x_hat = model(sample)
				loss = criterion(sample, x_hat)
				val_loss += loss.item()

				# ===================log========================
				# if i % 1000  == 0:
				# 	avg_loss = float(val_loss / (i+1))
				# 	print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

			avg_loss = float(val_loss / len(val_loader))
			print('Validation Epoch {} | Average Validation Loss: {}'.format(epoch, avg_loss))

		if avg_loss < best_val_loss:
			best_val_loss = avg_loss
			torch.save(model.state_dict(), 'AE_pretrain_new_01.pt')


