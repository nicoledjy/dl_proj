import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

import os
import random

import numpy as np
import pandas as pd

import matplotlib

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch.nn.functional as F


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box



class Autoencoder(nn.Module):
	def __init__(self):
        super().__init__()
        d = 30
        self.encoder = nn.Sequential(
        	nn.Linear(800 * 800 *3, d),
        	nn.Tanh(),
        	)
        self.decoder = nn.Sequential(
        	nn.Linear(d, 800 * 800 * 3),
        	nn.Tanh(),
        	)
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


def my_collate_fn(batch):
	imgs = []
	for x in batch:
		front = torch.cat( (torch.tensor( x[0] ), torch.tensor( x[1] ), torch.tensor( x[2] )), 2 )
		back = torch.cat( (torch.tensor( x[3] ), torch.tensor( x[4] ), torch.tensor( x[5] )), 2 )
		curr_image = torch.cat( (front, back), 1)
        
		trans = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((800,800)),
			transforms.ToTensor()])
		comb = trans(curr_image)
		imgs.append(comb)

	return torch.stack(imgs)


if __name__ == '__main__':
	image_folder = '../data'
	annotation_csv = '../data/annotation.csv'
	
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);

	unlabeled_scene_index = np.arange(106)
	unlabeled_scene_index_shuf = unlabeled_scene_index
	random.shuffle(unlabeled_scene_index_shuf)

	# split trian/val datasets
	train_unlabeled_scene_index = unlabeled_scene_index_shuf[:-25]
	val_unlabeled_scene_index = unlabeled_scene_index_shuf[-25:]

	transform = torchvision.transforms.ToTensor()

	# train loader
	unlabeled_trainset = UnlabeledDataset(
		image_folder=image_folder,
		scene_index=train_unlabeled_scene_index, 
		first_dim='sample', 
		transform=transform)

	train_loader = torch.utils.data.DataLoader(
		unlabeled_trainset,
		batch_size=2,
		shuffle=True,
		num_workers=2,
		collate_fn=my_collate_fn)

	# val loader
	unlabeled_valset = UnlabeledDataset(
		image_folder=image_folder,
		scene_index=val_unlabeled_scene_index, 
		first_dim='sample',
		transform=transform)

	val_loader = torch.utils.data.DataLoader(
		unlabeled_valset,
		batch_size=2,
		shuffle=True,
		num_workers=2,
		collate_fn=my_collate_fn)


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = Autoencoder().to(device)
	criterion = nn.MSELoss()
	for param in model.parameters():
		param.requires_grad = True

	learning_rate = 1e-3
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=learning_rate,
		)

	# Training and testing the VAE
	epochs = 10
	best_val_loss = 100

	for epoch in range(0, epochs + 1):
		# Training
		model.train()
		train_loss = 0
		for i, sample in enumerate(train_loader):
			sample = sample.to(device)
			sample = sample.view(sample.size(0), -1)
			
			# ===================forward=====================
			x_hat = model(sample)
			loss = criterion(sample, x_hat)
			train_loss += loss.item()
			
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# ===================log========================
			if i % 1000  == 0:
				avg_loss = float(train_loss / (i+1))
				print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

		avg_loss = float(train_loss / len(train_loader))
		print('Trained Epoch {} | Average Train Loss: {}'.format(epoch, avg_loss))

		# Testing
		with torch.no_grad():
			model.eval()
			val_loss = 0

			for i,sample in enumerate(val_loader):
				sample = sample.to(device)
				sample = sample.view(sample.size(0), -1)

				# ===================forward=====================
				x_hat = model(sample)
				loss = criterion(sample, x_hat)
				val_loss += loss.item()

				# ===================log========================
				if i % 1000  == 0:
					avg_loss = float(val_loss / (i+1))
					print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))

			avg_loss = float(val_loss / len(val_loader))
			print('Validation Epoch {} | Average Validation Loss: {}'.format(epoch, avg_loss))

		if avg_loss < best_val_loss:
			best_val_loss = avg_loss
			torch.save(model.state_dict(), 'best_AE_pretrain_01.pt')


