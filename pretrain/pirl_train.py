import os
import argparse

import torch
import torchvision

import numpy as np
import pandas as pd

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler
from dataset_helpers import set_random_generators_seed
from get_dataset import GetUnlabeledDataForPIRL
from models import pirl_resnet
from train_test_helper import PIRLModelTrainTest

# log_save_folder = '/content/drive/My Drive/self_dl/log_data'
# model_file_path = '/content/drive/My Drive/self_dl/pre_train_subsample/'
log_save_folder = '/scratch/jd4138/dl_log_data'
model_file_path = '/scratch/jd4138/dl_pretrain/'

def log_experiment(exp_name, n_epochs, train_losses, val_losses, train_accs, val_accs):
    observations_df = pd.DataFrame()
    observations_df['epoch count'] = [i for i in range(1, n_epochs + 1)]
    observations_df['train loss'] = train_losses
    observations_df['val loss'] = val_losses
    observations_df['train acc'] = train_accs
    observations_df['val acc'] = val_accs
    observations_file_path = os.path.join(log_save_folder, exp_name + '_observations.csv')
    observations_df.to_csv(observations_file_path)


if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='Pre train script for PIRL task')
    parser.add_argument('--num-scene', type=int, default=106, help='number of scenes')
    parser.add_argument('--model-type', type=str, default='res18', help='backbone architecture')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--tmax-for-cos-decay', type=int, default=70)
    parser.add_argument('--warm-start', type=bool, default=False)
    parser.add_argument('--count-negatives', type=int, default=6400,
                        help='No of samples in memory bank of negatives')
    parser.add_argument('--beta', type=float, default=0.5, help='Exponential running average constant'
                                                                'in memory bank update')
    parser.add_argument('--non-linear-head', type=bool, default=False,
                        help='If true apply non-linearity to the output of function heads '
                             'applied to resnet image representations')
    parser.add_argument('--temp-parameter', type=float, default=0.07, help='Temperature parameter in NCE probability')
    parser.add_argument('--cont-epoch', type=int, default=0, help='Epoch to start the training from, helpful when using'
                                                                  'warm start')
    parser.add_argument('--experiment-name', type=str, default='e1_resnet50_')
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define train_set, val_set objects
    scene_index = np.arange(args.num_scene)
    #image_folder = '/content/drive/My Drive/self_dl/student_data/data'
    image_folder = '/scratch/jd4138/data'
    train_set = GetUnlabeledDataForPIRL(image_folder=image_folder, scene_index=scene_index, first_dim='image')
    val_set = GetUnlabeledDataForPIRL(image_folder=image_folder, scene_index=scene_index, first_dim='image')


    # Define train and validation data loaders
    len_train_set = len(train_set)
    train_indices = list(range(len_train_set))
    np.random.shuffle(train_indices)


    train_indices = train_indices[:len_train_set]
    val_indices = train_indices[len_train_set:]


    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                             num_workers=8)

    # Train required model using data loaders defined above
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    # If using Resnet18
    model_to_train = pirl_resnet(args.model_type, args.non_linear_head)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = CosineAnnealingLR(sgd_optimizer, args.tmax_for_cos_decay, eta_min=1e-4, last_epoch=-1)

    # Initialize model weights with a previously trained model if using warm start
    if args.warm_start and os.path.exists(model_file_path):
        model_to_train.load_state_dict(torch.load(model_file_path, map_location=device))

    # Start training
    all_images_mem = np.random.randn(len_train_set, 128)
    model_train_test_obj = PIRLModelTrainTest(
        model_to_train, device, model_file_path, all_images_mem, train_indices, val_indices, args.count_negatives,
        args.temp_parameter, args.beta
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(args.cont_epoch, args.cont_epoch + epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            sgd_optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_loader, val_data_loader=val_loader,
            no_train_samples=len(train_indices), no_val_samples=len(val_indices)
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()

    # Log train-test results
    log_experiment(args.experiment_name, args.epochs, train_losses, val_losses, train_accs, val_accs)
