import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset_helpers import get_nine_crops, pirl_full_img_transform, pirl_stl10_jigsaw_patch_transform

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]


class GetUnlabeledDataForPIRL(Dataset):
    'Characterizes PyTorch Dataset object'
    def __init__(self, image_folder, scene_index, first_dim):
        'Initialization'
        self.image_folder = image_folder
        self.scene_index = scene_index
        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        'Denotes the total number of samples'
        if self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
            
            # add the convert part
            original = Image.open(image_path)
            image_tensor = pirl_full_img_transform(original)
            nine_crops = get_nine_crops(original)

            # Form the jigsaw order for this image
            original_order = np.arange(9)
            permuted_order = np.copy(original_order)
            np.random.shuffle(permuted_order)


            # Permut the 9 patches obtained from the image
            permuted_patches_arr = [None] * 9
            for patch_pos, patch in zip(permuted_order, nine_crops):
                permuted_patches_arr[patch_pos] = patch

            # Apply data transforms
            # TODO: Remove hard coded values from here
            tensor_patches = torch.zeros(9, 3, 30, 30)
            for ind, patch in enumerate(permuted_patches_arr):
                patch_tensor = pirl_stl10_jigsaw_patch_transform(patch)
                tensor_patches[ind] = patch_tensor

            return [image_tensor, tensor_patches], index


