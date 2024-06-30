import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import Dataset
from typing import Tuple, Sequence


class BloodDataset(Dataset):
    def __init__(
            self,
            hyperspectral_data_path: str,
            ground_truth_data_path: str,
    ):
        
        hyperspectral_data_files = os.listdir(hyperspectral_data_path)

        hyperspectral_files = [f for f in hyperspectral_data_files if os.path.isfile(os.path.join(hyperspectral_data_path, f))]

        float_files = []
        hdr_files = []

        for file in hyperspectral_files:
            if file.endswith('.float'):
                float_files.append(file)
            elif file.endswith('.hdr'):
                hdr_files.append(file)

        float_files = sorted(float_files)
        hdr_files = sorted(hdr_files)

        self.pixel_values = []

        if len(float_files) == len(hdr_files):
            number_of_images = len(float_files)
            
        for i in range(number_of_images):

            img = envi.open(f'{hyperspectral_data_path}/{hdr_files[i]}', f'{hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)
            number_of_pixels = img.shape[0] * img.shape[1]
            
            img = img.reshape(number_of_pixels, img.shape[2])

            self.pixel_values.append(img)

        self.pixel_values = np.concatenate(self.pixel_values)
        self.pixel_values = self.pixel_values/self.pixel_values.max()
        self.pixel_values = torch.Tensor(self.pixel_values)

        ground_truth_data_files = os.listdir(ground_truth_data_path)

        ground_truth_files = [f for f in ground_truth_data_files if os.path.isfile(os.path.join(ground_truth_data_path, f))]
        ground_truth_files = sorted(ground_truth_files)

        self.ground_truth = []

        for i in range(number_of_images):

            gt = np.load(f'{ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt[gt > 7] = 0
            number_of_pixels = gt.shape[0] * gt.shape[1]
            gt = gt.reshape(number_of_pixels, 1)

            self.ground_truth.append(gt)

        self.ground_truth = np.concatenate(self.ground_truth)
        self.ground_truth = torch.Tensor(self.ground_truth)

    def  __len__(self):
        return len(self.pixel_values)
    
    def __getitem__(self, index: int) -> Tuple[Sequence, int]:

        pixel_values = self.pixel_values[index]
        pixel_values = pixel_values.reshape(1, len(pixel_values))

        label = self.ground_truth[index]
        label = self.onehot_encoding(int(label.item()))
        
        return pixel_values, label
    
    def onehot_encoding(self, label: torch.TensorType) -> Sequence[int]:
        onehot_label = torch.zeros(len(torch.unique(self.ground_truth)))
        onehot_label[label] = 1.0
        return onehot_label