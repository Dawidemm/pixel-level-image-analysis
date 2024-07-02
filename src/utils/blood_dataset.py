import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import Dataset
from typing import Tuple, Sequence

def blood_pixel_generator(
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

    if len(float_files) == len(hdr_files):
        number_of_images = len(float_files)

    float_files = sorted(float_files)
    hdr_files = sorted(hdr_files)

    ground_truth_data_files = os.listdir(ground_truth_data_path)

    ground_truth_files = [f for f in ground_truth_data_files if os.path.isfile(os.path.join(ground_truth_data_path, f))]
    ground_truth_files = sorted(ground_truth_files)

    for i in range(number_of_images):

            img = envi.open(f'{hyperspectral_data_path}/{hdr_files[i]}', f'{hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)

            gt = np.load(f'{ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt[gt > 7] = 0

            if img.shape[:2] == gt.shape:
                rows, cols, bands = img.shape

            for row in range(rows):
                for col in range(cols):
                    yield torch.tensor(img[row, col]), torch.tensor(gt[row, col])

def blood_dataset_params(
        hyperspectral_data_path: str,
        ground_truth_data_path: str,
):
    pixel_max_value = 0
    dataset_length = 0
    classes = 0

    hyperspectral_data_files = os.listdir(hyperspectral_data_path)

    hyperspectral_files = [f for f in hyperspectral_data_files if os.path.isfile(os.path.join(hyperspectral_data_path, f))]

    float_files = []
    hdr_files = []

    for file in hyperspectral_files:
        if file.endswith('.float'):
            float_files.append(file)
        elif file.endswith('.hdr'):
            hdr_files.append(file)

    if len(float_files) == len(hdr_files):
        number_of_images = len(float_files)

    float_files = sorted(float_files)
    hdr_files = sorted(hdr_files)

    ground_truth_data_files = os.listdir(ground_truth_data_path)

    ground_truth_files = [f for f in ground_truth_data_files if os.path.isfile(os.path.join(ground_truth_data_path, f))]
    ground_truth_files = sorted(ground_truth_files)

    for i in range(number_of_images):

            img = envi.open(f'{hyperspectral_data_path}/{hdr_files[i]}', f'{hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)
            rows, cols, bands = img.shape

            if img.max() >= pixel_max_value:
                pixel_max_value = img.max()

            dataset_length += rows * cols

            gt = np.load(f'{ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt[gt > 7] = 0

            if len(np.unique(gt)) >= classes:
                classes = len(np.unique(gt))

    return pixel_max_value, dataset_length, classes    

class BloodDataset(Dataset):
    def __init__(
            self,
            hyperspectral_data_path: str,
            ground_truth_data_path: str,
    ):
        self.pixel_max_value, self.dataset_length, self.classes = blood_dataset_params(
            hyperspectral_data_path=hyperspectral_data_path,
            ground_truth_data_path=ground_truth_data_path
        )

        self.blood_generator = blood_pixel_generator(
            hyperspectral_data_path=hyperspectral_data_path,
            ground_truth_data_path=ground_truth_data_path
        )

    def  __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index: int) -> Tuple[Sequence, int]:

        pixel, label = next(self.blood_generator)
        pixel = pixel/self.pixel_max_value
        pixel = pixel.reshape(1, pixel.shape[0])
        label = self.onehot_encoding(int(label.item()))
        
        return pixel, label
    
    def onehot_encoding(self, label: torch.TensorType) -> Sequence[int]:
        onehot_label = torch.zeros(self.classes)
        onehot_label[label] = 1.0
        return onehot_label