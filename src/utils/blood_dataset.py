import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import IterableDataset

from typing import Sequence, Union, List
from enum import Enum


NOISY_BANDS_INDICES = np.array([0, 1, 2, 3, 4, 5, 48, 49, 50, 121, 122, 123, 124, 125, 126, 127])
BACKGROUND_VALUE = 0

class Stage(Enum):
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'


def blood_dataset_params(
        hyperspectral_data_path: str,
        ground_truth_data_path: str,
        remove_noisy_bands: bool=True
):
    pixel_max_value = 0
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

        if remove_noisy_bands:
            img = np.delete(img, NOISY_BANDS_INDICES, axis=2)

        if img.max() >= pixel_max_value:
            pixel_max_value = img.max()

        gt = np.load(f'{ground_truth_data_path}/{ground_truth_files[i]}')
        gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
        gt[gt > 7] = 0

        if len(np.unique(gt)) >= classes:
            classes = len(np.unique(gt))

    return pixel_max_value, classes    


class BloodIterableDataset(IterableDataset):
    def __init__(
            self,
            hyperspectral_data_path: str,
            ground_truth_data_path: str,
            load_specific_images: Union[List[str], None]=None,
            remove_noisy_bands: bool=True,
            remove_background: bool=False,
            stage = Stage,
            shuffle = bool,
    ):  
        '''
        A PyTorch IterableDataset for loading and processing hyperspectral images and their corresponding ground truth data. 
        The dataset provides an iterable over pixel-label pairs, supporting different processing stages (TRAIN, VALIDATE, TEST)
        and optional background removal and shuffling.

        Parameters:
        - hyperspectral_data_path (str): Path to the folder containing hyperspectral image files (.hdr and .float format).
        - ground_truth_data_path (str): Path to the folder containing ground truth data files (.npz format).
        - load_specific_images (Union[List[str], None], optional): List of specific images to load (without extension).
        If None, all available images will be loaded. Default is None.
        - remove_noisy_bands (bool, optional): Flag indicating whether noisy spectral bands should be removed. Default is True.
        - remove_background (bool, optional): If True, pixels corresponding to background in the ground truth are excluded.
        Default is False.
        - stage (Stage, optional): Specifies the processing stage (TRAIN, VALIDATE, or TEST), which determines how the data is split.
        Default is Stage.TRAIN.
        - shuffle (bool, optional): Flag indicating whether to shuffle the dataset. Default is False.

        Attributes:
        - pixel_max_value (float): Maximum pixel value from hyperspectral data used for normalization.
        - classes (int): The number of unique classes in the ground truth data for one-hot encoding.

        Methods:
        - __iter__(): An iterable that yields pairs of normalized pixel data (torch.Tensor) and ground truth labels.
        Labels are one-hot encoded during training and validation, and background pixels can be optionally removed.
        The dataset is shuffled if specified.
        - onehot_encoding(label: torch.TensorType) -> Sequence[int]: Converts a scalar label into a one-hot encoded vector.

        Data Splitting:
        - Training: The first 75% of the dataset is used for training.
        - Validation: 15% of the dataset is used for validation (from 75% to 90%).
        - Test: The remaining 10% of the dataset is used for testing.
        '''
        np.random.seed(100)

        self.hyperspectral_data_path = hyperspectral_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.load_specific_images = load_specific_images
        self.remove_noisy_bands = remove_noisy_bands
        self.remove_background = remove_background
        self.stage = stage
        self.shuffle = shuffle

        self.pixel_max_value, self.classes = blood_dataset_params(
            hyperspectral_data_path=hyperspectral_data_path,
            ground_truth_data_path=ground_truth_data_path,
            remove_noisy_bands=self.remove_noisy_bands
        )

    def __iter__(self):

        if self.load_specific_images is None:
            hyperspectral_data_files = os.listdir(self.hyperspectral_data_path)
            hyperspectral_files = [f for f in hyperspectral_data_files if os.path.isfile(os.path.join(self.hyperspectral_data_path, f))]

            float_files = []
            hdr_files = []

            for file in hyperspectral_files:
                if file.endswith('.float'):
                    float_files.append(file)
                elif file.endswith('.hdr'):
                    hdr_files.append(file)

            float_files = sorted(float_files)
            hdr_files = sorted(hdr_files)

            ground_truth_data_files = os.listdir(self.ground_truth_data_path)

            ground_truth_files = [f for f in ground_truth_data_files if os.path.isfile(os.path.join(self.ground_truth_data_path, f))]
            ground_truth_files = sorted(ground_truth_files)

            number_of_images = len(ground_truth_files)

        else:
            hdr_files = [image + '.hdr' for image in self.load_specific_images]
            float_files = [image + '.float' for image in self.load_specific_images]
            ground_truth_files = [image + '.npz' for image in self.load_specific_images]
            number_of_images = len(self.load_specific_images)

        for i in range(number_of_images):

            img = envi.open(f'{self.hyperspectral_data_path}/{hdr_files[i]}', f'{self.hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)
            img /= self.pixel_max_value

            if self.remove_noisy_bands:
                img = np.delete(img, NOISY_BANDS_INDICES, axis=2)

            gt = np.load(f'{self.ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt[gt > 7] = 0

            if img.shape[:2] == gt.shape:
                rows, cols, bands = img.shape
            else:
                error_message = f'Dimensions mismatch.\nHyperspectral image shape: {img.shape},\nGroud truth shape: {gt.shape}.'
                raise ValueError(error_message)
            
            gt = gt.flatten()
            img = img.reshape(rows*cols, bands)

            if self.remove_background:
                background_indices = np.where(gt == BACKGROUND_VALUE)[0]
                
                gt = np.delete(gt, background_indices)
                img = np.delete(img, background_indices, axis=0)

            if self.shuffle == True:
                combined = list(zip(gt, img))
                np.random.shuffle(combined)
                gt, img = zip(*combined)
                gt, img = np.array(gt), np.array(img)

            if self.stage == Stage.TRAIN:
                gt = gt[:int(0.75*len(gt))]
                img = img[:int(0.75*len(img))]  

            elif self.stage == Stage.VALIDATE:
                gt = gt[int(0.75*len(gt)):int(0.90*len(gt))]
                img = img[:int(0.75*len(img)):int(0.90*len(gt))]

            elif self.stage == Stage.TEST:
                gt = gt[int(0.9*len(gt))]
                img = img[:int(0.9*len(img))]

            for i in range(len(img)):
                pixel = torch.tensor(img[i])
                pixel = pixel.reshape(1, pixel.shape[0])
                label = torch.tensor(gt[i])

                if self.stage == Stage.TRAIN or self.stage == Stage.VALIDATE:
                    label = self.onehot_encoding(int(label.item()))
                    
                yield pixel, label

    def onehot_encoding(self, label: torch.TensorType) -> Sequence[int]:
        onehot_label = torch.zeros(self.classes)
        onehot_label[label] = 1.0
        return onehot_label