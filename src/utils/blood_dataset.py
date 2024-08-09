import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import IterableDataset

from typing import Sequence, Union
from enum import Enum


NOISY_BANDS_INDICES = np.array([0, 1, 2, 3, 4, 5, 48, 49, 50, 121, 122, 123, 124, 125, 126, 127])

class Stage(Enum):
    TRAIN = 'train'
    IMG_SEG = 'image_segmentation'

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
            num_images_to_load: Union[int, None]=None,
            load_specific_image: Union[str, None]=None,
            remove_noisy_bands: bool=True,
            stage = Stage
    ):  
        '''
        A dataset class for loading hyperspectral images and ground truth data in an iterable format.

        Parameters:
        - hyperspectral_data_path (str): Path to the directory containing hyperspectral data. Should include .hdr and .float files.
        - ground_truth_data_path (str): Path to the directory containing ground truth data. Should include .npz files with ground truth data.
        - num_images_to_load (Union[int, None], optional): Number of images to load. If None, all images in the directory will be loaded.
        - load_specific_image (Union[str, None], optional): Specific image to load (without extension). If None, images will be loaded based on `num_images_to_load`.
        - remove_noisy_bands (bool, optional): Flag indicating whether to remove noisy bands from hyperspectral data. Default is True.
        - stage (Stage, optional): Processing stage (e.g., training). Default is Stage.

        Attributes:
        - pixel_max_value (float): Maximum pixel value in the hyperspectral data used for normalization.
        - classes (int): Number of classes in the ground truth data, used for one-hot encoding labels.

        Methods:
        - __iter__(): Generator that iterates over hyperspectral images and ground truth data. Normalizes hyperspectral data and converts ground truth to one-hot encoding if required.
        - onehot_encoding(label: torch.TensorType) -> Sequence[int]: Converts a label to a one-hot encoded format.
        '''

        self.hyperspectral_data_path = hyperspectral_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.num_images_to_load = num_images_to_load
        self.load_specific_image = load_specific_image
        self.remove_noisy_bands = remove_noisy_bands
        self.stage = stage

        self.pixel_max_value, self.classes = blood_dataset_params(
            hyperspectral_data_path=hyperspectral_data_path,
            ground_truth_data_path=ground_truth_data_path,
            remove_noisy_bands=self.remove_noisy_bands
        )

    def __iter__(self):

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

        if self.num_images_to_load == None:
            number_of_images = len(float_files)
        else:
            number_of_images = self.num_images_to_load

        if self.load_specific_image != None:
            hdr_files = [self.load_specific_image + '.hdr']
            float_files = [self.load_specific_image + '.float']
            ground_truth_files = [self.load_specific_image + '.npz']
            number_of_images = 1

        for i in range(number_of_images):

            img = envi.open(f'{self.hyperspectral_data_path}/{hdr_files[i]}', f'{self.hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)

            if self.remove_noisy_bands:
                img = np.delete(img, NOISY_BANDS_INDICES, axis=2)

            gt = np.load(f'{self.ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt[gt > 7] = 0

            if img.shape[:2] == gt.shape:
                rows, cols, bands = img.shape

            for row in range(rows):
                for col in range(cols):
                    pixel = torch.tensor(img[row, col]) / self.pixel_max_value
                    pixel = pixel.reshape(1, pixel.shape[0])
                    label = torch.tensor(gt[row, col])

                    if self.stage == Stage.TRAIN:
                        label = self.onehot_encoding(int(label.item()))

                    yield pixel, label

    def onehot_encoding(self, label: torch.TensorType) -> Sequence[int]:
        onehot_label = torch.zeros(self.classes)
        onehot_label[label] = 1.0
        return onehot_label