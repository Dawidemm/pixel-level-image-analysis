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
        img = np.delete(img, NOISY_BANDS_INDICES, axis=2)

        if img.max() >= pixel_max_value:
            pixel_max_value = img.max()

        gt = np.load(f'{ground_truth_data_path}/{ground_truth_files[i]}')
        gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
        gt = np.delete(gt, NOISY_BANDS_INDICES, axis=2)
        gt[gt > 7] = 0

        if len(np.unique(gt)) >= classes:
            classes = len(np.unique(gt))

    return pixel_max_value, classes    


class BloodIterableDataset(IterableDataset):
    def __init__(
            self,
            hyperspectral_data_path: str,
            ground_truth_data_path: str,
            num_images_to_load: Union[int, None],
            stage = Stage
    ):
        self.hyperspectral_data_path = hyperspectral_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.num_images_to_load = num_images_to_load
        self.stage = stage

        self.pixel_max_value, self.classes = blood_dataset_params(
            hyperspectral_data_path=hyperspectral_data_path,
            ground_truth_data_path=ground_truth_data_path
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

        for i in range(number_of_images):

            img = envi.open(f'{self.hyperspectral_data_path}/{hdr_files[i]}', f'{self.hyperspectral_data_path}/{float_files[i]}')
            img = np.asarray(img[:,:,:], dtype=np.float32)
            img = np.delete(img, NOISY_BANDS_INDICES, axis=2)

            gt = np.load(f'{self.ground_truth_data_path}/{ground_truth_files[i]}')
            gt = np.asarray(gt['gt'][:,:], dtype=np.float32)
            gt = np.delete(gt, NOISY_BANDS_INDICES, axis=2)
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