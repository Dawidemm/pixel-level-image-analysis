import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import IterableDataset
from src.utils import utils
from typing import Sequence, Union, Tuple, List
from numpy.typing import ArrayLike
from enum import Enum


NOISY_BANDS_INDICES = np.array([0, 1, 2, 3, 4, 5, 48, 49, 50, 121, 122, 123, 124, 125, 126, 127])
BACKGROUND_VALUE = 0

class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    SEG = 'segmentation'


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
            shuffle: bool = True,
            random_seed: int = 42,
            partition: Union[Tuple[float, float], None] = None
    ):  
        '''
        PyTorch IterableDataset for loading and processing hyperspectral images and their corresponding ground truth data.
        The dataset provides an iterable over pixel-label pairs, supporting different processing stages (TRAIN, VALIDATION, TEST),
        with options for background removal, noisy band filtering, and dataset shuffling.

        Parameters:
        - hyperspectral_data_path (str): Path to the folder containing hyperspectral image files (.hdr and .float format).
        - ground_truth_data_path (str): Path to the folder containing ground truth data files (.npz format).
        - load_specific_images (Union[List[str], None], optional): List of specific images to load (without extension).
        If None, all available images will be loaded. Default is None.
        - remove_noisy_bands (bool, optional): Whether to remove noisy spectral bands from the hyperspectral data. Default is True.
        - remove_background (bool, optional): Whether to exclude background pixels (value 0 in ground truth) from the dataset. Default is False.
        - stage (Stage, optional): The processing stage (TRAIN, VAL, or TEST), determining how the data is split.
        - shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.
        - random_seed (int, optional): Seed for random shuffling. Default is 42.

        Attributes:
        - pixel_max_value (float): Maximum pixel value from the hyperspectral data, used for normalization.
        - classes (int): The number of unique classes in the ground truth data, used for one-hot encoding.

        Methods:
        - __iter__(): Provides an iterable that yields pairs of normalized pixel data (torch.Tensor) and ground truth labels.
        Labels are one-hot encoded during training and validation, and background pixels can be optionally removed.
        The dataset is shuffled if specified.
        - onehot_encoding(label: torch.TensorType) -> Sequence[int]: Converts a scalar label into a one-hot encoded vector.
        - shuffle_data(gt: ArrayLike, img: ArrayLike) -> Tuple[ArrayLike, ArrayLike]: Shuffles the ground truth and image data.
        - train_val_test_split(gt: ArrayLike, img: ArrayLike, stage: Stage) -> Tuple[ArrayLike, ArrayLike]: Splits the data 
        into training, validation, or test sets based on the specified stage.
        '''

        self.hyperspectral_data_path = hyperspectral_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.load_specific_images = load_specific_images
        self.remove_noisy_bands = remove_noisy_bands
        self.remove_background = remove_background
        self.stage = stage
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.partition = partition

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

        ground_truth_pixels = np.array([])
        hyperspectral_pixels = []

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

            ground_truth_pixels = np.append(ground_truth_pixels, gt)
            hyperspectral_pixels.append(img)

        hyperspectral_pixels = np.vstack(hyperspectral_pixels)

        if self.shuffle == True:
            ground_truth_pixels, hyperspectral_pixels = self.shuffle_data(
                gt=ground_truth_pixels,
                img=hyperspectral_pixels,
            )
        
        if self.partition != None:
            ground_truth_pixels = ground_truth_pixels[self.partition[0]:self.partition[1]]
            hyperspectral_pixels = hyperspectral_pixels[self.partition[0]:self.partition[1]]
        
        ground_truth_pixels, hyperspectral_pixels = self.train_val_test_split(
            gt=ground_truth_pixels,
            img=hyperspectral_pixels,
            stage=self.stage
        )

        for i in range(len(hyperspectral_pixels)):
            pixel = torch.tensor(hyperspectral_pixels[i])
            pixel = pixel.reshape(1, pixel.shape[0])
            label = torch.tensor(ground_truth_pixels[i])

            if self.stage == Stage.TRAIN or self.stage == Stage.VAL:
                label = self.onehot_encoding(int(label.item()))
                    
            yield pixel, label

    def onehot_encoding(
            self, 
            label: torch.TensorType
        ) -> Sequence[int]:

        onehot_label = torch.zeros(self.classes)
        onehot_label[label] = 1.0

        return onehot_label
    
    def shuffle_data(
            self,
            gt: ArrayLike,
            img: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        
        random_state = np.random.RandomState(seed=self.random_seed)

        combined = list(zip(gt, img))
        random_state.shuffle(combined)
        gt, img = zip(*combined)
        gt, img = np.array(gt), np.array(img)

        return gt, img
    
    def train_val_test_split(
            self,
            gt: ArrayLike,
            img: ArrayLike,
            stage: Stage
    ) -> Tuple[ArrayLike, ArrayLike]:
        
        if stage == Stage.TRAIN:
            gt = gt[:int(0.8*len(gt))]
            img = img[:int(0.8*len(img))]

            gt = gt[:int(0.8*len(gt))]
            img = img[:int(0.8*len(img))]
            # print(f'train dataset gini: {utils.gini_index(gt)}')
        elif stage == Stage.VAL:
            gt = gt[:int(0.8*len(gt))]
            img = img[:int(0.8*len(img))]

            gt = gt[int(0.8*len(gt)):]
            img = img[int(0.8*len(img)):]
            # print(f'val dataset gini: {utils.gini_index(gt)}')
        elif self.stage == Stage.TEST:
            gt = gt[int(0.8*len(gt)):]
            img = img[int(0.8*len(img)):]
            # print(f'test dataset gini: {utils.gini_index(gt)}')

        elif self.stage == Stage.SEG:
            pass

        return gt, img