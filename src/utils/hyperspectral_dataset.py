import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from enum import IntEnum, Enum
from typing import Union
from src.utils.utils import train_test_split

class ImagePartitions(IntEnum):
    TRAIN_IMAGE = 0
    TEST_IMAGE = 1
    TRAIN_LABEL = 2
    TEST_LABEL = 3

class Stage(Enum):
    TRAIN = 'train'
    TEST = 'test'

class HyperspectralDataset(Dataset):
    def __init__(
            self, 
            hyperspectral_data: Union[str, np.array], 
            ground_truth_data: Union[str, np.array],
            stage: Stage
    ):
        if isinstance(hyperspectral_data, str):
            hyperspectral_image = tifffile.imread(hyperspectral_data)
        else:
            hyperspectral_image = hyperspectral_data

        if isinstance(ground_truth_data, str):
            ground_truth_image = tifffile.imread(ground_truth_data)
        else:
            ground_truth_image = ground_truth_data

        if len(ground_truth_image.shape) == 2:
            ground_truth_image = ground_truth_image.reshape(1, ground_truth_image.shape[0], ground_truth_image.shape[1])

        hyperspectral_image = hyperspectral_image.astype(np.float32)
        ground_truth_image = ground_truth_image.astype(np.float32)
        
        hyperspectral_image /= hyperspectral_image.max()

        dataset = train_test_split(hyperspectral_image, ground_truth_image, split=0.2)

        if stage == Stage.TRAIN:

            hyperspectral_image = dataset[ImagePartitions.TRAIN_IMAGE]
            ground_truth_image = dataset[ImagePartitions.TRAIN_LABEL]

        elif stage == Stage.TEST:

            hyperspectral_image = dataset[ImagePartitions.TEST_IMAGE]
            ground_truth_image= dataset[ImagePartitions.TEST_LABEL]

        else:
            raise ValueError(f'Stage should be set as one from ["train", "test"] values.')
        
        self.hyperspectral_image = torch.tensor(hyperspectral_image)
        self.ground_truth_image = torch.tensor(ground_truth_image)

    def  __len__(self):
        return len(self.hyperspectral_image)
    
    def __getitem__(self, index):

        pixel_values = self.hyperspectral_image[index]
        pixel_values = pixel_values.reshape(1, len(pixel_values))
        label = self.ground_truth_image.clone().detach()[index]
        
        return pixel_values, label