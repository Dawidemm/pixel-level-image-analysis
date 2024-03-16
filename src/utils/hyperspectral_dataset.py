import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
from enum import IntEnum, Enum
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
            hyperspectral_image_path: str, 
            ground_truth_image_path: str,
            stage: Stage
    ):
        hyperspectral_image = tifffile.imread(hyperspectral_image_path)
        ground_truth_image = tifffile.imread(ground_truth_image_path)

        hyperspectral_image = hyperspectral_image.astype(np.float32)
        ground_truth_image = ground_truth_image.astype(np.float32)
        
        hyperspectral_image /= hyperspectral_image.max()
        ground_truth_image = ground_truth_image.reshape(1, ground_truth_image.shape[0], ground_truth_image.shape[1])

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
        return len(self.hyperspectral_image[0])
    
    def __getitem__(self, index):

        pixel_values = self.hyperspectral_image[index]
        label = self.ground_truth_image.clone().detach()[index]
        
        return pixel_values, label