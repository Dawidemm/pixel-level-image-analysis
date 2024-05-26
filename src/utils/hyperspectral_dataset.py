import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from enum import IntEnum, Enum
from typing import Union, Tuple, Sequence
from numpy.typing import ArrayLike
from src.utils import utils

class ImagePartitions(IntEnum):
    TRAIN_IMAGE = 0
    TEST_IMAGE = 1
    TRAIN_LABEL = 2
    TEST_LABEL = 3
    SEG_IMG = 0
    SEG_LABEL = 1

class Stage(Enum):
    TRAIN = 'train'
    TEST = 'test'
    IMG_SEG = 'image_segmentation'

class HyperspectralDataset(Dataset):
    def __init__(
            self, 
            hyperspectral_data: Union[str, ArrayLike], 
            ground_truth_data: Union[str, ArrayLike],
            stage: Stage,
            split: float=0.2,
            class_filter=False
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

        if stage == Stage.IMG_SEG:
            split = 0

        dataset = utils.train_test_split(hyperspectral_image, ground_truth_image, split=split)

        if stage == Stage.TRAIN:

            hyperspectral_image = dataset[ImagePartitions.TRAIN_IMAGE]
            ground_truth_image = dataset[ImagePartitions.TRAIN_LABEL]

        elif stage == Stage.TEST:

            hyperspectral_image = dataset[ImagePartitions.TEST_IMAGE]
            ground_truth_image= dataset[ImagePartitions.TEST_LABEL]

        elif stage == Stage.IMG_SEG:

            hyperspectral_image = dataset[ImagePartitions.SEG_IMG]
            ground_truth_image= dataset[ImagePartitions.SEG_LABEL]

        if class_filter:
            hyperspectral_image, ground_truth_image = utils.classes_filter(
                hyperspectral_vector=hyperspectral_image,
                ground_truth_vector=ground_truth_image,
                classes_to_remove=class_filter
            )
        
        self.hyperspectral_image = torch.tensor(hyperspectral_image)
        self.ground_truth_image = torch.tensor(ground_truth_image)

    def  __len__(self):
        return len(self.hyperspectral_image)
    
    def __getitem__(self, index: int) -> Tuple[Sequence, int]:

        pixel_values = self.hyperspectral_image[index]
        pixel_values = pixel_values.reshape(1, len(pixel_values))
        label = self.ground_truth_image.clone().detach()[index]
        label = self.onehot_encoding(int(label))
        
        return pixel_values, label
    
    def onehot_encoding(self, label: torch.TensorType) -> Sequence[int]:
        onehot_label = torch.zeros(len(torch.unique(self.ground_truth_image)))
        onehot_label[label] = 1.0
        return onehot_label