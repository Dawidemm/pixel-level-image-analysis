import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from src.utils.utils import train_test_split


class HyperspectralDataset(Dataset):
    def __init__(
            self, 
            hyperspectral_image_path: str, 
            ground_truth_image_path: str,
            stage: str
    ):
        hyperspectral_image = tifffile.imread(hyperspectral_image_path)
        ground_truth_image = tifffile.imread(ground_truth_image_path)

        hyperspectral_image = hyperspectral_image.astype(np.float32)
        ground_truth_image = ground_truth_image.astype(np.float32)
        
        hyperspectral_image /= hyperspectral_image.max()
        ground_truth_image = ground_truth_image.reshape(1, ground_truth_image.shape[0], ground_truth_image.shape[1])

        output = train_test_split(hyperspectral_image, ground_truth_image, split=0.2)

        if stage == 'train':

            hyperspectral_image = output[0]
            ground_truth_image = output[2]

        elif stage == 'test':

            hyperspectral_image = output[1]
            ground_truth_image= output[3]

        else:
            raise ValueError(f'Stage should be set as one from ["train", "test"] values.')
        
        self.hyperspectral_image = torch.tensor(hyperspectral_image)
        self.ground_truth_image = torch.tensor(ground_truth_image)

        self.image_shape = self.hyperspectral_image.shape
        self.bands = self.image_shape[0]
        self.width = self.image_shape[1]
        self.height = self.image_shape[2]

    def  __len__(self):
        return self.width * self.height
    
    def __getitem__(self, index):

        x = index // self.width
        y = index % self.height

        pixel_values = self.hyperspectral_image[:, x, y]
        padding = torch.zeros(36)
        combined_pixel_values = torch.cat((pixel_values, padding), dim=0)
        combined_pixel_values = combined_pixel_values.reshape(1, 16, 16)

        label = self.ground_truth_image.clone().detach()
        label = int(label[:, x, y].item())
        
        return combined_pixel_values, label