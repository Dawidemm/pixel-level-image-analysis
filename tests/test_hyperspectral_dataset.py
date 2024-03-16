import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

@pytest.fixture
def hyperspectral_dataset_train():
    hyperspectral_data = np.random.rand(220, 145, 145).astype(np.float32)
    ground_truth_data = np.random.rand(1, 145, 145).astype(np.float32)
    return HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TRAIN)

@pytest.fixture
def hyperspectral_dataset_test():
    hyperspectral_data = np.random.rand(220, 145, 145).astype(np.float32)
    ground_truth_data = np.random.rand(1, 145, 145).astype(np.float32)
    return HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TEST)

def test_hyperspectral_dataset_shape(hyperspectral_dataset_train, hyperspectral_dataset_test):
    assert hyperspectral_dataset_train.hyperspectral_image.shape == torch.Size([21025, 220])
    assert hyperspectral_dataset_train.ground_truth_image.shape == torch.Size([21025])
    assert hyperspectral_dataset_test.hyperspectral_image.shape == torch.Size([0, 220])
    assert hyperspectral_dataset_test.ground_truth_image.shape == torch.Size([0])

# def test_hyperspectral_dataset_length(hyperspectral_dataset_train, hyperspectral_dataset_test):
#     assert len(hyperspectral_dataset_train) == 220
#     assert len(hyperspectral_dataset_test) == 0

def test_hyperspectral_dataset_getitem(hyperspectral_dataset_train, hyperspectral_dataset_test):
    sample_index = 0
    pixel_values_train, label_train = hyperspectral_dataset_train[sample_index]
    pixel_values_test, label_test = hyperspectral_dataset_test[sample_index]

    assert isinstance(pixel_values_train, torch.Tensor)
    assert isinstance(label_train, torch.Tensor)
    assert pixel_values_train.shape == torch.Size([220])
    assert label_train.shape == torch.Size([0])

    assert isinstance(pixel_values_test, torch.Tensor)
    assert isinstance(label_test, torch.Tensor)
    assert pixel_values_test.shape == torch.Size([220])
    assert label_test.shape == torch.Size([0])

# def test_hyperspectral_dataset_dataloader(hyperspectral_data, ground_truth_data):

#     dataset = HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TRAIN)
#     batch_size = 4
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for batch in dataloader:
#         assert len(batch) == 2
#         assert batch[0].shape[0] == batch_size
#         assert batch[1].shape[0] == batch_size
#         assert batch[0].shape[1:] == torch.Size([20, 30])
#         assert batch[1].shape[1:] == torch.Size([1, 20, 30])
