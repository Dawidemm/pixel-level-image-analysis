import pytest
import numpy as np
import torch
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

@pytest.fixture
def hyperspectral_dataset_train():
    hyperspectral_data = np.random.rand(220, 145, 145)
    ground_truth_data = np.random.randint(0, 16, size=(1, 145, 145))
    return HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TRAIN)

@pytest.fixture
def hyperspectral_dataset_test():
    hyperspectral_data = np.random.rand(220, 145, 145)
    ground_truth_data = np.random.randint(0, 16, size=(1, 145, 145))
    return HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TEST)

def test_hyperspectral_dataset_getitem(hyperspectral_dataset_train, hyperspectral_dataset_test):
    sample_index = 0
    pixel_values_train, label_train = hyperspectral_dataset_train[sample_index]
    pixel_values_test, label_test = hyperspectral_dataset_test[sample_index]

    assert isinstance(pixel_values_train, torch.Tensor)
    assert isinstance(label_train, torch.Tensor)
    assert pixel_values_train.shape == torch.Size([220])
    assert label_train.shape == torch.Size([])

    assert isinstance(pixel_values_test, torch.Tensor)
    assert isinstance(label_test, torch.Tensor)
    assert pixel_values_test.shape == torch.Size([220])
    assert label_test.shape == torch.Size([])