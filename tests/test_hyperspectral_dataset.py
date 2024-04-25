import pytest
import numpy as np
import torch
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

@pytest.fixture
def hyperspectral_dataset_train():
    hyperspectral_data = np.random.rand(220, 145, 145)
    ground_truth_data = np.random.randint(0, 16, size=(1, 145, 145))
    return HyperspectralDataset(hyperspectral_data, ground_truth_data, Stage.TRAIN)

def test_hyperspectral_dataset_getitem_type(hyperspectral_dataset_train):
    sample_index = 0
    pixel_values_train, label_train = hyperspectral_dataset_train[sample_index]

    assert isinstance(pixel_values_train, torch.Tensor)
    assert isinstance(label_train, torch.Tensor)

def test_hyperspectral_dataset_getitem_shape(hyperspectral_dataset_train):
    sample_index = 0
    pixel_values, label = hyperspectral_dataset_train[sample_index]

    assert pixel_values.shape == torch.Size([1, 220])
    assert label.shape == torch.Size([])