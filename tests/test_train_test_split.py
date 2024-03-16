from src.utils import utils
import numpy as np
import pytest

HYPERSPECTRAL_IMAGE = np.random.rand(220, 145, 145)
GROUND_TRUTH_IMAGE = np.random.randint(0, 16, size=(1, 145, 145))

def test_train_test_split_output():
    dataset = utils.train_test_split(HYPERSPECTRAL_IMAGE, GROUND_TRUTH_IMAGE)

    assert isinstance(dataset, tuple)
    assert len(dataset) == 4

    for data in dataset:
        assert isinstance(data, np.ndarray)

def test_dimension_mismatch_error():
    with pytest.raises(ValueError):
        utils.train_test_split(HYPERSPECTRAL_IMAGE, np.random.rand(1, 50, 50))

def test_multiple_splits():
    splits_to_test = [0.1, 0.15, 0.2, 0.25, 0.3]
    for split in splits_to_test:
        _, _, ground_truth_train, ground_truth_test = utils.train_test_split(HYPERSPECTRAL_IMAGE, GROUND_TRUTH_IMAGE, split=split)
        expected_split = len(ground_truth_test)/(len(ground_truth_train)+len(ground_truth_test))

        assert np.isclose(split, expected_split, atol=0.01)

def test_class_balance():
    split = 0.35
    dataset = utils.train_test_split(HYPERSPECTRAL_IMAGE, GROUND_TRUTH_IMAGE, split=split)
    _, _, _, removed_ground_truth = dataset

    classes_in_ground_truth_image = np.unique(GROUND_TRUTH_IMAGE)
    total_samples_per_class = {class_pixel_value: len(np.argwhere(GROUND_TRUTH_IMAGE== class_pixel_value)) for class_pixel_value in classes_in_ground_truth_image}

    for class_pixel_value in classes_in_ground_truth_image:
        total_samples = total_samples_per_class[class_pixel_value]
        expected_removed_samples = int(total_samples * split)
        removed_samples_for_class = removed_ground_truth[removed_ground_truth == class_pixel_value]

        assert removed_samples_for_class.size == expected_removed_samples