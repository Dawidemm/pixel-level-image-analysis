import numpy as np
import torch
from sklearn.metrics import rand_score

np.random.seed(10)


def train_test_split(hyperspectral_image: np.array, ground_truth_image: np.array, split=0.2):
    '''
    Splits the hyperspectral and ground truth images into training and testing datasets.

    Args:
    - hyperspectral_image (np.array): An array representing the hyperspectral image.
    - ground_truth_image (np.array): An array representing the ground truth image.
    - split (float): The fraction of samples to be reserved for testing. Defaults to 0.2.

    Returns:
    Tuple: A tuple containing four elements:
    - hyperspectral_remaining_samples (np.array): Array of remaining hyperspectral samples for training.
    - hyperspectral_removed_samples (np.array): Array of removed hyperspectral samples for testing.
    - ground_truth_remaining_samples (np.array): Array of remaining ground truth samples for training.
    - ground_truth_removed_samples (np.array): Array of removed ground truth samples for testing.
    '''

    if ground_truth_image.shape[1] != hyperspectral_image.shape[1] or ground_truth_image.shape[2] != hyperspectral_image.shape[2]:
        raise ValueError('Dimension mismatch between ground truth image and hyperspectral image.')  

    total_samples = ground_truth_image.shape[1] * ground_truth_image.shape[2]
    samples_to_remove = int(split * total_samples)

    ground_truth_remaining_samples = []
    ground_truth_removed_samples = []

    ground_truth_image = ground_truth_image[0]
    ground_truth_image = ground_truth_image.flatten()
    indices_to_remove = np.random.choice(ground_truth_image.size, size=samples_to_remove, replace=False)
    remaining_indices = np.setdiff1d(np.arange(ground_truth_image.size), indices_to_remove)

    ground_truth_remaining_data = ground_truth_image[remaining_indices]
    ground_truth_removed_data = ground_truth_image[indices_to_remove]

    ground_truth_remaining_data = np.pad(ground_truth_remaining_data,
                            (0, int(np.ceil(np.sqrt(len(ground_truth_remaining_data)))**2 - len(ground_truth_remaining_data))),
                            mode='constant', 
                            constant_values=0)
    
    ground_truth_removed_data = np.pad(ground_truth_removed_data,
                          (0, int(np.ceil(np.sqrt(len(ground_truth_removed_data)))**2 - len(ground_truth_removed_data))),
                          mode='constant',
                          constant_values=0)
    
    ground_truth_remaining_data_shape = int(np.sqrt(len(ground_truth_remaining_data)))
    ground_truth_remaining_data = ground_truth_remaining_data.reshape(ground_truth_remaining_data_shape, ground_truth_remaining_data_shape)

    ground_truth_removed_data_shape = int(np.sqrt(len(ground_truth_removed_data)))
    ground_truth_removed_data = ground_truth_removed_data.reshape(ground_truth_removed_data_shape, ground_truth_removed_data_shape)

    ground_truth_remaining_samples.append(ground_truth_remaining_data)
    ground_truth_removed_samples.append(ground_truth_removed_data)

    ground_truth_remaining_samples = np.array(ground_truth_remaining_samples)
    ground_truth_removed_samples = np.array(ground_truth_removed_samples)

    hyperspectral_remaining_samples = []
    hyperspectral_removed_samples = []

    bands = hyperspectral_image.shape[0]

    for band in range(bands):

        hyperspectral_data = hyperspectral_image[band]
        hyperspectral_data = hyperspectral_data.flatten()

        remaining_indices = np.setdiff1d(np.arange(hyperspectral_data.size), indices_to_remove)

        hyperspectral_remaining_data = hyperspectral_data[remaining_indices]
        hyperspectral_removed_data = hyperspectral_data[indices_to_remove]

        hyperspectral_remaining_data = np.pad(hyperspectral_remaining_data,
                                              (0, int(np.ceil(np.sqrt(len(hyperspectral_remaining_data)))**2 - len(hyperspectral_remaining_data))), 
                                              mode='constant',
                                              constant_values=0)
        
        hyperspectral_removed_data = np.pad(hyperspectral_removed_data,
                                            (0, int(np.ceil(np.sqrt(len(hyperspectral_removed_data)))**2 - len(hyperspectral_removed_data))),
                                            mode='constant',
                                            constant_values=0)
        
        hyperspectral_remaining_data_shape = int(np.sqrt(len(hyperspectral_remaining_data)))
        hyperspectral_remaining_data = hyperspectral_remaining_data.reshape(hyperspectral_remaining_data_shape, hyperspectral_remaining_data_shape)

        hyperspectral_removed_data_shape = int(np.sqrt(len(hyperspectral_removed_data)))
        hyperspectral_removed_data = hyperspectral_removed_data.reshape(hyperspectral_removed_data_shape, hyperspectral_removed_data_shape)

        hyperspectral_remaining_samples.append(hyperspectral_remaining_data)
        hyperspectral_removed_samples.append(hyperspectral_removed_data)

    hyperspectral_remaining_samples = np.array(hyperspectral_remaining_samples)
    hyperspectral_removed_samples = np.array(hyperspectral_removed_samples)

    dataset = (hyperspectral_remaining_samples, 
              hyperspectral_removed_samples, 
              ground_truth_remaining_samples, 
              ground_truth_removed_samples)

    return dataset

class ThresholdFinder:
    def __init__(self, test_dataloader, encoder, rbm):
        self.test_dataloader = test_dataloader
        self.encoder = encoder
        self.rbm = rbm

    def find_threshold(self, thresholds):
        '''
        Finds the best threshold value for binarizing RBM output based on Rand Score.

        Args:
        - thresholds (list): A list of threshold values to be evaluated.

        Returns:
        Tuple (best_threshold, best_rand_score): A tuple containing the best threshold
        value and its corresponding Rand Score achieved.
        '''

        self.best_threshold = None
        self.best_rand_score = float('-inf')

        for threshold in thresholds:

            unique_labels = set()
            labels = []
            y_true = []

            for _, (X, y) in enumerate(self.test_dataloader):

                encoder_output, _ = self.encoder.forward(X)

                rbm_input = encoder_output.detach().numpy()

                label = self.rbm.binarize_rbm_output(rbm_input, threshold)

                unique_label = tuple(map(tuple, label))
                unique_labels.add(unique_label)

                label = tuple(map(tuple, label))
                labels.append(label)

                y_true.append(y)
            
            y_true = torch.cat(y_true, dim=0)
            y_true = np.array(y_true)

            unique_labels = list(unique_labels)
            mapped_labels = self.map_to_indices(labels, unique_labels)
            mapped_labels = np.array(mapped_labels)

            rand_score_value = rand_score(y_true, mapped_labels)

            if rand_score_value > self.best_rand_score:
                self.best_threshold = threshold
                self.best_rand_score = rand_score_value

        return self.best_threshold, self.best_rand_score

    @staticmethod
    def map_to_indices(values_to_map: list, target_list: list):
        '''
        Maps a list of values to their corresponding indices in another list.

        Args:
        - values_to_map (list): A list of values to be mapped to indices.
        - target_list (list): The list containing the elements to be mapped to.

        Returns:
        List of indices: A list containing the indices of the values in the provided list.
        '''

        indices = [target_list.index(value) for value in values_to_map]

        return indices