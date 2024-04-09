import numpy as np
import torch
from sklearn.metrics import rand_score
import matplotlib.pyplot as plt
import lightning
from typing import List

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

    Raises:
    - ValueError: If the dimensions of the ground truth image do not match the dimensions of the hyperspectral image.
    '''

    if split == 0:
        hyperspectral_image = hyperspectral_image.reshape(
            hyperspectral_image.shape[1] * hyperspectral_image.shape[2],
            hyperspectral_image.shape[0]
        )
        
        ground_truth_image = ground_truth_image.reshape(
            ground_truth_image.shape[1] * ground_truth_image.shape[2])

        return hyperspectral_image, ground_truth_image

    if ground_truth_image.shape[1] != hyperspectral_image.shape[1] or ground_truth_image.shape[2] != hyperspectral_image.shape[2]:
        raise ValueError('Dimension mismatch between ground truth image and hyperspectral image.')  

    ground_truth_remaining_samples = []
    ground_truth_removed_samples = []
    hyperspectral_remaining_samples = []
    hyperspectral_removed_samples = []

    ground_truth_image = ground_truth_image[0]
    ground_truth_image = ground_truth_image.flatten()

    classes_in_ground_truth_image = np.unique(ground_truth_image)
 
    samples_to_remove_per_class = {class_pixel_value: int(len(np.argwhere(ground_truth_image == class_pixel_value)) * split) for class_pixel_value in classes_in_ground_truth_image}

    indices_to_remove = []

    for class_pixel_value in classes_in_ground_truth_image:

        class_indices = np.argwhere(ground_truth_image == class_pixel_value)
        count_of_indices_to_remove = samples_to_remove_per_class[class_pixel_value]
        indices_to_remove_per_class = np.random.choice(
            class_indices.flatten(),
            size=count_of_indices_to_remove,
            replace=False
        )
        
        indices_to_remove.append(indices_to_remove_per_class)

    indices_to_remove = np.concatenate(indices_to_remove)
    remaining_indices = np.setdiff1d(np.arange(ground_truth_image.size), indices_to_remove)

    ground_truth_remaining_samples = ground_truth_image[remaining_indices].T
    ground_truth_removed_samples = ground_truth_image[indices_to_remove].T

    bands = hyperspectral_image.shape[0]

    for band in range(bands):

        hyperspectral_data = hyperspectral_image[band]
        hyperspectral_data = hyperspectral_data.flatten()

        remaining_indices = np.setdiff1d(np.arange(hyperspectral_data.size), indices_to_remove)

        hyperspectral_remaining_data = hyperspectral_data[remaining_indices]
        hyperspectral_removed_data = hyperspectral_data[indices_to_remove]

        hyperspectral_remaining_samples.append(hyperspectral_remaining_data)
        hyperspectral_removed_samples.append(hyperspectral_removed_data)

    hyperspectral_remaining_samples = np.array(hyperspectral_remaining_samples).T
    hyperspectral_removed_samples = np.array(hyperspectral_removed_samples).T

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
    
class LossLoggerCallback(lightning.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        loss = trainer.logged_metrics['loss']
        self.losses.append(loss)
    
def plot_loss(
        epochs: int, 
        loss_values: List[float], 
        plot_title: str, 
        save: bool=True
):

    plt.figure(figsize=(6, 5))
    plt.plot(list(range(epochs)), loss_values)
    plt.gcf().set_facecolor('none')
    plt.gca().set_facecolor('none')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(plot_title)
    plt.grid()

    if save:
        plt.savefig(f'{plot_title.lower()}_learning.pdf', dpi=300)

    plt.show()