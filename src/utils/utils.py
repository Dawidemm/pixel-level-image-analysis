import os
import numpy as np
import torch
from sklearn.metrics import rand_score, adjusted_rand_score, completeness_score, homogeneity_score
import plotly.graph_objects as go
import lightning
from sklearn.datasets import make_blobs

from itertools import combinations
from typing import Union, Sequence, Tuple, Optional
from numpy.typing import ArrayLike

np.random.seed(10)


def train_test_split(
    hyperspectral_image: ArrayLike, 
    ground_truth_image: ArrayLike, 
    split=0.2
):
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

    if split <= 0:
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


def classes_filter(
        hyperspectral_vector: ArrayLike, 
        ground_truth_vector: ArrayLike, 
        classes_to_remove: list
):
    '''
    Filters a hyperspectral vector and ground truth vector based on list of class values to remove.
    
    Args:
        image (np.array): Hyperspectral vector.
        ground_truth (np.array): Ground truth vector.
        values_to_keep (list): List of values to remove.
    
    Returns:
        np.array: Filtered hyperspectral image vector.
        np.array: Filtered ground truth vector.
    '''

    indices_to_remove = np.where(np.isin(ground_truth_vector, classes_to_remove))[0]

    filtered_image = np.delete(hyperspectral_vector, indices_to_remove, axis=0)
    filtered_gt = np.delete(ground_truth_vector, indices_to_remove, axis=0)
    
    return filtered_image, filtered_gt

    
class ThresholdFinder:
    def __init__(
            self,
            dataloader,
            rbm,
            encoder=None
    ):
        self.dataloader = dataloader
        self.rbm = rbm
        self.encoder = encoder

    def find_threshold(self, thresholds: Union[Sequence, ArrayLike]):

        self.best_threshold = None
        self.adjusted_rand_score = float('-inf')

        for threshold in thresholds:

            unique_labels = set()
            labels = []
            y_true = []

            for X, y in self.dataloader:

                if self.encoder is None:
                    rbm_input = X.detach().numpy()[0]
                else:
                    encoder_output, _ = self.encoder.forward(X, epoch=1)
                    rbm_input = encoder_output.detach().numpy()

                label = self.rbm.binarized_rbm_output(rbm_input, threshold)

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

            adj_rand_score_value = adjusted_rand_score(y_true, mapped_labels)

            if adj_rand_score_value > self.adjusted_rand_score:
                self.best_threshold = threshold
                self.adjusted_rand_score = adj_rand_score_value
                self.rand_score = rand_score(y_true, mapped_labels)
                self.homogenity = homogeneity_score(y_true, mapped_labels)
                self.completeness = completeness_score(y_true, mapped_labels)
                self.mapped_labels = mapped_labels

        return self.best_threshold, self.adjusted_rand_score, self.rand_score, self.homogenity, self.completeness, self.mapped_labels

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
    
def spectral_angle(vector_a: ArrayLike, vector_b: ArrayLike):
    dot_product = np.dot(np.squeeze(vector_a), np.squeeze(vector_b))
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    cos_theta = dot_product/(norm_a * norm_b)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle

def euklidean_distance(vector_a: ArrayLike, vector_b: ArrayLike):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    return np.linalg.norm(vector_a - vector_b)
    
def spectral_angle_distance_matrix(
        objects: ArrayLike, 
        rbm_labels: Optional[ArrayLike]=None
):
    n = len(objects)
    distance_matrix = np.zeros((n, n))

    if rbm_labels != None:
        for i, j in combinations(range(n), 2):
            if np.array_equal(rbm_labels[i], rbm_labels[j]):
                distance = 0
            else:
                distance = euklidean_distance(objects[i], objects[j])

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    else:
        for i, j in combinations(range(n), 2):
            distance = euklidean_distance(objects[i], objects[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
    
class LossLoggerCallback(lightning.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.validation_losses = []

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        train_loss = trainer.logged_metrics['train_loss']
        self.train_losses.append(train_loss)

        val_loss = trainer.logged_metrics['val_loss']
        self.validation_losses.append(val_loss)
    
def plot_loss(
        train_loss_values: Sequence[float],
        validation_loss_values: Sequence[float],
        plot_title: str,
        save: bool=True,
        format: str='pdf',
        experiment_number: Union[int, None]=None
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(train_loss_values))), y=train_loss_values, mode='lines', name='Train Loss'))

    if len(train_loss_values) != len(validation_loss_values):
        fig.add_trace(
            go.Scatter(
                x=[validation_loss_values[idx][0] for idx in range(len(validation_loss_values))],
                y=[validation_loss_values[idx][1] for idx in range(len(validation_loss_values))], 
                mode='lines', 
                name='Val Loss'
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(validation_loss_values))), 
                y=validation_loss_values, 
                mode='lines', 
                name='Val Loss'
            )
        )

    fig.update_layout(
        title=plot_title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    if save:
        if experiment_number != None:
            experiment_path = f'./experiments/exp_{experiment_number}/'
            os.makedirs(experiment_path, exist_ok=True)
            plot_loss_path = os.path.join(experiment_path, f'{plot_title.lower()}_learning.{format}')
        else:
            plot_loss_path = f'{plot_title.lower()}_learning.{format}'

        fig.write_image(
            plot_loss_path,
            width=800,
            height=600,
            scale=1
        )

class SyntheticDataGenerator():
    def __init__(
            self,
            n_pixels: int,
            n_features: int,
            n_classes: int,
            image_width: int,
            image_height: int,
    ):
        '''
        Generator for synthetic image data.

        Attributes:
        - n_pixels (int): Number of pixels.
        - n_features (int): Number of features.
        - n_classes (int): Number of classes.
        - image_width (int): Width of the image.
        - image_height (int): Height of the image.

        Methods:
        - generate_synthetic_data(quantize=False) -> Tuple[ArrayLike, ArrayLike]:
            Generates synthetic image data and corresponding labels.
        - should_quantize(x: ArrayLike) -> ArrayLike:
            Quantizes the input data by changing all values to -1 or 1.
        '''

        self.n_pixels = n_pixels
        self.n_features = n_features
        self.n_classes = n_classes
        self.image_width = image_width
        self.image_height = image_height

    def generate_synthetic_data(
            self,
            quantize: bool=False
    ) -> Tuple[ArrayLike, ArrayLike]:
        
        '''
        Generates synthetic image data.

        Parameters:
        - quantize (bool): Whether to quantize the data. Default is False.

        Returns:
        Tuple[ArrayLike, ArrayLike]: A tuple containing the synthetic image and corresponding labels.
        - synthetic_image (ArrayLike): The generated synthetic data reshaped into (features, image_width, image_height).
        - synthetic_labels (ArrayLike): The labels for the synthetic data reshaped into (1, image_width, image_height).
        '''
            
        blobs = make_blobs(
            n_samples=self.n_pixels,
            n_features=self.n_features,
            centers=self.n_classes,
            random_state=100
        )

        synthetic_image = blobs[0].reshape((self.n_features, self.image_width, self.image_height))
        synthetic_labels = blobs[1].reshape((1, self.image_width, self.image_height))

        if quantize == True:
            for band in range(len(synthetic_image[0])):
                synthetic_image[band] = synthetic_image[band]/synthetic_image[band].max()

            synthetic_image = self.should_quantize(synthetic_image)

        return synthetic_image, synthetic_labels
    
    @staticmethod
    def should_quantize(x: ArrayLike) -> ArrayLike:
        x = np.sign(x)
        x[x == 0] = 1
        return x
    
def gini_index(x: ArrayLike) -> float:

    x = x.flatten()

    if np.amin(x) < 0:
        x -= np.amin(x)

    x = x + 0.0000001
    x = np.sort(x)
    index = np.arange(1, x.shape[0]+1)
    num_elements = len(x)

    return round(((np.sum((2 * index - num_elements - 1) * x)) / (num_elements * np.sum(x))), 3)