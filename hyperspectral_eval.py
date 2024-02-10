import torch
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from hyperspectral_dataset import HyperspectralDataset
from sklearn.metrics import rand_score

from qbm4eo.lbae import LBAE
from qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 16
NUM_HIDDEN = 17

THRESHOLDS = np.linspace(1/10, 1, 10)

def binarize_rbm_output(h_probs_given_v: np.array, threshold: float) -> np.array:

    h_probs_given_v[h_probs_given_v <= threshold] = 0
    h_probs_given_v[h_probs_given_v > threshold] = 1

    return h_probs_given_v

def map_to_indices(values, lst):
    indices = [lst.index(value) for value in values]
    return indices

def find_threshold(thresholds, test_dataloader, lbae, rbm):

    rand_scores = []
    mapped_labels_list = []

    for threshold in thresholds:

        unique_labels = set()
        labels = []

        y_true = []

        for batch, (X, y) in enumerate(test_dataloader):

            encoder, _ = lbae.encoder.forward(X)

            rbm_input = encoder.detach().numpy()

            probabilities = rbm.h_probs_given_v(rbm_input)
            label = binarize_rbm_output(probabilities, threshold)

            unique_label = tuple(map(tuple, label))
            unique_labels.add(unique_label)

            label = tuple(map(tuple, label))
            labels.append(label)

            y_true.append(y)
        
        y_true = torch.cat(y_true, dim=0)
        y_true = np.array(y_true)

        unique_labels = list(unique_labels)

        mapped_labels = map_to_indices(labels, unique_labels)

        mapped_labels = np.array(mapped_labels)
        mapped_labels_list.append(mapped_labels)

        rand_score_value = rand_score(y_true, mapped_labels)
        rand_scores.append(rand_score_value)

    rand_scores = np.array(rand_scores)
    rand_score_max_index = np.argmax(rand_scores)
    best_threshold = thresholds[rand_score_max_index]
    best_rand_score = np.max(rand_scores)

    return best_threshold, best_rand_score

def main():
    test_dataset = HyperspectralDataset(
        hyperspectral_image_path='dataset/hyperspectral_image.tif',
        ground_truth_image_path='dataset/ground_truth_image.tif',
        stage='test'
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lbae = LBAE.load_from_checkpoint(checkpoint_path='lightning_logs/version_0/checkpoints/epoch=99-step=211300.ckpt', 
                                    hparams_file='lightning_logs/version_0/hparams.yaml',
                                    map_location=torch.device('cpu'))

    lbae.eval()

    predictions = []
    X_true = []

    with torch.no_grad():
        for X, _ in test_dataloader:

            X_true.append(X)

            outputs = lbae(X)
            predictions.append(outputs)

    predictions= torch.cat(predictions, dim=0)
    predictions = predictions.reshape(predictions.shape[0],
                                    predictions.shape[1] * predictions.shape[2] * predictions.shape[3])

    X_true = torch.cat(X_true, dim=0)
    X_true = X_true.reshape(X_true.shape[0],
                                    X_true.shape[1] * X_true.shape[2] * X_true.shape[3])

    distancse = pairwise_euclidean_distance(X_true, predictions)
    mean_distance = torch.mean(distancse)

    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
    rbm.load(file='rbm.npz')

    best_threshold, best_rand_score = find_threshold(THRESHOLDS, test_dataloader, lbae, rbm)

    print(f'\n---------------------------------------------')
    print(f'Autoencoder')
    print(f'Pairwise euclidean distance: {round(mean_distance.item(), 3)}.')

    print(f'\n---------------------------------------------')
    print(f'RBM')
    print(f'Best threshold: {round(best_threshold, 3)}.')
    print(f'Best rand score: {round(best_rand_score, 3)}.')
    print(f'\n')

if __name__ == '__main__':
    main()