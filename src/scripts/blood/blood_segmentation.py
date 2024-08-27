import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import completeness_score, homogeneity_score
import matplotlib.pyplot as plt

from src.utils.blood_dataset import BloodIterableDataset, Stage
from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)


HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=19-step=58060.ckpt'
AUTOENCODER_HPARAMS_PATH = 'lightning_logs/version_0/hparams.yaml'

IMAGES = ['D_1', 'E_1', 'F_1']

BACKGROUND_VALUE = 0

def main():

    test_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.TEST,
        remove_noisy_bands=True,
        remove_background=True
    )

    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            drop_last=True
        )

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )

    lbae.eval()

    y_true = []
    hidden_representations = []

    with torch.no_grad():
        for X, y in test_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()
    
            y_true.append(y)
            hidden_representations.append(hidden_representation)

    y_true = np.concatenate(y_true)

    hidden_representations = np.concatenate(hidden_representations)

        # lbae_hdbscan = HDBSCAN()
        # lbae_hdbscan.fit(hidden_representations)
        # lbae_hdbscan_clustering = lbae_hdbscan.labels_

        # lbae_hdbscan_homogeneity = homogeneity_score(y_true, lbae_hdbscan_clustering)
        # lbae_hdbscan_completeness = completeness_score(y_true, lbae_hdbscan_clustering)

    lbae_kmeans = KMeans(n_clusters=7)
    lbae_kmeans.fit(hidden_representations)
    lbae_kmeans_clustering = lbae_kmeans.labels_
        
    lbae_kmeans_homogeneity = homogeneity_score(y_true, lbae_kmeans_clustering)
    lbae_kmeans_completeness = completeness_score(y_true, lbae_kmeans_clustering)
        
        # with open(f'lbae_{image}_metrics.txt', 'a+') as file:
        #     file.write(f'LBAE+HDBSCAN Clustering:\n')
        #     file.write(f'Homogenity score: {round(lbae_hdbscan_homogeneity, 3)}.\n')
        #     file.write(f'Completeness score: {round(lbae_hdbscan_completeness, 3)}.\n')

        #     file.write(f'LBAE+Kmeans Clustering:\n')
        #     file.write(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}.\n')
        #     file.write(f'Completeness score: {round(lbae_kmeans_completeness, 3)}.\n')

    print(f'LBAE+Kmeans Clustering:\n')
    print(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}.\n')
    print(f'Completeness score: {round(lbae_kmeans_completeness, 3)}.\n')

if __name__ == '__main__':
    main()