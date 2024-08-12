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

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

IMAGES = ['D_1', 'E_1', 'F_1']

BACKGROUND_VALUE = 0

def main():

    for image in IMAGES:

        blood_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH ,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            load_specific_image=image,
            remove_noisy_bands=False,
            stage=Stage.IMG_SEG
        )

        blood_dataloader = DataLoader(
            dataset=blood_dataset,
            batch_size=1
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
            for X, y in blood_dataloader:
                hidden_representation, _ = lbae.encoder(X, epoch=1)
                hidden_representation = hidden_representation.detach().numpy()
    
                y_true.append(y)
                hidden_representations.append(hidden_representation)

        y_true = np.concatenate(y_true)
        background_values = np.sum(y_true == BACKGROUND_VALUE)

        hidden_representations = np.concatenate(hidden_representations)

        lbae_hdbscan = HDBSCAN()
        lbae_hdbscan.fit(hidden_representations)
        lbae_hdbscan_clustering = lbae_hdbscan.labels_

        lbae_hdbscan_homogeneity = homogeneity_score(y_true, lbae_hdbscan_clustering)
        lbae_hdbscan_completeness = completeness_score(y_true, lbae_hdbscan_clustering)

        lbae_kmeans = KMeans(n_clusters=8)
        lbae_kmeans.fit(hidden_representations)
        lbae_kmeans_clustering = lbae_kmeans.labels_

        lbae_kmeans_homogeneity = homogeneity_score(y_true, lbae_kmeans_clustering)
        lbae_kmeans_completeness = completeness_score(y_true, lbae_kmeans_clustering)

        lbae_hdbscan_clustering = lbae_hdbscan_clustering.reshape((520, 696))
        lbae_kmeans_clustering = lbae_kmeans_clustering.reshape((520, 696))
        y_true = y_true.reshape((520, 696))

        fig, axes = plt.subplots(1, 3, figsize=(10, 6))

        ax = axes[0]
        ax.imshow(lbae_hdbscan_clustering)
        ax.set_title(f'LBAE+HDBSCAN')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(lbae_kmeans_clustering)
        ax.set_title(f'LBAE+KMeans')
        ax.axis('off')

        ax = axes[2]
        ax.imshow(y_true)
        ax.set_title(f'Ground Truth (Background: {(background_values/y_true.size)*100}%)')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'clustering_comparison_{image}.png', dpi=300)
        
        with open(f'lbae_{image}_metrics.txt', 'a+') as file:
            file.write(f'LBAE+HDBSCAN Clustering:\n')
            file.write(f'Homogenity score: {round(lbae_hdbscan_homogeneity, 3)}.\n')
            file.write(f'Completeness score: {round(lbae_hdbscan_completeness, 3)}.\n')

            file.write(f'LBAE+Kmeans Clustering:\n')
            file.write(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}.\n')
            file.write(f'Completeness score: {round(lbae_kmeans_completeness, 3)}.\n')

if __name__ == '__main__':
    main()