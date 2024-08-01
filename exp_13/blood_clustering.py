import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import completeness_score, homogeneity_score

from src.utils.blood_dataset import BloodIterableDataset, Stage
from src.utils import utils

from src.qbm4eo.lbae import LBAE


np.random.seed(10)
torch.manual_seed(0)

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

def main():
    try:
        
        blood_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH ,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            num_images_to_load=1,
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
        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "\t1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "\t2. Run the training pipeline before starting the evaluation.\n"
              "The application will terminate now.")
        return

    lbae.eval()

    X_true = []
    y_true = []
    hidden_representations = []

    with torch.no_grad():
        for X, y in blood_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()

            X_true.append(y)
            y_true.append(y)
            hidden_representations.append(hidden_representation)

    X_true = np.concatenate(X_true)
    y_true = np.concatenate(y_true)
    hidden_representations = np.concatenate(hidden_representations)

    # sad_lbae_matrix= utils.spectral_angle_distance_matrix(
    #     objects=hidden_representations,
    # )

    # lbae_ahc = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='average')
    # lbae_ahc.fit(sad_lbae_matrix)
    lbae_ahc = AgglomerativeClustering(n_clusters=8)
    lbae_ahc.fit(hidden_representations)
    lbae_ahc_clustering = lbae_ahc.labels_

    # lbae_kmeans = KMeans(n_clusters=8, metric='precomputed', linkage='average')
    # lbae_kmeans.fit(sad_lbae_matrix)
    lbae_kmeans = KMeans(n_clusters=8)
    lbae_kmeans.fit(hidden_representations)
    lbae_kmeans_clustering = lbae_kmeans.labels_

    lbae_ahc_homogeneity = homogeneity_score(y_true, lbae_ahc_clustering)
    lbae_ahc_completeness = completeness_score(y_true, lbae_ahc_clustering)

    lbae_kmeans_homogeneity = homogeneity_score(y_true, lbae_kmeans_clustering)
    lbae_kmeans_completeness = completeness_score(y_true, lbae_kmeans_clustering)

    # sad_matrix= utils.spectral_angle_distance_matrix(
    #     objects=X_true,
    # )

    # ahc = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='average')
    # ahc.fit(sad_matrix)
    ahc = AgglomerativeClustering(n_clusters=8)
    ahc.fit(X_true)
    ahc_clustering = ahc.labels_

    # kmeans = KMeans(n_clusters=8, metric='precomputed', linkage='average')
    # kmeans.fit(sad_matrix)
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(X_true)
    kmeans_clustering = kmeans.labels_

    ahc_homogeneity = homogeneity_score(y_true, ahc_clustering)
    ahc_completeness = completeness_score(y_true, ahc_clustering)

    kmeans_homogeneity = homogeneity_score(y_true, kmeans_clustering)
    kmeans_completeness = completeness_score(y_true, kmeans_clustering)

    os.makedirs('./exp_13', exist_ok=True)
    with open('exp_13/lbae_metrics.txt', 'a+') as file:
        file.write(f'LBAE+AHC Clustering:')
        file.write(f'Homogenity score: {round(lbae_ahc_homogeneity, 3)}')
        file.write(f'Completeness score: {round(lbae_ahc_completeness, 3)}\n')

        file.write(f'LBAE+Kmeans Clustering:')
        file.write(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}')
        file.write(f'Completeness score: {round(lbae_kmeans_completeness, 3)}\n')

        file.write(f'AHC Clustering:')
        file.write(f'Homogenity score: {round(ahc_homogeneity, 3)}')
        file.write(f'Completeness score: {round(ahc_completeness, 3)}\n')

        file.write(f'Kmeans Clustering:')
        file.write(f'Homogenity score: {round(kmeans_homogeneity, 3)}')
        file.write(f'Completeness score: {round(kmeans_completeness, 3)}\n')

if __name__ == '__main__':
    main()