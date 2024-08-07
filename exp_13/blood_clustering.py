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

NUM_VISIBLE = 28
NUM_HIDDEN = 8

THRESHOLDS = np.linspace(1/10, 1, 10)

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

RBM_WEIGHTS_PATH = 'exp_13/exp_13_rbm.npz'

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

        rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
        rbm.load(file=RBM_WEIGHTS_PATH)
        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "\t1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "\t2. Run the training pipeline before starting the evaluation.\n"
              "The application will terminate now.")
        return

    lbae.eval()

    X_true = []
    reconstruced_img = []
    y_true = []
    hidden_representations = []

    with torch.no_grad():
        for X, y in blood_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)

            X_reconstrucetd = lbae.decoder(hidden_representation)
            reconstruced_img.append(X_reconstrucetd)

            hidden_representation = hidden_representation.detach().numpy()
            

            X_true.append(X)
            y_true.append(y)
            hidden_representations.append(hidden_representation)

    X_true = np.concatenate(X_true)
    X_true = X_true.reshape(X_true.shape[0], X_true.shape[2])
    X_true2 = X_true.reshape((520, 696, 128))
    reconstruced_img = np.concatenate(reconstruced_img)
    reconstruced_img = reconstruced_img.reshape((520, 696, reconstruced_img.shape[2]))

    y_true = np.concatenate(y_true)
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


    threshold_finder = utils.ThresholdFinder(
        dataloader=blood_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    _, _, _, rbm_homogeneity, rbm_completeness, rbm_labels = threshold_finder.find_threshold(THRESHOLDS)

    lbae_hdbscan_clustering = lbae_hdbscan_clustering.reshape((520, 696))
    lbae_kmeans_clustering = lbae_kmeans_clustering.reshape((520, 696))
    rbm_labels = rbm_labels.reshape((520, 696))
    y_true = y_true.reshape((520, 696))

    fig, axes = plt.subplots(1, 4, figsize=(6, 12))

    ax = axes[0]
    ax.imshow(lbae_hdbscan_clustering)
    ax.set_title(f'LBAE+HDBSCAN')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(lbae_kmeans_clustering)
    ax.set_title(f'LBAE+KMeans')
    ax.axis('off')

    ax = axes[2]
    ax.imshow(rbm_labels)
    ax.set_title(f'LBAE+RBM')
    ax.axis('off')

    ax = axes[3]
    ax.imshow(y_true)
    ax.set_title(f'Ground Truth')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'./exp_13/clustering_comparison.png', dpi=300)
    
    # noisy_bands_indices = [0,1,2,3,4,48,49,121,122,123,124,125,126,127]
    # bands_to_display_indices = sorted(noisy_bands_indices + [i*10 for i in range(1, 13)])

    # for i in range(len(bands_to_display_indices)):

    #     fig, axes = plt.subplots(1, 2, figsize=(6, 6))

    #     ax = axes[0]
    #     ax.imshow(reconstruced_img[:, :, bands_to_display_indices[i]])
    #     ax.set_title(f'X_rec. band: {bands_to_display_indices[i]}')
    #     ax.axis('off')

    #     ax = axes[1, ]
    #     ax.imshow(X_true2[:, :, bands_to_display_indices[i]])
    #     ax.set_title(f'X_true band: {bands_to_display_indices[i]}')
    #     ax.axis('off')

    #     plt.tight_layout()

    #     os.makedirs('./exp_13/LBAE/', exist_ok=True)
    #     if bands_to_display_indices[i] in noisy_bands_indices:
    #         plt.savefig(f'./exp_13/LBAE/LBAE_Xtrue_vs_Xrecontruced-noisy_band-{bands_to_display_indices[i]}.png', dpi=300)
    #     else:
    #         plt.savefig(f'./exp_13/LBAE/LBAE_Xtrue_vs_Xrecontruced-band-{bands_to_display_indices[i]}.png', dpi=300)


    os.makedirs('./exp_13', exist_ok=True)
    with open('exp_13/lbae_metrics.txt', 'a+') as file:
        file.write(f'LBAE+HDBSCAN Clustering:\n')
        file.write(f'Homogenity score: {round(lbae_hdbscan_homogeneity, 3)}.\n')
        file.write(f'Completeness score: {round(lbae_hdbscan_completeness, 3)}.\n')

        file.write(f'LBAE+Kmeans Clustering:\n')
        file.write(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}.\n')
        file.write(f'Completeness score: {round(lbae_kmeans_completeness, 3)}.\n')

        file.write(f'LBAE+RBM Clustering:\n')
        file.write(f'Homogenity score: {round(rbm_homogeneity, 3)}.\n')
        file.write(f'Completeness score: {round(rbm_completeness, 3)}.\n')

if __name__ == '__main__':
    main()