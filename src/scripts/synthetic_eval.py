import torch
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import completeness_score, homogeneity_score

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 55
NUM_HIDDEN = 17

SYNTH_IMG_SHAPE = 64
N_FEATURES = 220
N_CLASSES = 17

THRESHOLDS = np.linspace(1/10, 1, 10)

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=24-step=5125.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_0/hparams.yaml'

LBAE_RBM_WEIGHTS_PATH = 'rbm.npz'
RBM_WEIGHTS_PATH = 'synthetic_rbm.npz'

def main():
    
    synthetic_data = utils.SyntheticDataGenerator(
        n_pixels=int(SYNTH_IMG_SHAPE*SYNTH_IMG_SHAPE),
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        image_width=SYNTH_IMG_SHAPE,
        image_height=SYNTH_IMG_SHAPE
    )

    synthetic_hyperspectral_image, synthetic_ground_truth_image = synthetic_data.generate_synthetic_data()

    synthetic_dataset = HyperspectralDataset(
        hyperspectral_data=synthetic_hyperspectral_image,
        ground_truth_data=synthetic_ground_truth_image,
        stage=Stage.IMG_SEG
    )

    synthetic_dataloader = DataLoader(
        dataset=synthetic_dataset,
        batch_size=1,
        shuffle=False
    )

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODE_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )

    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
    rbm.load(file=LBAE_RBM_WEIGHTS_PATH)

    lbae.eval()

    X_true = []
    y_true = []
    hidden = []

    with torch.no_grad():
        for X, y in synthetic_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()

            X_true.append(X)
            y_true.append(y)
            hidden.append(hidden_representation)

    X_true = np.concatenate(X_true).reshape((int(SYNTH_IMG_SHAPE**2), N_FEATURES))
    y_true = np.concatenate(y_true)
    hidden = np.concatenate(hidden)

    sad_matrix = utils.spectral_angle_distance_matrix(X_true)

    ahc = AgglomerativeClustering(n_clusters=17)
    ahc.fit(sad_matrix)
    ahc_segmentation = ahc.labels_

    kmeans = KMeans(n_clusters=17)
    kmeans.fit(sad_matrix)
    kmeans_segmentation = kmeans.labels_

    ahc_homogeneity = homogeneity_score(y_true, ahc_segmentation)
    ahc_completeness = completeness_score(y_true, ahc_segmentation)

    kmeans_homogeneity = homogeneity_score(y_true, kmeans_segmentation)
    kmeans_completeness = completeness_score(y_true, kmeans_segmentation)

    print(f'\nAHC Clustering:')
    print(f'Homogenity score: {round(ahc_homogeneity, 3)}')
    print(f'Completeness score: {round(ahc_completeness, 3)}\n')

    print(f'\nKmeans Clustering:')
    print(f'Homogenity score: {round(kmeans_homogeneity, 3)}')
    print(f'Completeness score: {round(kmeans_completeness, 3)}\n')

    lbae_sad_matrix = utils.spectral_angle_distance_matrix(hidden)

    lbae_ahc = AgglomerativeClustering(n_clusters=17)
    lbae_ahc.fit(lbae_sad_matrix)
    lbae_ahc_segmentation = lbae_ahc.labels_

    lbae_kmeans = KMeans(n_clusters=17)
    lbae_kmeans.fit(lbae_sad_matrix)
    lbae_kmeans_segmentation = lbae_kmeans.labels_

    lbae_ahc_homogeneity = homogeneity_score(y_true, lbae_ahc_segmentation)
    lbae_ahc_completeness = completeness_score(y_true, lbae_ahc_segmentation)

    lbae_kmeans_homogeneity = homogeneity_score(y_true, lbae_kmeans_segmentation)
    lbae_kmeans_completeness = completeness_score(y_true, lbae_kmeans_segmentation)

    print(f'\nLBAE+AHC Clustering:')
    print(f'Homogenity score: {round(lbae_ahc_homogeneity, 3)}')
    print(f'Completeness score: {round(lbae_ahc_completeness, 3)}\n')

    print(f'\nLBAE+Kmeans Clustering:')
    print(f'Homogenity score: {round(lbae_kmeans_homogeneity, 3)}')
    print(f'Completeness score: {round(lbae_kmeans_completeness, 3)}\n')

    threshold_finder = utils.ThresholdFinder(
        test_dataloader=synthetic_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    best_threshold, adjusted_rand_score, rand_score, rbm_homogeneity, rbm_completeness = threshold_finder.find_threshold(THRESHOLDS)

    print(f'\n---------------------------------------------')
    print(f'LBAE+RBM')
    print(f'Best threshold: {round(best_threshold, 3)}.,')
    print(f'Adjusted rand score: {round(adjusted_rand_score, 3)},.')
    print(f'Rand score: {round(rand_score, 3)},.')
    
    print(f'\nLBAE+RBM Clustering:')
    print(f'Homogenity score: {round(rbm_homogeneity, 3)}')
    print(f'Completeness score: {round(rbm_completeness, 3)}\n')
    
    synthetic_hyperspectral_image, synthetic_ground_truth_image = synthetic_data.generate_synthetic_data(quantize=True)

    synthetic_dataset = HyperspectralDataset(
        hyperspectral_data=synthetic_hyperspectral_image,
        ground_truth_data=synthetic_ground_truth_image,
        stage=Stage.IMG_SEG
    )

    synthetic_dataloader = DataLoader(
        dataset=synthetic_dataset,
        batch_size=1,
        shuffle=False
    )

    rbm = RBM(num_visible=220, num_hidden=NUM_HIDDEN)
    rbm.load(file=RBM_WEIGHTS_PATH)

    threshold_finder = utils.ThresholdFinder(
        test_dataloader=synthetic_dataloader,
        rbm=rbm,
        encoder=None
    )

    best_threshold, adjusted_rand_score, rand_score, rbm_homogeneity, rbm_completeness = threshold_finder.find_threshold(THRESHOLDS)

    print(f'\n---------------------------------------------')
    print(f'RBM')
    print(f'Best threshold: {round(best_threshold, 3)}.,')
    print(f'Adjusted rand score: {round(adjusted_rand_score, 3)},.')
    print(f'Rand score: {round(rand_score, 3)},.')
    
    print(f'\nRBM Clustering:')
    print(f'Homogenity score: {round(rbm_homogeneity, 3)}')
    print(f'Completeness score: {round(rbm_completeness, 3)}\n')
    print(f'\n')


if __name__ == '__main__':
    main()