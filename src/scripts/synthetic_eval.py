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

THRESHOLDS = np.linspace(1/10, 1, 10)

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_1/checkpoints/epoch=24-step=5125.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_1/hparams.yaml'

RBM_WEIGHTS_PATH = 'rbm.npz'

def main():
    try:
        synthetic_data = utils.SyntheticDataGenerator(
            n_pixels=int(SYNTH_IMG_SHAPE*SYNTH_IMG_SHAPE),
            n_features=220,
            n_classes=17,
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
        rbm.load(file=RBM_WEIGHTS_PATH)
        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "\t1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "\t2. Run the training pipeline before starting the evaluation.\n"
              "The application will terminate now. ")
        return

    lbae.eval()

    y_true = []
    hidden = []

    with torch.no_grad():
        for X, y in synthetic_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()

            y_true.append(y)
            hidden.append(hidden_representation)

    y_true = np.concatenate(y_true)
    hidden = np.concatenate(hidden)

    sad_matrix = utils.spectral_angle_distance_matrix(hidden)

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

    print(f'\nLBAE+AHC Clustering:')
    print(f'Homogenity score: {round(ahc_homogeneity, 3)}')
    print(f'Completeness score: {round(ahc_completeness, 3)}\n')

    print(f'\nLBAE+Kmeans Clustering:')
    print(f'Homogenity score: {round(kmeans_homogeneity, 3)}')
    print(f'Completeness score: {round(kmeans_completeness, 3)}\n')

    threshold_finder = utils.ThresholdFinder(
        test_dataloader=synthetic_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    best_threshold, best_rand_score, adjusted_rand_score = threshold_finder.find_threshold(THRESHOLDS)

    print(f'\n---------------------------------------------')
    print(f'RBM')
    print(f'Best threshold: {round(best_threshold, 3)}.,')
    print(f'Best rand score: {round(best_rand_score, 3)},.')
    print(f'Adjusted rand score: {round(adjusted_rand_score, 3)},.')
    print(f'\n')

    rbm_labels = []

    for X, y in synthetic_dataloader:
        hidden_representation, _ = lbae.encoder(X, epoch=1)
        hidden_representation = hidden_representation.detach().numpy()
        rbm_label = rbm.binarized_rbm_output(hidden_representation, threshold=best_threshold)
        rbm_labels.append(rbm_label)

    rbm_labels = np.concatenate(rbm_labels)

    rbm_ahc_homogeneity = homogeneity_score(y_true, rbm_labels)
    rbm_ahc_completeness = completeness_score(y_true, rbm_labels)

    print(f'\nLBAE+RBM+AHC Clustering:')
    print(f'Homogenity score: {round(rbm_ahc_homogeneity, 3)}')
    print(f'Completeness score: {round(rbm_ahc_completeness, 3)}\n')

if __name__ == '__main__':
    main()