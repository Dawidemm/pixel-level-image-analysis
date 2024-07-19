import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from src.utils import utils

from torchmetrics.functional.pairwise import pairwise_euclidean_distance

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 28
NUM_HIDDEN = 8

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'ares2/pixel-level-image-analysis/lightning_logs/version_0/checkpoints/epoch=9-step=2940600.ckpt'
AUTOENCODER_HPARAMS_PATH = 'ares2/pixel-level-image-analysis/lightning_logs/version_0/hparams.yaml'

RBM_WEIGHTS_PATH = 'ares2/pixel-level-image-analysis/rbm.npz'

THRESHOLDS = np.linspace(1/10, 1, 10)

def main():
    try:
        
        blood_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH ,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            num_images_to_load=1,
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

    mean_distances = []

    with torch.no_grad():
        for X, _ in blood_dataloader:
            prediction = lbae(X)
            prediction = prediction.reshape(prediction.shape[0]*prediction.shape[2], 1)
            X = X.reshape(X.shape[0]*X.shape[2], 1)
            distance = pairwise_euclidean_distance(X, prediction)
            mean_distances.append(torch.mean(distance))

    print(f'\n---------------------------------------------')
    print(f'Autoencoder')
    print(f'Pairwise euclidean distance: {round(torch.mean(torch.tensor(mean_distances)).item(), 3)}.')

    threshold_finder = utils.ThresholdFinder(
        dataloader=blood_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    best_threshold, adjusted_rand_score, rand_score, rbm_homogeneity, rbm_completeness = threshold_finder.find_threshold(THRESHOLDS)

    print(f'\n---------------------------------------------')
    print(f'LBAE+RBM')
    print(f'Best threshold: {round(best_threshold, 3)}.')
    print(f'Adjusted rand score: {round(adjusted_rand_score, 3)}.')
    print(f'Rand score: {round(rand_score, 3)}.')
    
    print(f'\nLBAE+RBM Clustering:')
    print(f'Homogenity score: {round(rbm_homogeneity, 3)}')
    print(f'Completeness score: {round(rbm_completeness, 3)}\n')

if __name__ == '__main__':
    main()