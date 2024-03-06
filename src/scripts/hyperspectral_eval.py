import torch
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from src.utils.hyperspectral_dataset import HyperspectralDataset
from src.utils.utils import ThresholdFinder

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 16
NUM_HIDDEN = 17

THRESHOLDS = np.linspace(1/10, 1, 10)

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
            predictions.append(lbae(X))

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

    threshold_finder = ThresholdFinder(
        test_dataloader=test_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    best_threshold, best_rand_score = threshold_finder.find_threshold(THRESHOLDS)

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