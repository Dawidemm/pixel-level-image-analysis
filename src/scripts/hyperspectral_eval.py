import torch
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
from src.utils.utils import ThresholdFinder

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 60
NUM_HIDDEN = 30

THRESHOLDS = np.linspace(1/100, 1, 100)

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x2678x614/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x2678x614/ground_truth_image.tif'

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_2/checkpoints/epoch=24-step=100.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_2/hparams.yaml'

RBM_WEIGHTS_PATH = 'rbm.npz'

def main():
    try:
        test_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.TEST
        )

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
              "1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "2. Run the training pipeline before starting the evaluation. \n"
              "   Training will generate a 'lightning_logs' folder containing model checkpoint and hparams files.\n"
              "3. The application will terminate now.")
        return

    lbae.eval()

    distance = []

    with torch.no_grad():
        for _, (X, _) in enumerate(test_dataloader):
            preds = lbae(X)
            preds = preds.reshape(preds.shape[0]*preds.shape[2], 1)
            X = X.reshape(X.shape[0]*X.shape[2], 1)
            dist = pairwise_euclidean_distance(X, preds)
            distance.append(torch.mean(dist))

    threshold_finder = ThresholdFinder(
        test_dataloader=test_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    best_threshold, best_rand_score = threshold_finder.find_threshold(THRESHOLDS)

    print(f'\n---------------------------------------------')
    print(f'Autoencoder')
    print(f'Pairwise euclidean distance: {round(torch.mean(torch.tensor(distance)).item(), 3)}.')

    print(f'\n---------------------------------------------')
    print(f'RBM')
    print(f'Best threshold: {round(best_threshold, 4)}.,')
    print(f'Best rand score: {round(best_rand_score, 4)},.')
    print(f'\n')

if __name__ == '__main__':
    main()