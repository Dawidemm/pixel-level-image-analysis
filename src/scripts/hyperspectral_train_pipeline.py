from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
import torch
import numpy as np

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

torch.set_float32_matmul_precision('medium')

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 16
NUM_HIDDEN = 17

MAX_EPOCHS = 100
RBM_STEPS = 1000
BATCH_SIZE = 8

HYPERSPECTRAL_IMAGE_PATH = 'dataset/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/ground_truth_image.tif'

def main():

    train_dataset = HyperspectralDataset(
        hyperspectral_image_path=HYPERSPECTRAL_IMAGE_PATH,
        ground_truth_image_path=GROUND_TRUTH_IMAGE_PATH,
        stage=Stage.TRAIN
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8, 
        shuffle=True, 
        num_workers=4
    )

    autoencoder = LBAE(
        input_size=(1, 16, 16),
        out_channels=8, 
        latent_size=NUM_VISIBLE,
        num_layers=2,
        quantize=list(range(MAX_EPOCHS))
    )
    
    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm)

    pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS, rbm_trainer='cd1')

if __name__ == '__main__':
    main()