from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset
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

def main():

    train_dataset = HyperspectralDataset(
    hyperspectral_image_path='dataset/hyperspectral_image.tif',
    ground_truth_image_path='dataset/ground_truth_image.tif',
    stage='train'
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)

    autoencoder = LBAE(input_size=(1, 16, 16), out_channels=8, zsize=NUM_VISIBLE, num_layers=2, quantize=list(range(MAX_EPOCHS)))
    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm, classifier=True)

    pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS)

if __name__ == '__main__':
    main()