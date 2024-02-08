from torch.utils.data import DataLoader
from hyperspectral_dataset import HyperspectralDataset
import torch

from lbae import LBAE
from pipeline import Pipeline
from rbm import RBM

torch.set_float32_matmul_precision('medium')

NUM_VISIBLE = 60
NUM_HIDDEN = 40

MAX_EPOCHS = 50
RBM_STEPS = 1000
BATCH_SIZE = 8

def main():

    dataset = HyperspectralDataset(
    hyperspectral_image_path='dataset/hyperspectral_image.tif',
    ground_truth_image_path='dataset/ground_truth_image.tif',
    stage='train'
    )

    train_dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    autoencoder = LBAE(input_size=(1, 16, 16), out_channels=8, zsize=NUM_VISIBLE, num_layers=2, quantize=list(range(MAX_EPOCHS)))
    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    pipeline = Pipeline(auto_encoder=autoencoder, rbm=True, classifier=True)

    pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS)

if __name__ == '__main__':
    main()