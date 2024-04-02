from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
import torch
import numpy as np

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM
from src.qbm4eo.classifier import Classifier

torch.set_float32_matmul_precision('medium')

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 60
NUM_HIDDEN = 30

MAX_EPOCHS = 50
RBM_STEPS = 1000
BATCH_SIZE = 32

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

def main():

    train_dataset = HyperspectralDataset(
        hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
        ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
        stage=Stage.TRAIN
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True
    )

    autoencoder = LBAE(
        input_size=(1, 220),
        out_channels=8, 
        latent_size=NUM_VISIBLE,
        num_layers=2,
        quantize=list(range(MAX_EPOCHS))
    )
    
    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    classifier = Classifier(output_size=17, encoder=autoencoder.encoder, rbm=rbm)

    pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm, classifier=classifier)

    pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS, rbm_trainer='cd1', skip_rbm=True)

if __name__ == '__main__':
    main()