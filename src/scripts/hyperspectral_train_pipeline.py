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

NUM_VISIBLE = 68
NUM_HIDDEN = 34

MAX_EPOCHS = 25
RBM_STEPS = 1000
BATCH_SIZE = 32

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

def main():

    try:
        train_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.TRAIN
        )

    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to provide paths to the hyperspectral image and ground truth image files.\n"
              "The application will terminate now.")
        return

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

    pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm)

    pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS, rbm_steps=RBM_STEPS, rbm_trainer='cd1', learnig_curve=True)

if __name__ == '__main__':
    main()