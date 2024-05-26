import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM
from src.qbm4eo.pipeline import Pipeline


NUM_VISIBLE = 55
NUM_HIDDEN = 17

BATCH_SIZE = 16
MAX_EPOCHS = 25
RBM_STEPS = 1000
RBM_LEARNING_RATE = 0.001

SYNTH_IMG_SHAPE = 64

def main():

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
        stage=Stage.TRAIN
    )

    synthetic_dataloader = DataLoader(
        dataset=synthetic_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
    
    rbm = RBM(
        num_visible=NUM_VISIBLE, 
        num_hidden=NUM_HIDDEN
    )

    pipeline = Pipeline(
        auto_encoder=autoencoder, 
        rbm=rbm
    )

    pipeline.fit(
        synthetic_dataloader, 
        max_epochs=MAX_EPOCHS, 
        rbm_steps=RBM_STEPS, 
        rbm_trainer='cd1', 
        learnig_curve=True
    )


if __name__ == '__main__':
    main()