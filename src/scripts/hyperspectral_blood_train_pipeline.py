from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodDataset
import torch
import numpy as np

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

torch.set_float32_matmul_precision('medium')

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 32
NUM_HIDDEN = 17

MAX_EPOCHS = 25
RBM_STEPS = 1000
BATCH_SIZE = 16

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

def main():

    try:
        train_dataset = BloodDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH
        )

    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to provide paths to the hyperspectral data and ground truth data folders.\n"
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
        input_size=(1, 128),
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