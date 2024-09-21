from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage
import torch
import numpy as np
import os

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

from torchmetrics.functional.pairwise import pairwise_euclidean_distance

torch.set_float32_matmul_precision('medium')

NUM_VISIBLE = 28
NUM_HIDDEN = 8

AUTOENCODER_EPOCHS = 20
AUTOENCODER_LEARNING_RATE = [0.001]
BATCH_SIZE = [8]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
IMAGES = ['D_1', 'E_1', 'F_1']

RANDOM_SEED = 10

EXPERIMENT_FOLDER_PATH = './experiments/'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():

    os.makedirs(EXPERIMENT_FOLDER_PATH, exist_ok=True)
    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
        file.write(f'experiment,batch_size,learning_rate,pairwise_euclidean_distance,spectral_angle_distance\n')

    experiment = 0

    for learning_rate in AUTOENCODER_LEARNING_RATE:
        for batch_size in BATCH_SIZE:
        
            train_dataset = BloodIterableDataset(
                hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                load_specific_images=IMAGES,
                stage=Stage.TRAIN,
                remove_noisy_bands=True,
                remove_background=True,
                shuffle=True
            )

            val_dataset = BloodIterableDataset(
                hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                load_specific_images=IMAGES,
                stage=Stage.VAL,
                remove_noisy_bands=True,
                remove_background=True,
                shuffle=True
            )

            test_dataset = BloodIterableDataset(
                hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                load_specific_images=IMAGES,
                stage=Stage.TEST,
                remove_noisy_bands=True,
                remove_background=True,
                shuffle=True
            )

            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size
            )

            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size
            )

            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size
            )

            lbae = LBAE(
                input_size=(1, 112),
                out_channels=8, 
                latent_size=NUM_VISIBLE,
                learning_rate=learning_rate,
                num_layers=2,
                quantize=list(range(AUTOENCODER_EPOCHS))
            )
                
            rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

            pipeline = Pipeline(auto_encoder=lbae, rbm=rbm)

            pipeline.fit(
                train_data_loader=train_dataloader, 
                validation_data_loader=val_dataloader,
                autoencoder_epochs=AUTOENCODER_EPOCHS,
                skip_rbm=True,
                rbm_trainer='cd1',
                learnig_curve=True,
                experiment_folder_path=EXPERIMENT_FOLDER_PATH,
                experiment_number=experiment
            )

            mean_euclidean_distances = []

            with torch.no_grad():
                for X, _ in test_dataloader:
                    X_reconstructed = lbae(X)
                    X_reconstructed = X_reconstructed.reshape(X_reconstructed.shape[0]*X_reconstructed.shape[2], 1)
                    X = X.reshape(X.shape[0]*X.shape[2], 1)
                    distance = pairwise_euclidean_distance(X, X_reconstructed)
                    mean_euclidean_distances.append(torch.mean(distance))


            pairwise_euclidean_distance_mean = round(torch.mean(torch.tensor(mean_euclidean_distances)).item(), 3)

            with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
                file.write(f'{experiment},{batch_size},{learning_rate},{pairwise_euclidean_distance_mean}\n')

            experiment += 1

if __name__ == '__main__':
    main()