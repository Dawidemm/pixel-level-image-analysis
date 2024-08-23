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


NUM_HIDDEN = 8

AUTOENCODER_EPOCHS = 25
AUTOENCODER_LEARNING_RATE = [0.01, 0.001, 0.0001]
BATCH_SIZE = [8, 16, 32]
NOISY_BANDS = [True, False]

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
        file.write(f'experiment,batch_size, learning_rate, noisy_bands, \n')

    experiment = 0

    for noisy_band in NOISY_BANDS:
        for learning_rate in AUTOENCODER_LEARNING_RATE:
            for batch_size in BATCH_SIZE:
        
                train_dataset = BloodIterableDataset(
                    hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                    ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                    load_specific_images=IMAGES,
                    stage=Stage.TRAIN,
                    remove_noisy_bands=noisy_band,
                    remove_background=True,
                    shuffle=True
                )

                val_dataset = BloodIterableDataset(
                    hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                    ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                    load_specific_images=IMAGES,
                    stage=Stage.VAL,
                    remove_noisy_bands=noisy_band,
                    remove_background=True,
                    shuffle=True
                )

                test_dataset = BloodIterableDataset(
                    hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                    ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                    load_specific_images=IMAGES,
                    stage=Stage.TEST,
                    remove_noisy_bands=noisy_band,
                    remove_background=True,
                    shuffle=False
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

                if noisy_band == True:
                    input_size = (1, 112)
                    num_visible = 28
                else:
                    input_size = (1, 128)
                    num_visible = 32

                lbae = LBAE(
                    input_size=input_size,
                    out_channels=8, 
                    latent_size=num_visible,
                    learning_rate=learning_rate,
                    num_layers=2,
                    quantize=list(range(AUTOENCODER_EPOCHS))
                )
                
                rbm = RBM(num_visible, NUM_HIDDEN)

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

                mean_distances = []

                with torch.no_grad():
                    for X, _ in test_dataloader:
                        prediction = lbae(X)
                        prediction = prediction.reshape(prediction.shape[0]*prediction.shape[2], 1)
                        X = X.reshape(X.shape[0]*X.shape[2], 1)
                        distance = pairwise_euclidean_distance(X, prediction)
                        mean_distances.append(torch.mean(distance))


                pairwise_euclidean_distance_mean = round(torch.mean(torch.tensor(mean_distances)).item(), 3)

                with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
                            file.write(f'{experiment},{batch_size},{learning_rate},noisy_bands={noisy_band},{pairwise_euclidean_distance_mean}\n')

                experiment += 1

if __name__ == '__main__':
    main()