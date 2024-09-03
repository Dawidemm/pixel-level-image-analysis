import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score

from src.utils.blood_dataset import BloodIterableDataset, Stage

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)


HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=19-step=290280.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

IMAGES = ['D_1', 'E_1', 'F_1']

EXPERIMENT_FOLDER_PATH = './kmeans/'

BACKGROUND_VALUE = 0

RANDOM_SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def main():

    os.makedirs(EXPERIMENT_FOLDER_PATH, exist_ok=True)
    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
        file.write(f'experiment,random_state,homogeneity,completeness\n')

    test_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.TEST,
        remove_noisy_bands=True,
        remove_background=True
    )

    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            drop_last=True
        )

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )

    lbae.eval()

    y_true = []
    hidden_representations = []

    X_true = []

    with torch.no_grad():
        for X, y in test_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()
    
            y_true.append(y)
            hidden_representations.append(hidden_representation)

            X_true.append(X)

    y_true = np.concatenate(y_true)

    hidden_representations = np.concatenate(hidden_representations)

    X_true = np.concatenate(X_true)
    X_true = X_true.reshape(X_true.shape[0], X_true.shape[2])

    experiment = 0

    for random_seed in RANDOM_SEEDS:

        lbae_kmeans = KMeans(n_clusters=7, random_state=random_seed)
        lbae_kmeans.fit(X_true)
        lbae_kmeans_clustering = lbae_kmeans.labels_
            
        homogeneity = homogeneity_score(y_true, lbae_kmeans_clustering)
        completeness = completeness_score(y_true, lbae_kmeans_clustering)

        with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
            file.write(f'{experiment},{random_seed},{homogeneity},{completeness}\n')

        experiment += 1

if __name__ == '__main__':
    main()