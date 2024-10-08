import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import completeness_score, homogeneity_score
import joblib

from src.utils.blood_dataset import BloodIterableDataset, Stage

from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from src.utils import utils

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 28
NUM_HIDDEN = 26
RANDOM_SEED = 60


HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=19-step=290280.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'
RBM_MODEL_PATH = 'experiments_nh_26_best/exp_6/rbm.npz'

IMAGES = ['D_1', 'E_1', 'F_1']

EXPERIMENT_FOLDER_PATH = './kmeans/'


def main():

    train_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.TRAIN,
        remove_noisy_bands=True,
        remove_background=True
    )

    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            drop_last=False
        )
    
    val_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.VAL,
        remove_noisy_bands=True,
        remove_background=True
    )

    val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            drop_last=False
        )
    
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
            drop_last=False
        )
    
    # dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    dataloaders = [test_dataloader]

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )

    lbae.eval()

    rbm = RBM(
        num_visible=NUM_VISIBLE,
        num_hidden=NUM_HIDDEN,
        random_seed=RANDOM_SEED
    )

    rbm.load(file=RBM_MODEL_PATH)

    y_true = []
    hidden_representations = []
    rbm_labels = []

    X_true = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for X, y in dataloader:
                hidden_representation, _ = lbae.encoder(X, epoch=1)
                hidden_representation = hidden_representation.detach().numpy()
                rbm_label = rbm.binarized_rbm_output(hidden_representation, threshold=0.4)
        
                hidden_representations.append(hidden_representation)
                rbm_labels.append(rbm_label)

                y_true.append(y)

    hidden_representations = np.concatenate(hidden_representations)
    y_true = np.concatenate(y_true)

    print('Strat agglomerative clustering...')
    sad_matrix = utils.spectral_angle_distance_matrix(hidden_representations, rbm_labels)

    model = AgglomerativeClustering(n_clusters=7, metric='precomputed', linkage='average')
    labels = model.fit_predict(sad_matrix)
    
    homogenity = homogeneity_score(y_true, labels)
    completeness = completeness_score(y_true, labels)

    joblib.dump(model, 'ahc.joblib')

    print(f'Homogenity: {round(homogenity, 3)}')
    print(f'Completeness: {round(completeness, 3)}')


if __name__ == '__main__':
    main()