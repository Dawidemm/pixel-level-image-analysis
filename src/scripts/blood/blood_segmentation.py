import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import completeness_score, homogeneity_score

from src.utils.blood_dataset import BloodIterableDataset, Stage

from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from src.utils import utils

import matplotlib.pyplot as plt

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

# IMAGES = ['D_1', 'E_1', 'F_1']
IMAGES = ['D_1']

EXPERIMENT_FOLDER_PATH = './kmeans/'

# PARTITIONS = [(13442*i, 13442+13442*i) for i in range(7)]
# PARTITIONS = [(47047*i, 47047+47047*i) for i in range(2)]
PARTITIONS = 1

def main():

    test_labels = np.array([])

    for partition in range(PARTITIONS):
    
        seg_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            load_specific_images=IMAGES,
            stage=Stage.SEG,
            remove_noisy_bands=True,
            remove_background=True,
            shuffle=False,
            partition=None
        )

        seg_dataloader = DataLoader(
                dataset=seg_dataset,
                batch_size=1,
                drop_last=False
            )

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


        with torch.no_grad():
            for idx, (X, y) in enumerate(seg_dataloader):
                if y == 0:
                    continue
                else:
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

        ahc = AgglomerativeClustering(n_clusters=7, metric='precomputed', linkage='average')
        labels = ahc.fit_predict(sad_matrix)
        labels = labels + 1
        labels = list(labels)

        test_labels = np.append(test_labels, labels)


    # homogenity = homogeneity_score(y_true, labels)
    # completeness = completeness_score(y_true, labels)

    # print(f'Homogenity: {round(homogenity, 3)}')
    # print(f'Completeness: {round(completeness, 3)}')

    seg_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            load_specific_images=IMAGES,
            stage=Stage.SEG,
            remove_noisy_bands=True,
            remove_background=False,
            shuffle=False,
            partition=None
        )
    
    seg_dataloader = DataLoader(
                dataset=seg_dataset,
                batch_size=1,
                drop_last=False
            )

    segmented_img = []
    counter = 0

    for idx, (X, y) in enumerate(seg_dataloader):
        if y.item() == 0:
            segmented_img.append(y.item())
        else:
            segmented_img.append(test_labels[counter])
            counter += 1

    segmented_img = np.array(segmented_img)
    segmented_img = np.reshape(segmented_img, ((520, 696)))

    plt.figure(figsize=(10, 8))
    plt.imshow(segmented_img)
    plt.title('F_1 Image Segmentation')
    plt.axis('off')
    plt.savefig('image_segmentation.png', dpi=300)


if __name__ == '__main__':
    main()