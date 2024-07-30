import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from src.utils.blood_dataset import BloodIterableDataset, Stage

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from src.utils import utils


np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 32
NUM_HIDDEN = 16

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

RBM_WEIGHTS_PATH = 'exp_13/exp_13_rbm.npz'

THRESHOLD = 0.4

def main():
    try:
        
        blood_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH ,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            num_images_to_load=1,
            remove_noisy_bands=False,
            stage=Stage.IMG_SEG
        )

        blood_dataloader = DataLoader(
            dataset=blood_dataset,
            batch_size=1
        )

        lbae = LBAE.load_from_checkpoint(
            checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
            hparams_file=AUTOENCODER_HPARAMS_PATH,
            map_location=torch.device('cpu')
        )

        rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
        rbm.load(file='experiments/experiments/exp_13/rbm.npz')

        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "\t1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "\t2. Run the training pipeline before starting the evaluation.\n"
              "The application will terminate now.")
        return

    lbae.eval()

    y_true = []
    hidden_representations = []
    rbm_labels = []

    with torch.no_grad():
        for X, y in blood_dataloader:
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()
            rbm_label = rbm.binarized_rbm_output(hidden_representation, threshold=THRESHOLD)

            y_true.append(y)
            hidden_representations.append(hidden_representation)
            rbm_labels.append(rbm_label)

    spectral_distance = utils.spectral_angle_distance_matrix(
        objects=hidden_representations,
        rbm_labels=rbm_labels
    )

    ahc = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='average')
    ahc.fit(spectral_distance)

    ahc_segmentation = ahc.labels_

    y_true = np.concatenate(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax1 = axes[0]
    ax1.imshow(ahc_segmentation.reshape(520, 696), cmap='gray')
    ax1.set_title('RBM+AHC Segmentation')
    ax1.axis('off') 

    ax2 = axes[1]
    ax2.imshow(y_true.reshape(520, 696), cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    plt.tight_layout()

    plt.savefig('comparison.png', dpi=300)

if __name__ == '__main__':
    main()