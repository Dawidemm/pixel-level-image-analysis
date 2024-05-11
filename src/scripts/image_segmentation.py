import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import completeness_score, homogeneity_score
from src.utils.utils import distance_matrix

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 55
NUM_HIDDEN = 5

THRESHOLD = 0.6

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=24-step=26300.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_0/hparams.yaml'

RBM_WEIGHTS_PATH = 'rbm.npz'


def main():
    try:
        seg_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.IMG_SEG
        )

        segmentation_dataloader = DataLoader(
            seg_dataset, 
            batch_size=1, 
            shuffle=False
        )

        lbae = LBAE.load_from_checkpoint(
            checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
            hparams_file=AUTOENCODE_HPARAMS_PATH,
            map_location=torch.device('cpu')
        )

        rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
        rbm.load(file=RBM_WEIGHTS_PATH)

        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "\t1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "\t2. Run the training pipeline before starting the evaluation.\n"
              "The application will terminate now.")
        return

    lbae.eval()

    y_true = []
    img_seg = []
    pixels = []

    with torch.no_grad():
        for X, y in segmentation_dataloader:

            encoder_output, _ = lbae.encoder.forward(X)
            rbm_input = encoder_output.detach().numpy()
            rbm_labels = rbm.binarized_rbm_output(rbm_input, threshold=THRESHOLD)

            y_true.append(y)
            img_seg.append(rbm_labels)
            pixels.append(rbm_input)

    img_seg = np.concatenate(img_seg)
    y_true = np.concatenate(y_true)
    pixels = np.concatenate(pixels)

    dist_matrix = distance_matrix(img_seg)

    ahc = AgglomerativeClustering(n_clusters=17, metric='precomputed', linkage='average')
    ahc.fit(dist_matrix)

    ahc_segmentation = ahc.labels_

    homogeneity = homogeneity_score(y_true, ahc_segmentation)
    completeness = completeness_score(y_true, ahc_segmentation)

    print(f'\nHomogenity score: {round(homogeneity, 3)}')
    print(f'Completeness score: {round(completeness, 3)}\n')

if __name__ == '__main__':
    main()