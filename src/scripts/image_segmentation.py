import torch
import lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM
from src.qbm4eo.classifier import Classifier

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 60
NUM_HIDDEN = 30

THRESHOLD = 0.5

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=49-step=26300.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_0/hparams.yaml'

RBM_WEIGHTS_PATH = 'rbm.npz'
CLASSIFIER_PATH = 'classifier.pt'

def main():
    try:
        seg_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.IMG_SEG
        )

        seg_dataloader = DataLoader(seg_dataset, batch_size=1, shuffle=False)

        lbae = LBAE.load_from_checkpoint(
            checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
            hparams_file=AUTOENCODE_HPARAMS_PATH,
            map_location=torch.device('cpu')
        )

        rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
        rbm.load(file=RBM_WEIGHTS_PATH)

        classifier = Classifier(output_size=17, encoder=lbae.encoder, rbm=rbm)
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH))
        
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to:\n"
              "1. Provide paths to the hyperspectral image and ground truth image files.\n"
              "2. Run the training pipeline before starting the evaluation. \n"
              "   Training will generate a 'lightning_logs' folder containing model checkpoint and hparams files.\n"
              "3. The application will terminate now.")
        return

    lbae.eval()

    y_true = []
    img_seg = []

    with torch.no_grad():
        for _, (X, y) in enumerate(seg_dataloader):

            encoder_output, _ = lbae.encoder.forward(X)

            rbm_input = encoder_output.detach().numpy()

            labels = rbm.binarize_rbm_output(rbm_input, threshold=THRESHOLD)
            labels = torch.tensor(labels, dtype=torch.float32)

            y_true.append(y)
            img_seg.append(labels)

    y_true = torch.cat(y_true)
    img_seg = torch.cat(img_seg)

    diff = torch.abs(y_true - img_seg)
    percent_diff = torch.sum(diff) / torch.numel(y_true) * 100
    print(f"Procentowa różnica między obrazkami: {percent_diff:.2f}%")

    y_true = y_true.reshape(145, 145, 1)
    img_seg = img_seg.reshape(145, 145, 1)

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(y_true)
    axes[0].set_title('Ground Truth')
    axes[0].set_axis_off()

    axes[1].imshow(img_seg)
    axes[1].set_title('Segmentation')
    axes[1].set_axis_off()

    plt.show()

if __name__ == '__main__':
    main()