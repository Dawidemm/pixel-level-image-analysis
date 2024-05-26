from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage
import torch
import numpy as np

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

from src.utils.utils import ThresholdFinder

torch.set_float32_matmul_precision('medium')

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 55
NUM_HIDDEN = 17

MAX_EPOCHS = 25
RBM_STEPS = 1000
RBM_LEARNING_RATE = 0.001

BATCH_SIZE = 16

THRESHOLDS = np.linspace(1/10, 1, 10)

HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=24-step=26300.ckpt'
AUTOENCODE_HPARAMS_PATH = 'lightning_logs/version_0/hparams.yaml'

RBM_WEIGHTS_PATH = 'rbm.npz'

CLASSES_TO_REMOVE = False

def main():

    try:
        train_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.TRAIN,
            class_filter=CLASSES_TO_REMOVE
        )

        test_dataset = HyperspectralDataset(
            hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
            ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
            stage=Stage.TEST,
            class_filter=CLASSES_TO_REMOVE
        )

    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to provide paths to the hyperspectral image and ground truth image files.\n"
              "The application will terminate now.")
        return

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True
    )

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODE_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )
    
    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    pipeline = Pipeline(auto_encoder=lbae, rbm=rbm)

    pipeline.fit(
        train_dataloader, 
        max_epochs=MAX_EPOCHS, 
        rbm_steps=RBM_STEPS,
        rbm_learning_rate=RBM_LEARNING_RATE,
        rbm_trainer='cd1', 
        learnig_curve=False,
        skip_autoencoder=True
    )

    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
    rbm.load(file=RBM_WEIGHTS_PATH)

    threshold_finder = ThresholdFinder(
        test_dataloader=test_dataloader,
        encoder=lbae.encoder,
        rbm=rbm
    )

    threshold, rand_score, adjusted_rand_score = threshold_finder.find_threshold(THRESHOLDS)

    with open('rbm_checks.txt', 'a+') as file:
        file.write(f'Removed classes: {CLASSES_TO_REMOVE}\n')
        file.write(f'Threshold: {round(threshold, 2)}; Rand Score: {round(rand_score, 2)}, Adjusted Rand Score: {round(adjusted_rand_score, 2)}\n')
        file.write(f'-------------------------------------------------------------\n')

if __name__ == '__main__':
    main()