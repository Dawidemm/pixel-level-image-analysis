from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage
import torch
import numpy as np
import os

from src.utils import utils

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

torch.set_float32_matmul_precision('medium')

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 32

NUM_HIDDEN = [8, 16, 32, 64, 128]
RBM_STEPS = [1000, 10000, 100000]
RBM_LEARNING_RATE = [0.01, 0.001, 0.0001]

BATCH_SIZE = 16

THRESHOLDS = np.linspace(2/10, 1, 9)[:-2]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

def main():

    try:
        train_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            remove_noisy_bands=False,
            num_images_to_load=1,
            stage=Stage.TRAIN
        )

        find_threshold_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            remove_noisy_bands=False,
            num_images_to_load=1
        )

    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        print("Please make sure to provide paths to the hyperspectral data and ground truth data folders.\n"
              "The application will terminate now.")
        return

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    find_threshold_dataloader = DataLoader(
        dataset=find_threshold_dataset,
        batch_size=1,
        num_workers=0,
    )

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH,
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )

    os.makedirs('./experiments/', exist_ok=True)
    with open('experiments/experiments_raport.csv', 'a+') as file:
        file.write(f'experiment,num_hidden,rbm_steps,rbm_lr,threshold,ari,rand_score,homogenity,completeness\n')
    
    experiment = 0
    
    for num_hidden in NUM_HIDDEN:
        for rbm_steps in RBM_STEPS:
            for rbm_lr in RBM_LEARNING_RATE:
    
                rbm = RBM(NUM_VISIBLE, num_hidden)

                pipeline = Pipeline(auto_encoder=lbae, rbm=rbm)

                pipeline.fit(
                    train_dataloader,
                    skip_autoencoder=True,
                    rbm_steps=rbm_steps,
                    rbm_learning_rate=rbm_lr, 
                    rbm_trainer='cd1', 
                    learnig_curve=True,
                    experiment_number=experiment
                )

                rbm = RBM(NUM_VISIBLE, num_hidden)
                rbm.load(file=f'./experiments/exp_{experiment}/rbm.npz')

                threshold_finder = utils.ThresholdFinder(
                    dataloader=find_threshold_dataloader,
                    encoder=lbae.encoder,
                    rbm=rbm
                )

                threshold, ari, rand_score, homogenity, completeness  = threshold_finder.find_threshold(THRESHOLDS)

                with open('experiments/experiments_raport.csv', 'a+') as file:
                    file.write(f'{experiment},{num_hidden},{rbm_steps},{rbm_lr},{threshold},{ari},{rand_score},{homogenity},{completeness}\n')

                experiment += 1

if __name__ == '__main__':
    main()