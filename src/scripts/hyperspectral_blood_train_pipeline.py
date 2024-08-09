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

NUM_HIDDEN = [3, 8, 16]
RBM_STEPS = [1000, 10000]
RBM_LEARNING_RATE = [0.001, 0.0001]

IMAGES = ['D_1', 'E_1', 'F_1']

BATCH_SIZE = 16

THRESHOLDS = np.linspace(1/10, 1, 10)[:-1]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=14-step=4410900.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

EXPERIMENT_FOLDER_PATH = './experiments/diff_imgs/'

def main():

    os.makedirs(EXPERIMENT_FOLDER_PATH, exist_ok=True)
    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
        file.write(f'image,experiment,num_hidden,rbm_steps,rbm_lr,threshold,ari,rand_score,homogenity,completeness\n')

    experiment = 0

    for image in IMAGES:

        train_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            load_specific_image=image,
            remove_noisy_bands=False,
            stage=Stage.TRAIN
        )

        find_threshold_dataset = BloodIterableDataset(
            hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
            ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
            load_specific_image=image,
            remove_noisy_bands=False,
        )

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
                        experiment_folder_path=EXPERIMENT_FOLDER_PATH,
                        experiment_number=experiment
                    )

                    rbm = RBM(NUM_VISIBLE, num_hidden)
                    rbm.load(file=f'{EXPERIMENT_FOLDER_PATH}exp_{experiment}/rbm.npz')

                    threshold_finder = utils.ThresholdFinder(
                        dataloader=find_threshold_dataloader,
                        encoder=lbae.encoder,
                        rbm=rbm
                    )

                    threshold, ari, rand_score, homogenity, completeness, _  = threshold_finder.find_threshold(THRESHOLDS)

                    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
                        file.write(f'{image},{experiment},{num_hidden},{rbm_steps},{rbm_lr},{threshold},{ari},{rand_score},{homogenity},{completeness}\n')

                    experiment += 1

if __name__ == '__main__':
    main()