from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage
from src.utils import utils
import torch
import numpy as np
import os

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM


torch.set_float32_matmul_precision('medium')

torch.manual_seed(10)

NUM_VISIBLE = 28

BATCH_SIZE = [8]
NUM_HIDDEN = [26]
RBM_LEARNING_RATE = [0.001]
RANDOM_SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

THRESHOLDS = np.linspace(1/10, 1, 10)[:-1]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
IMAGES = ['D_1', 'E_1', 'F_1']

AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=19-step=290280.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'

EXPERIMENT_FOLDER_PATH = './experiments/'

def main():

    os.makedirs(EXPERIMENT_FOLDER_PATH, exist_ok=True)
    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
        file.write(f'experiment,batch_size,learning_rate,num_hidden,random_state,threshold,ari,rand_score,homogeneity,completeness\n')

    experiment = 0

    for learning_rate in RBM_LEARNING_RATE:
        for batch_size in BATCH_SIZE:
            for num_hidden in NUM_HIDDEN:
                for random_seed in RANDOM_SEEDS:

                    train_dataset = BloodIterableDataset(
                        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                        load_specific_images=IMAGES,
                        stage=Stage.TRAIN,
                        remove_noisy_bands=True,
                        remove_background=True
                    )

                    val_dataset = BloodIterableDataset(
                        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                        load_specific_images=IMAGES,
                        stage=Stage.VAL,
                        remove_noisy_bands=True,
                        remove_background=True
                    )

                    test_dataset = BloodIterableDataset(
                        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                        load_specific_images=IMAGES,
                        stage=Stage.TEST,
                        remove_noisy_bands=True,
                        remove_background=True
                    )

                    train_dataloader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        drop_last=True
                    )

                    val_dataloader = DataLoader(
                        dataset=val_dataset,
                        batch_size=batch_size,
                        drop_last=True
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

                    rbm = RBM(
                        num_visible=NUM_VISIBLE,
                        num_hidden=num_hidden,
                        random_seed=random_seed
                    )

                    pipeline = Pipeline(
                        auto_encoder=lbae, 
                        rbm=rbm
                    )

                    pipeline.fit(
                        train_data_loader=train_dataloader,
                        validation_data_loader=val_dataloader,
                        skip_autoencoder=True,
                        rbm_trainer='annealing',
                        rbm_learning_rate=learning_rate,
                        rbm_epochs=1,
                        learnig_curve=True,
                        experiment_folder_path=EXPERIMENT_FOLDER_PATH,
                        experiment_number=experiment
                    )

                    rbm = RBM(
                        num_visible=NUM_VISIBLE,
                        num_hidden=num_hidden,
                        random_seed=random_seed
                    )

                    rbm.load(file=f'./experiments/exp_{experiment}/rbm.npz')

                    threshold_finder = utils.ThresholdFinder(
                        dataloader=test_dataloader,
                        encoder=lbae.encoder,
                        rbm=rbm
                    )

                    threshold, ari, rand_score, homogeneity, completeness, _ = threshold_finder.find_threshold(THRESHOLDS)

                    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
                        file.write(f'{experiment},{batch_size},{learning_rate},{num_hidden},{random_seed},{threshold},{ari},{rand_score},{homogeneity},{completeness}\n')

                    experiment += 1

if __name__ == '__main__':
    main()