from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

from src.utils import utils

torch.set_float32_matmul_precision('medium')

torch.manual_seed(10)

NUM_VISIBLE = 28
THRESHOLDS = np.linspace(1/10, 1, 10)[:-1]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
IMAGES = ['D_1', 'E_1', 'F_1']

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_3/checkpoints/epoch=19-step=40000.ckpt'
AUTOENCODER_HPARAMS_PATH = 'lightning_logs/version_3/hparams.yaml'

EXPERIMENT_FOLDER_PATH = './experiments_QA/'

def natural_key(file_name):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file_name)]

def main():

    columns = ['num_hidden',
               'random_seed',
               'epoch',
               'threshold',
               'ari',
               'rand_score',
               'homogeneity',
               'completeness',
               'num_unique_labels']
    
    df = pd.DataFrame(columns=columns)

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH,
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )
    # print(f'Number of folders to search: {len(os.listdir(EXPERIMENT_FOLDER_PATH))}')
    # print(os.listdir(EXPERIMENT_FOLDER_PATH))
    for i in tqdm(range(len(os.listdir(EXPERIMENT_FOLDER_PATH))), desc="Processing Experiment Folders"):
        if i == 260:
            break
        exp_path = f'experiments_QA/exp_{i}/'
        model_files = sorted([f for f in os.listdir(exp_path) if f.endswith('.npz') and 'rbm' in f], key=natural_key)
        print(model_files)
        # # model_files = [model_files[-1]]
        # model_files = [model_files[3]]

        for model_path in tqdm(model_files, desc=f"Processing Models in {exp_path}", leave=False):

            rbm_file_to_load = exp_path+model_path

            match = re.search(r"rbm_nh=(\d+)_seed=(\d+)_epoch=(\d+)", rbm_file_to_load)

            if match:
                num_hidden = int(match.group(1))
                seed = int(match.group(2))
                epoch = int(match.group(3))
        
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
                batch_size=256,
                drop_last=False
            )

            rbm = RBM(
                num_visible=NUM_VISIBLE,
                num_hidden=num_hidden,
                random_seed=seed
            )
            
            rbm = rbm.load(file=rbm_file_to_load)

            threshold_finder = utils.ThresholdFinder(
                dataloader=test_dataloader,
                rbm=rbm,
                encoder=lbae.encoder
            )

            threshold, ars, rand_score, homogeneity, completeness, num_unique_labels = threshold_finder.find_threshold(THRESHOLDS)

            data = [num_hidden,
                    seed,
                    epoch,
                    threshold, 
                    ars, 
                    rand_score, 
                    homogeneity, 
                    completeness, 
                    num_unique_labels]
            df.loc[len(df)] = data
    
    print('Obtained metrics saved to: ', EXPERIMENT_FOLDER_PATH+'experiments_raport.csv')
    df.to_csv(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', index=False)

if __name__ == '__main__':
    main()