from torch.utils.data import DataLoader
from src.utils.blood_dataset import BloodIterableDataset, Stage
import torch

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM


torch.set_float32_matmul_precision('medium')

torch.manual_seed(10)

NUM_VISIBLE = 28

BATCH_SIZE = 8
NUM_HIDDEN = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
RANDOM_SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
IMAGES = ['D_1', 'E_1', 'F_1']

AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_3/checkpoints/epoch=19-step=40000.ckpt'
AUTOENCODER_HPARAMS_PATH = 'lightning_logs/version_3/hparams.yaml'

EXPERIMENT_FOLDER_PATH = './experiments/'

def main():

    experiment = 0

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

            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=8,
                drop_last=True
            )

            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=8,
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
                rbm_trainer='cd1',
                rbm_epochs=1,
                learnig_curve=True,
                experiment_folder_path=EXPERIMENT_FOLDER_PATH,
                experiment_number=experiment
            )

            experiment += 1


if __name__ == '__main__':
    main()