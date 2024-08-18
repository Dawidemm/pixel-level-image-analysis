import os
import torch
import lightning as pl
from torch.utils.data import DataLoader

from src.qbm4eo.rbm import CD1Trainer, AnnealingRBMTrainer
from src.utils import utils

from typing import Union

def encoded_dataloader(data_loader, encoder):
    while True:
        for batch_idx, (data, target) in enumerate(data_loader):
            yield encoder(data)[0], target


class Pipeline:

    def __init__(self, auto_encoder, rbm):
        self.auto_encoder = auto_encoder
        self.rbm = rbm

    def fit(
        self,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        gpus=1,
        precision=32,
        max_epochs=100,
        enable_checkpointing=True,
        rbm_learning_rate=0.0001,
        rbm_steps=100,
        skip_autoencoder=False,
        skip_rbm=False,
        rbm_trainer=None,
        learnig_curve=True,
        experiment_folder_path: Union[str, None]=None,
        experiment_number: Union[int, None]=None
    ):
        # Adjust flags for skipping training components. If given component
        # is None, we train it anyway, otherwise whole process does not make sense.
        skip_autoencoder = skip_autoencoder or self.auto_encoder is None
        skip_rbm = skip_rbm or self.rbm is None

        loss_logs = utils.LossLoggerCallback()

        trainer = pl.Trainer(
            accelerator='cpu',
            precision=precision,
            max_epochs=max_epochs,
            logger=True,
            enable_checkpointing=enable_checkpointing,
            callbacks=[loss_logs]
        )

        if skip_autoencoder:
            print("Skipping autoencoder training as requested.")
        else:
            print("Training autoencoder.")
            trainer.fit(
                model=self.auto_encoder,
                train_dataloaders=train_data_loader,
                val_dataloaders=validation_data_loader
            )

            if learnig_curve:
                utils.plot_loss(
                    epochs=trainer.max_epochs, 
                    train_loss_values=loss_logs.train_losses,
                    validation_loss_values=loss_logs.validation_losses,
                    plot_title='Autoencoder'
                )

        encoder = self.auto_encoder.encoder
        for param in encoder.parameters():
            param.requires_grad = False

        if skip_rbm:
            print("Skipping RBM training as requested.")
        else:
            if rbm_trainer == 'cd1':
                print('RBM training with CD1Trainer.')
                rbm_trainer = CD1Trainer(rbm_steps, learning_rate=rbm_learning_rate)
                rbm_trainer.fit(self.rbm, encoded_dataloader(train_data_loader, encoder))

                if learnig_curve:
                    utils.plot_loss(
                        epochs=rbm_trainer.num_steps, 
                        loss_values=rbm_trainer.losses, 
                        plot_title='RBM',
                        experiment_number=experiment_number
                    )

            elif rbm_trainer == 'annealing':
                print('RBM training with AnnealingRBMTrainer.')
                rbm_trainer = AnnealingRBMTrainer(rbm_steps, sampler='placeholder', learning_rate=rbm_learning_rate)
                rbm_trainer.fit(self.rbm, encoded_dataloader(train_data_loader, encoder))
            else:
                raise ValueError(f'Argument "rbm_trainer" should be set as one from ["cd1", "annealing"] values.')
            
        if experiment_number != None:
            experiment_path = f'{experiment_folder_path}/exp_{experiment_number}/'
            os.makedirs(experiment_path, exist_ok=True)
            self.rbm.save(os.path.join(experiment_path, 'rbm.npz'))
        else:
            self.rbm.save(f'rbm.npz')