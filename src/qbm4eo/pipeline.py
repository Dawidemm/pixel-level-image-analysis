from itertools import islice
from pathlib import Path

import numpy as np
import torch
import lightning as pl
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping

from src.qbm4eo.rbm import CD1Trainer, AnnealingRBMTrainer
import time

def encoded_dataloader(data_loader, encoder):
    while True:
        for batch_idx, (data, target) in enumerate(data_loader):
            yield encoder(data)[0], target


class Pipeline:

    def __init__(self, auto_encoder, rbm, classifier):
        self.auto_encoder = auto_encoder
        self.rbm = rbm
        self.classifier = classifier

    def fit(
        self,
        data_loader: DataLoader,
        gpus=1,
        precision=32,
        max_epochs=100,
        enable_checkpointing=True,
        rbm_learning_rate=0.01,
        rbm_steps=100,
        skip_autoencoder=False,
        skip_rbm=False,
        skip_classifier=False,
        rbm_trainer=None
    ):
        # Adjust flags for skipping training components. If given component
        # is None, we train it anyway, otherwise whole process does not make sense.
        skip_autoencoder = skip_autoencoder or self.auto_encoder is None
        skip_rbm = skip_rbm or self.rbm is None
        skip_classifier = skip_classifier or self.classifier is None

        early_stopping = EarlyStopping(
            monitor='loss',
            mode='min',
            patience=3
        )

        trainer = pl.Trainer(
            accelerator='cpu',
            precision=precision,
            max_epochs=max_epochs,
            logger=True,
            enable_checkpointing=enable_checkpointing,
            callbacks=[early_stopping]
        )

        if skip_autoencoder:
            print("Skipping autoencoder training as requested.")
        else:
            print("Training autoencoder.")
            trainer.fit(self.auto_encoder, data_loader)

        encoder = self.auto_encoder.encoder
        for param in encoder.parameters():
            param.requires_grad = False

        torch.save(encoder.state_dict(), "encoder.pt")

        if skip_rbm:
            print("Skipping RBM training as requested.")
        else:
            if rbm_trainer == 'cd1':
                print('RBM training.')
                rbm_trainer = CD1Trainer(rbm_steps, learning_rate=rbm_learning_rate)
                rbm_trainer.fit(self.rbm, encoded_dataloader(data_loader, encoder))
            elif rbm_trainer == 'annealing':
                rbm_trainer = AnnealingRBMTrainer(rbm_steps, sampler='placeholder', learning_rate=rbm_learning_rate)
                rbm_trainer.fit(self.rbm, encoded_dataloader(data_loader, encoder))
            else:
                raise ValueError(f'Argument "rbm_trainer" should be set as one from ["cd1", "annealing"] values.')
            
        self.rbm.save("rbm.npz")