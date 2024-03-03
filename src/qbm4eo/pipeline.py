from itertools import islice
from pathlib import Path

import numpy as np
import torch
import lightning as pl
# from pylab import subplots
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader

# from torchvision import transforms

# from qbm4eo.cim import CIMSampler, ramp
from src.qbm4eo.classifier import Classifier
from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import CD1Trainer, RBM


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
        skip_classifier=False
    ):
        # Adjust flags for skipping training components. If given component
        # is None, we train it anyway, otherwise whole process does not make sense.
        skip_autoencoder = skip_autoencoder or self.auto_encoder is None
        skip_rbm = skip_rbm or self.rbm is None
        skip_classifier = skip_classifier or self.classifier is None

        trainer = pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "mps",
            precision=precision,
            max_epochs=max_epochs,
            enable_checkpointing=enable_checkpointing
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
            rbm_trainer = CD1Trainer(rbm_steps, learning_rate=rbm_learning_rate)
            rbm_trainer.fit(self.rbm, encoded_dataloader(data_loader, encoder))

        self.rbm.save("rbm.npz")

    #     if skip_classifier:
    #         print("Skipping classifier training.")
    #     else:
    #         trainer.fit(self.classifier, data_loader)
    #         self.classifier.freeze()
    #         torch.save(self.classifier.state_dict(), "classifier.pt")
    #         trainer.save_checkpoint("classifier.ckpt")

    # def predict(self, data_loader):
    #     return self.classifier(
    #         torch.from_numpy(
    #         self.rbm.h_probs_given_v(
    #             self.auto_encoder.encoder(data_loader)[0].detach().numpy()
    #         )
    #         ).float())

    # @classmethod
    # def load(cls, path):
    #     path = Path(path)
    #     auto_encoder = LBAE.load_from_checkpoint(str(path / "lbae.ckpt"))
    #     rbm = RBM.load(path / "rbm.npz")
    #     classifier = Classifier.load_from_checkpoint(
    #         str(path / "classifier.ckpt"),
    #         encoder=auto_encoder.encoder,
    #         rbm=rbm
    #     )
    #     return cls(auto_encoder, rbm, classifier)