import numpy as np
import lightning as pl
import torch.nn.functional
from torch import nn, optim


class Classifier(pl.LightningModule):

    def __init__(self, output_size, encoder, rbm):
        super().__init__()
        self.save_hyperparameters("output_size")
        self.encoder = encoder
        self.rbm = rbm
        self.linear = nn.Sequential(nn.Linear(rbm.num_hidden, output_size), nn.Sigmoid())

    def training_step(self, batch, _batch_idx):
        x, target = batch
        probs = self.forward(
            torch.from_numpy(
                self.rbm.h_probs_given_v(
                    self.encoder(x)[0].cpu().detach().numpy()
                ).astype(np.float32)
            ).cuda()
        )
        return torch.nn.functional.binary_cross_entropy(probs, target.float())

    def forward(self, data):
        return self.linear(data)

    def configure_optimizers(self):
        return {"optimizer": optim.Adam(self.parameters(), lr=1e-3)}