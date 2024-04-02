import numpy as np
import lightning as pl
import torch.nn.functional
from torch import nn, optim
from torchmetrics import Accuracy

class Classifier(pl.LightningModule):

    def __init__(self, output_size, encoder, rbm):
        super().__init__()
        self.save_hyperparameters("output_size")
        self.encoder = encoder
        self.rbm = rbm

        linear = [
            nn.Linear(rbm.num_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Softmax(dim=1)
        ]
        self.linear = nn.Sequential(*linear)

        self.accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        return self.linear(data)

    def training_step(self, batch, _batch_idx):
        x, target = batch
        probs = self.forward(
            torch.from_numpy(
                self.rbm.binarize_rbm_output(
                    self.encoder(x)[0].cpu().detach().numpy(),
                    threshold=0.5
                ).astype(np.float32)
            )
        )

        loss = self.loss(probs, target.long())        
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        step_acc = self.accuracy(probs.argmax(dim=1), target)
        self.log("acc", step_acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)