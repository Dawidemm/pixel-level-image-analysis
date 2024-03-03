from typing import Any
import torch
import lightning as pl
import torch.nn.functional as F
from torch import optim

from decoder import LBAEDecoder
from encoder import LBAEEncoder


def loss(xr, x):
    return F.mse_loss(xr, x, reduction="sum")


class LBAE(pl.LightningModule):
    def __init__(
        self, input_size, out_channels, zsize, num_layers, quantize, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("input_size", "out_channels", "zsize", "num_layers", "quantize")

        self.encoder = LBAEEncoder(input_size, out_channels, zsize, num_layers, quantize)
        self.decoder = LBAEDecoder(self.encoder.final_conv_size, input_size, zsize, num_layers)
        self.epoch = 0

    def forward(self, x):
        z, quant_err = self.encoder(x, self.epoch)
        xr = self.decoder(z)
        self.log("quant_error", quant_err)
        return xr

    def training_step(self, batch, batch_idx):
        x, _ = batch
        if self.epoch == 0 and batch_idx == 0:
            self.reference_image = x[0:1, :, :, :]
        xr = self.forward(x)
        l = loss(xr.view(x.size()), x)
        self.log("loss", l, logger=True)
        return l

    def predict_step(self, batch, batch_idx):
        x, labels = batch
        return self.forward(x), x, labels

    def test_step(self, batch, batch_idx):
        x, _ = batch
        xr = self.forward(x)
        return loss(xr.view(x.size()), x)

    def configure_optimizers(self):
        return {"optimizer": optim.Adam(self.parameters(), lr=1e-3)}

    def on_training_epoch_end(self, outputs):
        with torch.no_grad():
            xr = self.forward(self.reference_image)
        if self.reference_image.size(1) > 1:
            dataformats = "CHW"
        else:
            dataformats = "HW"
        self.logger.experiment.add_image("input", self.reference_image.squeeze(), self.epoch, dataformats=dataformats)
        self.logger.experiment.add_image("recovery", xr.squeeze(), self.epoch, dataformats=dataformats)
        self.logger.experiment.flush()
        self.epoch += 1