from typing import Any
import torch
import lightning as pl
from torch.nn.functional import mse_loss
from torch import optim

from src.qbm4eo.decoder import LBAEDecoder
from src.qbm4eo.encoder import LBAEEncoder


class LBAE(pl.LightningModule):
    def __init__(
        self, input_size, out_channels, latent_size, num_layers, quantize, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        
        self.save_hyperparameters("input_size", "out_channels", "latent_size", "num_layers", "quantize")

        self.encoder = LBAEEncoder(input_size, out_channels, latent_size, num_layers, quantize)
        self.decoder = LBAEDecoder(self.encoder.final_conv_size, input_size, latent_size, num_layers)
        self.epoch = 0

    def forward(self, x):
        z, quant_err = self.encoder(x, self.epoch)
        x_reconstructed = self.decoder(z)
        self.log("quant_error", quant_err)

        return x_reconstructed

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self.forward(x)
        loss = mse_loss(x_reconstructed.view(x.size()), x, reduction="sum")
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx):
        x, labels = batch

        return self.forward(x), x, labels

    def test_step(self, batch, batch_idx):
        x, _ = batch
        xr = self.forward(x)
        
        return mse_loss(xr.view(x.size()), x, reduction="sum")

    def configure_optimizers(self):
        return {"optimizer": optim.Adam(self.parameters(), lr=1e-3)}