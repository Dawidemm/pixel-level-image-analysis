from math import prod

import torch
import torch.nn.functional as F
from torch import nn


class ResBlockDeConvPart(nn.Module):
    def __init__(self, channels, negative_slope=0.02, bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subnet = nn.Sequential(
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.subnet(x)


class ResBlockDeConv(nn.Module):
    def __init__(
        self, channels, in_channels=None, negative_slope=0.02, bias=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if in_channels is None:
            in_channels = channels

        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(channels),
        )

        self.middle_block = nn.Sequential(
            ResBlockDeConvPart(channels, negative_slope, bias),
            ResBlockDeConvPart(channels, negative_slope, bias),
        )

        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.initial_block(x)
        y = x
        x = self.middle_block(x)
        return F.leaky_relu(x + y, self.negative_slope)


class LBAEDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        zsize,
        num_layers,
        negative_slope=0.02,
        bias=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.linear = nn.Linear(zsize, prod(input_size))

        layers = []
        for i in range(num_layers):
            layers.append(
                ResBlockDeConv(input_size[0] // (2 ** (i + 1)), input_size[0] // (2**i))
            )

        layers += [
            # Again, swapped input/output number of channels
            nn.ConvTranspose2d(
                input_size[0] // 2**num_layers,
                input_size[0] // 2**num_layers,
                kernel_size=4,
                stride=2,
                bias=bias,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(input_size[0] // 2**num_layers),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(
                input_size[0] // 2**num_layers,
                output_size[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(x.size(0), *self.input_size)

        x = self.net(x)
        x = torch.sigmoid(x)
        return x
