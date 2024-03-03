import torch
import torch.nn.functional as F
from torch import nn


class QuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dropout=0):
        x = torch.sign(input)
        x[x == 0] = 1
        return x

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class ResBlockConvPart(nn.Module):
    def __init__(self, channels, negative_slope=0.02, bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subnet = nn.Sequential(
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.subnet(x)


class ResBlockConv(nn.Module):
    def __init__(
        self, channels, in_channels=None, negative_slope=0.02, bias=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if in_channels is None:
            in_channels = channels

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(channels),
        )

        self.middle_block = nn.Sequential(
            ResBlockConvPart(channels, negative_slope, bias),
            ResBlockConvPart(channels, negative_slope, bias),
        )
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.initial_block(x)
        y = x
        x = self.middle_block(x)
        return F.leaky_relu(x + y, self.negative_slope)


class LBAEEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        out_channels,
        zsize,
        num_layers,
        quantize,
        negative_slope=0.02,
        bias=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        layers = [
            nn.Conv2d(
                input_size[0],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope),
        ]
        for i in range(num_layers):
            if i == 0:
                new_layer = ResBlockConv(out_channels)
            else:
                new_layer = ResBlockConv(2**i * out_channels, 2 ** (i - 1) * out_channels)
            layers.append(new_layer)

        layers.append(
            nn.Conv2d(
                2 ** (num_layers - 1) * out_channels,
                2**num_layers * out_channels,
                kernel_size=4,
                stride=2,
                bias=bias,
                padding=1,
            )
        )
        self.net = nn.Sequential(*layers)

        final_res = (
            input_size[1] // 2 ** (num_layers + 1),
            input_size[2] // 2 ** (num_layers + 1),
        )
        final_channels = 2**num_layers * out_channels
        self.final_conv_size = (final_channels, *final_res)
        lin_in_size = final_channels * final_res[0] * final_res[1]
        self.linear = nn.Linear(lin_in_size, zsize)

        self.quantize = quantize
        self.quant = QuantizerFunc.apply

    def forward(self, x, epoch=None):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.tanh(x)
        if epoch in self.quantize:
            xq = self.quant(x)
        else:
            xq = x
        err_quant = torch.abs(x - xq)
        x = xq
        return x, err_quant.sum() / (x.size(0) * x.size(1))