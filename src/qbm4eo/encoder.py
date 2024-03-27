import torch
from torch.nn.functional import leaky_relu
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

        subnet = [nn.LeakyReLU(negative_slope),
                  nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
                  nn.BatchNorm1d(channels)
        ]
        
        self.subnet = nn.Sequential(*subnet)

    def forward(self, x):
        return self.subnet(x)
    

class ResBlockConv(nn.Module):
    def __init__(self, channels, in_channels=None, negative_slope=0.02, bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if in_channels is None:
            in_channels = channels

        initial_block = [nn.Conv1d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
                         nn.BatchNorm1d(channels)
        ]
        middle_block = [ResBlockConvPart(channels, negative_slope, bias),
                        ResBlockConvPart(channels, negative_slope, bias)
        ]
        
        self.initial_block = nn.Sequential(*initial_block)
        self.middle_block = nn.Sequential(*middle_block)

        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.initial_block(x)
        y = x
        x = self.middle_block(x)
        
        return leaky_relu(x + y, self.negative_slope)


class LBAEEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        out_channels,
        latent_size,
        num_layers,
        should_quantize,
        negative_slope=0.02,
        bias=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        layers = [
            nn.Conv1d(
                input_size[0],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
        ]
        for i in range(num_layers):
            if i == 0:
                new_layer = ResBlockConv(out_channels)
            else:
                new_layer = ResBlockConv(2**i * out_channels, 2 ** (i - 1) * out_channels)
            layers.append(new_layer)

        layers.append(
            nn.Conv1d(
                2 ** (num_layers - 1) * out_channels,
                2**num_layers * out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            )
        )
        self.net = nn.Sequential(*layers)

        final_channels = 2**num_layers * out_channels
        self.final_conv_size = (final_channels, input_size[1])
        lin_in_size = final_channels * input_size[1]
        self.linear = nn.Linear(lin_in_size, latent_size)

        self.quantize = should_quantize
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