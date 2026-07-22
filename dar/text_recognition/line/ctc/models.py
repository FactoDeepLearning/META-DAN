import torch
from torch.nn.functional import log_softmax
from torch.nn import Conv2d, AdaptiveMaxPool2d, Conv1d
from torch.nn import Module
from dar.generic.models import FCN_Encoder


class CTCDecoder(Module):
    def __init__(self, params):
        super().__init__()

        self.vocab_size = params["vocab_size"]

        self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.end_conv = Conv1d(in_channels=params["enc_dim"], out_channels=self.vocab_size, kernel_size=1)

    def forward(self, x):
        x = self.ada_pool(x).squeeze(2)
        x = self.end_conv(x)
        return log_softmax(x, dim=1)


class FCN(Module):
    def __init__(self, params):
        super(FCN, self).__init__()
        self.vocab_size = params["vocab_size"]

        self.encoder = FCN_Encoder(params)

        self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.end_conv = Conv2d(in_channels=256, out_channels=self.vocab_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.ada_pool(x)
        x = self.end_conv(x)
        x = torch.squeeze(x, dim=2)
        return log_softmax(x, dim=1)
