import torch
from torch import nn

from .ema import EMA

class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    DEMA(x) = 2 * EMA(x) - EMA(EMA(x))
    """
    def __init__(self, alpha):
        super(DEMA, self).__init__()
        self.alpha = alpha
        self.ema1 = EMA(alpha)
        self.ema2 = EMA(alpha)

    def forward(self, x):
        # x: [Batch, Time, Channel]
        ema_x = self.ema1(x)
        ema_ema_x = self.ema2(ema_x)
        dema = 2 * ema_x - ema_ema_x
        return dema