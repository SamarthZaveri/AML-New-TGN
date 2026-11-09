"""Learned encoding of nonnegative elapsed HOURS, compressed with log1p."""
import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        if time_dim < 1:
            raise ValueError('time_dim must be positive')
        self.w_linear = nn.Parameter(torch.ones(1))
        self.b_linear = nn.Parameter(torch.zeros(1))
        self.w_periodic = nn.Parameter(3 * torch.pow(10., -torch.linspace(0, 1, time_dim - 1)))
        self.b_periodic = nn.Parameter(torch.zeros(time_dim - 1))

    def forward(self, delta_t):
        x = torch.log1p(delta_t.clamp(min=0)).to(self.w_linear.dtype).reshape(-1, 1)
        return torch.cat((x * self.w_linear + self.b_linear,
                          torch.sin(x * self.w_periodic + self.b_periodic)), dim=-1)
