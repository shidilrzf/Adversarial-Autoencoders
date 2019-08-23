import torch
from torch import nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self, dim_input , dim_z):
        super(Encoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_input, self.dim_input// 2),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_input // 2),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_z),
            nn.ReLU(),
        ])
        self.network = nn.Sequential(*self.network)
    def forward(self, x):
        z = self.network(x)
        return z

class Decoder(nn.Module):
    def __init__(self, dim_input , dim_z):
        super(Decoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, self.dim_input// 2),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_input // 2),
            nn.ReLU(),
            nn.Linear(self.dim_input // 2, self.dim_input),
            nn.ReLU(),
        ])
        self.network = nn.Sequential(*self.network)
    def forward(self, z):
        x_recon = self.network(z)
        return x_recon

class Discriminator(nn.Module):
    def __init__(self, dim_z , dim_h):
        super(Discriminator,self).__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h,1),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        disc = self.network(z)
        return disc
