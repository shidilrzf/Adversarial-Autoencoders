import numpy as np

import models
from torchvision.utils import save_image


import torch

from torch.utils.data import DataLoader, dataset
from torchvision.datasets import MNIST
import torchvision.transforms as T
eps = np.finfo(float).eps

import argparse
import os


desc = "Pytorch implementation of AAE'"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dim_z', type=int, help='Dimensionality of latent variables', default=32)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of ADAM optimizer')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--use_cuda', type=bool, default=False, help='Use GPU?')
parser.add_argument('--log_interval', type=int, default=100)

args = parser.parse_args()

EPS = 1e-15


""" GPU """
# Enable CUDA, set tensor type and device

use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
print(use_cuda)
if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

""" Directory loading """

Model_dir = 'Models/'

if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)

Data_dir = 'Data/'

if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)

log_dir = 'logs/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

Fig_dir = 'Figs/'

if not os.path.exists(Fig_dir):
    os.makedirs(Fig_dir)



""" Data loaders """

train_loader = torch.utils.data.DataLoader(MNIST('Data/', train=True, download=True,
                             transform=T.Compose([
                               T.transforms.ToTensor(),
                               T.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=args.batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(MNIST('Data/', train=False, download=True,
                             transform=T.Compose([
                               T.transforms.ToTensor(),
                               T.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=args.batch_size, shuffle=False)

encoder = models.Encoder(784,args.dim_z).to(device)
decoder = models.Decoder(784,args.dim_z).to(device)

encoder.load_state_dict(torch.load(Model_dir + 'encoder_z'+ str(args.dim_z)+'_epch'+str(args.epochs)+'.pt'))
decoder.load_state_dict(torch.load(Model_dir + 'decoder_z'+ str(args.dim_z)+'_epch'+str(args.epochs)+'.pt'))


def reconstruct(encoder, decoder, device, dtype, loader_val):
    encoder.eval()
    decoder.eval()
    X_val, _= next(iter(loader_val))
    X_val = X_val.view(-1,784)
    X_val = X_val.type(dtype)
    X_val = X_val.to(device)
    z_val = encoder(X_val)
    X_hat_val = decoder(z_val)

    X_val = X_val[:10].cpu().view(10 * 28, 28)
    X_hat_val = X_hat_val[:10].cpu().view(10 * 28, 28)
    comparison = torch.cat((X_val, X_hat_val), 1).view(10 * 28, 2 * 28)
    return comparison

comparison = reconstruct(encoder,decoder, device, dtype, val_loader)
save_image(comparison,
           Fig_dir + 'AAE_comparison_z_{}_epch_{}.png'.format(args.dim_z,args.epochs))