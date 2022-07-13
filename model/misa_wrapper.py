import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

import .model.MISAK MISA

def MISA_wrapper(latent_dim, n_layers, lr, seed,
                 ckpt_file='misa.pt', test=False):
    model = MISA(weights = list(), index = index, subspace = subspace, beta = beta, eta = eta, lam = lam, input_dim = input_dim, output_dim = output_dim)
