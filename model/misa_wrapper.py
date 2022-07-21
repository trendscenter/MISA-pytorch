import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, epochs, lr, seed,
                 weights = list(), ckpt_file='misa.pt', test=False):
    model = MISA(weights = weights,
                 index = index, 
                 subspace = subspace, 
                 eta = eta, 
                 beta = beta, 
                 lam = lam, 
                 input_dim = input_dim, 
                 output_dim = output_dim,
                 seed = seed)
    if not test:
        model.training(data_loader, epochs, lr)
    else:
        model.predict(data_loader)