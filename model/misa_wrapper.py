import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

# import .model.MISAK MISA

def MISA_wrapper(latent_dim, n_layers, lr, seed,
                 ckpt_file='misa.pt', test=False):
    pass