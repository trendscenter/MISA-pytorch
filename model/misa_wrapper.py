import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr,
                 weights=list(), device='cpu', ckpt_file='misa.pt', test=False):
    model=MISA(weights=weights,
                 index=index, 
                 subspace=subspace, 
                 eta=eta, 
                 beta=beta, 
                 lam=lam, 
                 input_dim=input_dim, 
                 output_dim=output_dim,
                 seed=seed,
                 device=device)
    model.to(device=device)
    if not test:
        model.train_me(data_loader, epochs, lr)
    else:
        model.predict(data_loader)
    torch.save({'model': model.state_dict(),
                'seed': model.seed.detach().cpu(),
                'index': model.index.detach().cpu(),
                'subspace': model.subspace.detach().cpu(),
                'eta': model.eta.detach().cpu(),
                'beta': model.beta.detach().cpu(),
                'lam': model.lam.detach().cpu()},
               ckpt_file)
    print("Saved to: " + ckpt_file)
    return model.output.detach().cpu()