import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr,
                 weights=list(), A=None, device='cpu', ckpt_file='misa.pt', test=False):
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
        training_loss, training_MISI = model.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        else:
            final_MISI = []
    else:
        model.predict(data_loader)
    
    torch.save({'model': model.state_dict(),
                'seed': model.seed,
                'index': model.index,
                'subspace': model.subspace,
                'eta': model.eta,
                'beta': model.beta,
                'lam': model.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI}, 
               ckpt_file)
    print("Saved checkpoint to: " + ckpt_file)
    
    return model.output, final_MISI