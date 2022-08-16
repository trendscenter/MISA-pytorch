import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr,
                 weights=list(), A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None):
    
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
    
    final_MISI = []
    
    if not test:
        training_loss, training_MISI, optimizer = model.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model.predict(test_data_loader)

        torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': model.seed,
                'index': model.index,
                'subspace': model.subspace,
                'eta': model.eta,
                'beta': model.beta,
                'lam': model.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI,
                'test_loss': test_loss},
               ckpt_file)
        print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model.load_state_dict(checkpoint['model'])
        test_loss = model.predict(test_data_loader)
    
    return model.output, final_MISI