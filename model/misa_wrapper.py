import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import datetime

from model.MISAK import MISA

def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr, beta1, beta2, batch_size, patience, fused, foreach,
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
        run_time_start = datetime.datetime.now().timestamp()
        training_loss, training_MISI, optimizer, epochs_completed = model.train_me(data_loader, epochs, lr, A, beta1, beta2, batch_size, weights, seed, patience, fused, foreach)
        run_time_stop = datetime.datetime.now().timestamp()
        run_time = run_time_stop - run_time_start
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model.predict(test_data_loader)
        print(f"test loss: {test_loss[0].detach().cpu().numpy():.3f}")

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
                'test_loss': test_loss,
                'epochs_completed': epochs_completed},
               ckpt_file)
        print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model.load_state_dict(checkpoint['model'])
        test_loss = model.predict(test_data_loader)
        print(f"test loss: {test_loss[0].numpy():.3f}")
    
    return model.output, final_MISI, run_time, epochs_completed