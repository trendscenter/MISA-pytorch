import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F
from model.MISAK import MISA

def MISA_wrapper_(data_loader, epochs, lr, A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None, model_MISA=None):
    
    model_MISA.to(device=device)
    
    final_MISI = []
    
    if not test:
        training_loss, training_MISI, optimizer = model_MISA.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model_MISA.predict(test_data_loader)
        print(f"MISA test loss: {test_loss[0].detach().cpu().numpy():.3f}")

        torch.save({'model_MISA': model_MISA.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': model_MISA.seed,
                'index': model_MISA.index,
                'subspace': model_MISA.subspace,
                'eta': model_MISA.eta,
                'beta': model_MISA.beta,
                'lam': model_MISA.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI,
                'test_loss': test_loss},
               ckpt_file)
        # print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model_MISA.load_state_dict(checkpoint['model_MISA'])
        test_loss = model_MISA.predict(test_data_loader)
        print(f"MISA test loss: {test_loss[0].detach().cpu().numpy():.3f}")
    
    return model_MISA, final_MISI


def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr,
                 weights=list(), A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None, model=None):
    
    model_MISA=MISA(weights=weights,
                 index=index, 
                 subspace=subspace, 
                 eta=eta, 
                 beta=beta, 
                 lam=lam, 
                 input_dim=input_dim, 
                 output_dim=output_dim,
                 seed=seed,
                 device=device,
                 model=model)
    
    model_MISA.to(device=device)
    
    final_MISI = []
    
    if not test:
        training_loss, training_MISI, optimizer = model_MISA.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model_MISA.predict(test_data_loader)
        print(f"test loss: {test_loss[0].detach().cpu().numpy():.3f}")

        torch.save({'model_MISA': model_MISA.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': model_MISA.seed,
                'index': model_MISA.index,
                'subspace': model_MISA.subspace,
                'eta': model_MISA.eta,
                'beta': model_MISA.beta,
                'lam': model_MISA.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI,
                'test_loss': test_loss},
               ckpt_file)
        print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model_MISA.load_state_dict(checkpoint['model_MISA'])
        test_loss = model_MISA.predict(test_data_loader)
        print(f"test loss: {test_loss[0].detach().cpu().numpy():.3f}")
    
    return model_MISA, final_MISI