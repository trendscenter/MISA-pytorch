import os
import numpy as np
import mat73
import scipy.io as sio

from model.misa_wrapper import MISA_wrapper
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from scipy.stats import loguniform

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

def zscore(data, axis):
    data -= data.mean(axis=axis, keepdims=True)
    data /= data.std(axis=axis, keepdims=True)
    return np.nan_to_num(data, copy=False)

def correlation(matrix1, matrix2):

    # 1/stdev(M1,axis=1)[None,:] * (M1 @ M2.T) * 1/stdev(M2,axis=1)[:,None] / (d1-1)

    d1 = matrix1.shape[-1]
    d2 = matrix2.shape[-1]

    assert d1 == d2
    assert matrix1.ndim <= 2
    assert matrix2.ndim <= 2
    
    matrix1 = zscore(matrix1.astype(float), matrix1.ndim - 1) / np.sqrt(d1)
    matrix2 = zscore(matrix2.astype(float), matrix2.ndim - 1) / np.sqrt(d2)

    if matrix1.ndim >= matrix2.ndim:
        return matrix1 @ matrix2.T
    else:
        return matrix2 @ matrix1.T

def run_misa(args, config):
    """run MISA"""

    # From args:
    data = args.data
    data_filename = args.filename
    w = args.weights
    test = args.test
    A_exist = args.a_exist
    lr = args.learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    batch_size = args.batch_size
    patience = args.patience 
    if args.ff == 0:
        fused = False
        foreach = False
    elif args.ff == 1:
        fused = False
        foreach = True
    elif args.ff == 2:
        fused = True
        foreach = False

    # From config:
    device = config.device
    # if 'mask_name' in config:
    #     mask_name = config.mask_name
    #     if mask_name.lower() in ['simtb16']:
    #         data_seed = config.data_seed
    #     elif mask_name.lower() in ['ukb2907-smri-aal2']:
    if 'input_dim' in config:
        input_dim = config.input_dim

    if 'output_dim' in config:
        output_dim = config.output_dim
    elif 'input_dim' in config:
        output_dim = config.input_dim

    if 'subspace' in config:
        subspace = config.subspace
    else:
        pass # TO DO: should error...

    if 'eta' in config:
        eta = config.eta
    
    if 'beta' in config:
        beta = config.beta

    if 'lam' in config:
        lam = config.lam

    if config.special.nRuns != []:
        nRuns = config.special.nRuns
    else:
        nRuns = loguniform_int(0, 10).rvs(size=1)[0]
    
    if config.special.epochs != []:
        epochs = config.special.epochs
    else:
        # the integer type returned by loguniform_int is int64, 
        # which can't recognized as an int in DataLoader,
        # so need to cast int64 to int here
        epochs = int(loguniform_int(100, 500).rvs(size=1)[0])

    # Batch size is being passed as arg from script, not from config file
    # if config.special.batch_size != []:
    #     batch_size = config.special.batch_size
    # else:
    #     batch_size = int(loguniform_int(20, 1000).rvs(size=1)[0])

    # Learning rate is being passed as arg from script, not from config file
    # if config.special.lr != []:
    #     lr = config.special.lr
    # else:
    #     lr = loguniform.rvs(0.00001, 0.1, size=1)[0]
    
    # results = {l: {n: [] for n in data_seed} for l in n_layers}

    # recovered_sources = {l: {n: [] for n in data_seed} for l in n_layers}
    recovered_sources = []
    final_MISIs = []

    # for l in n_layers:
    #     for n in data_seed:
    if data.lower() == 'mat':
        # load the data
        # matfile = os.path.join('./simulation_data', 'sim-{}.mat'.format(config.dataset))
        matfile = os.path.join('/data/users4/xli/MISA-pytorch/simulation_data', data_filename)
        ds=Dataset(data_in=matfile, device=device)
        if len(ds) < batch_size:
            batch_size = len(ds)
        train_data=DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
        test_data=DataLoader(dataset=ds, batch_size=len(ds), shuffle=False)
        
        try:
            matdict=sio.loadmat(matfile)
            matW=np.squeeze(matdict[w])
            if A_exist:
                matA=np.squeeze(matdict['A'])
                matY=np.squeeze(matdict['Y'])
        except:
            matdict=mat73.loadmat(matfile)
            matW=matdict[w]
            if A_exist:
                matA=matdict['A']
                matY=matdict['Y']
        
        # LOAD INITIAL WEIGHTS
        initial_weights = [i for _, i in enumerate(matW)]
        
        # LOAD GROUND-TRUTH A MATRIX
        ground_truth_A = None
        if A_exist:
            ground_truth_A = [i for _, i in enumerate(matA)]
            ground_truth_Y = [i.T for _, i in enumerate(matY)]
            # config.output_dim = [AA.shape[1] for AA in ground_truth_A]
        
        num_modal = ds.num_modal
        index = slice(0, num_modal)
        
        input_dim = [torch.tensor(dd.shape[-1],device=device) for dd in ds.mat_data]
        if config.output_dim != []:
            output_dim = [torch.tensor(dd,device=device) for dd in output_dim]
        else:
            output_dim = input_dim
        print(input_dim, output_dim)
        if subspace.lower() == 'iva':
            subspace = [torch.eye(dd, device=device) for dd in output_dim]
        
        if len(eta) > 0:
            eta = torch.tensor(eta, dtype=torch.float32, device=device)
            if len(eta) == 1:
                eta = eta*torch.ones(subspace[0].size(-2), device=device)
        else:
            # should error
            pass
        if len(beta) > 0:
            beta = torch.tensor(beta, dtype=torch.float32, device=device)
            if len(beta) == 1:
                beta = beta*torch.ones(subspace[0].size(-2), device=device)
        else:
            # should error
            pass
        if len(lam) > 0:
            lam = torch.tensor(lam, dtype=torch.float32, device=device)
            if len(lam) == 1:
                lam = lam*torch.ones(subspace[0].size(-2), device=device)
        else:
            # should error
            pass
        # load ground-truth sources for comparison
        # s = ...
        # pass
    else:
        if mask_name.lower() in ['simtb16']:
            pass
        elif mask_name.lower() in ['ukb2907-smri-aal2']:
            pass

    for seed in range(nRuns):
        # print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
        
        if data.lower() == 'mat':
            # ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_s{}.pt'.format(data, config.dataset, seed))
            ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_{}_s{}.pt'.format(data.lower(), data_filename.split('.')[0], w, seed))
        # else:
        #     ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_s{}.pt'.format(data, mask_name, seed))
        recov_sources, final_MISI, run_time, epochs_completed = MISA_wrapper(data_loader=train_data,
                                     index=index,
                                     subspace=subspace, 
                                     eta=eta, 
                                     beta=beta, 
                                     lam=lam,
                                     input_dim=input_dim, 
                                     output_dim=output_dim, 
                                     seed=seed,
                                     epochs=epochs,
                                     lr=lr,
                                     weights=initial_weights,
                                     A=ground_truth_A,
                                     device=device,
                                     ckpt_file=ckpt_file,
                                     test=test,
                                     test_data_loader=test_data,
                                     beta1=beta1,
                                     beta2=beta2,
                                     batch_size=batch_size,
                                     patience=patience,
                                     fused=fused,
                                     foreach=foreach)
        
        # store results
        # recovered_sources[l][n].append(recov_sources)
        recovered_sources.append(recov_sources)
        final_MISIs.append(final_MISI)

        # if mask_name.lower() in ['ukb2907-smri-aal2']:
        #     continue

        # results[l][n].append(np.min([metric(z, s) for z in recov_sources]))
        # print(np.min([metric(z, s) for z in recov_sources]))

    # prepare output
    if data.lower() == 'mat':
        pass
        Results = {
            # 'input_dim': input_dim,
            # 'CorrelationCoef': results,
            'recovered_sources': recovered_sources,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'final_MISIs': final_MISIs,
            'beta1': beta1,
            'beta2': beta2,
            'run_time': run_time,
            'patience': patience,
            'fused': fused,
            'foreach': foreach,
            'epochs_completed': epochs_completed}
    # else:
    #     if mask_name.lower() in ['simtb16']:
    #         Results = {
    #             'mask_name': mask_name,
    #             'CorrelationCoef': results,
    #             'recovered_sources': recovered_sources
    #         }
    #     elif mask_name.lower() in ['ukb2907-smri-aal2']:
    #         Results = {
    #             'mask_name': mask_name,
    #             'recovered_sources': recovered_sources
    #         }
        

    return Results