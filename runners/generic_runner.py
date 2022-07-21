import os

import numpy as np
# from sklearn.decomposition import FastICA

# from metrics.mcc import mean_corr_coef
from model.misa_wrapper import MISA_wrapper



def run_misa(args, config):
    """run MISA"""
    n_layers = config.n_layers
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
    
    

    lr = config.special.lr
    epochs = config.special.epochs
    

    # results = {l: {n: [] for n in data_seed} for l in n_layers}

    data = args.data
    nRuns = config.nRuns
    test = args.test

    # recovered_sources = {l: {n: [] for n in data_seed} for l in n_layers}
    recovered_sources = []

    # for l in n_layers:
    #     for n in data_seed:
    if data.lower() == 'mat':
        # load the data
        
        num_modal = len(x)
        index = slice(0, num_modal)

        train_data = DataLoader("Insert parameters here", lr = lr, shuffle = True)
        
        # load ground-truth sources for comparison
        # s = ...
        
    else:
        if mask_name.lower() in ['simtb16']:
            pass
        elif mask_name.lower() in ['ukb2907-smri-aal2']:
            pass

    for seed in range(nRuns):
        # print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
        
        if data.lower() == 'mat':
            ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_s{}.pt'.format(data, config.dataset, seed))
        # else:
        #     ckpt_file = os.path.join(args.checkpoints, 'misa_{}_{}_s{}.pt'.format(data, mask_name, seed))
        recov_sources = MISA_wrapper(data_loader=train_data,
                                    index = index,
                                    n_layers=n_layers,
                                    input_dim=input_dim, 
                                    output_dim=output_dim, 
                                    epochs=epochs,
                                    lr=lr,
                                    seed=seed,
                                    ckpt_file=ckpt_file,
                                    test=test)
        
        
        # store results
        # recovered_sources[l][n].append(recov_sources)
        recovered_sources.append(recov_sources)

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
            'recovered_sources': recovered_sources
        }
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