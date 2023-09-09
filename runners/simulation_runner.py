import os
import torch
import numpy as np
import scipy.io as sio
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from data.imca import generate_synthetic_data, ConditionalDataset
from metrics.mcc import mean_corr_coef, mean_corr_coef_per_segment
from metrics.mmse import MMSE
from model.ivae.ivae_core import iVAE
from model.ivae.ivae_wrapper import IVAE_wrapper_
from model.MISAK import MISA
from model.misa_wrapper import MISA_wrapper_
from scipy.stats import loguniform
from data.utils import to_one_hot
from sklearn.decomposition import PCA

def run_ivae_exp(args, config):
    """run iVAE simulations"""
    method = args.method
    n_modality = config.n_modalities
    n_sims = config.n_sims
    experiment = config.experiment
    
    # iVAE config
    if args.n_sources:
        data_dim = args.n_sources
    else:
        data_dim = config.data_dim
    
    if args.n_segments:
        n_segments = args.n_segments
    else:
        n_segments = config.n_segments
    
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    data_seed = config.data_seed
    cuda = config.ivae.cuda
    batch_size_ivae = config.ivae.batch_size
    device = config.device
    dataset = config.dataset

    # MISA config
    input_dim = [data_dim] * n_modality
    output_dim = [data_dim] * n_modality
    subspace = config.subspace
    if subspace.lower() == 'iva':
        subspace = [torch.eye(dd, device=device) for dd in output_dim]

    eta = config.eta
    beta = config.beta
    lam = config.lam
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
    
    ### TODO update code!
    if args.n_epochs:
        n_epochs = args.n_epochs
    else:
        n_epochs = config.n_epochs

    if args.ivae_lr:
        lr_ivae_list = [args.ivae_lr]
    else:
        lr_ivae_list = config.ivae.lr

    if args.ivae_max_iter_per_epoch:
        max_iter_per_epoch_list = [args.ivae_max_iter_per_epoch]
    else:
        max_iter_per_epoch_list = config.ivae.max_iter_per_epoch
    
    n_runs = config.misa.n_runs
    batch_size_misa = config.misa.batch_size
    lr_misa_list = config.misa.lr
    index = slice(0, n_modality)
    data_path = args.data_path

    epoch_interval = 10 # save result every 10 epochs
    epoch_last = n_epochs // epoch_interval
    res_corr = {l: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for l in n_layers}
    res_recovered_source = {l: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for l in n_layers}
    res_ground_truth_source = {l: [] for l in n_layers}
    res_metric = {l: [] for l in n_layers}

    for l in n_layers:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(data_dim, n_segments, n, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                initial_weights = []
            
            elif experiment == "img":
                # TODO implement MGPCA
                # TODO scale covariance matrix to have equal weight
                # TODO reconstruct W 44318 x 30 x 2
                data = sio.loadmat(data_path)
                x_orig = data['x'] # 2907 x 44318 x 2
                w = data['w'] # 30 x 44318 x 2
                x = np.concatenate([np.expand_dims(x_orig[:,:,0] @ w[:,:,0].T, axis=2), np.expand_dims(x_orig[:,:,1] @ w[:,:,1].T, axis=2)], axis=2)
                initial_weights = []

                # x_cat = np.concatenate([x_orig[:,:,0], x_orig[:,:,1]], axis=0) # 5814 x 44318
                # pca = PCA(n_components=data_dim)
                # x_pca = pca.fit_transform(x_cat) # 5814 x 30
                # x = np.concatenate([np.expand_dims(x_pca[:x_orig.shape[0],:], axis=2), np.expand_dims(x_pca[x_orig.shape[0]:,:], axis=2)], axis=2) # 2907 x 30 x 2

                u = data['u'] 
                y = to_one_hot(u)[0] # 2907 x 14
            
            # x dimension: 4000 samples x 10 sources x 2 modalities; y dimension: 4000 samples x 20 one-hot encoding labels
            for seed in range(n_sims):
                for mi_ivae in max_iter_per_epoch_list:
                    for lr_ivae in lr_ivae_list:
                        
                        lr_misa = lr_ivae/n_segments
                        mi_misa = mi_ivae
                        
                        print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segments}; n_source={data_dim}; seed={seed}; n_epochs={n_epochs}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
                        
                        loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}

                        # TODO optimize workflow, move code block to a wrapper function
                        if method.lower() == 'diva':
                            model_iVAE_list = []

                            # initiate iVAE model for each modality
                            for m in range(n_modality):
                                ckpt_file = os.path.join(args.run, f'{experiment}_diva_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_modality{m+1}_epoch{n_epochs}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                                ds = ConditionalDataset(x[:,:,m].astype(np.float32), y.astype(np.float32), device)
                                train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_ivae, **loader_params)
                                data_dim, latent_dim, aux_dim = ds.get_dims() # data_dim = 30, latent_dim = 30, aux_dim = 14
                                
                                model_iVAE = iVAE(latent_dim, 
                                                data_dim, 
                                                aux_dim, 
                                                activation='lrelu', 
                                                device=device, 
                                                n_layers=l, 
                                                hidden_dim=data_dim * 2,
                                                method=method.lower())
                                
                                res_iVAE, model_iVAE, params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, n_layers=l, 
                                                            hidden_dim=data_dim * 2, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                            ckpt_file=ckpt_file, seed=seed, model=model_iVAE) #model=model_MISA.input_model[m]
                                
                                model_iVAE_list.append(model_iVAE)
                            
                            for m in range(n_modality):
                                model_iVAE_list[m].set_aux(False)
                                print(f"model_iVAE_list[{m}].use_aux = {model_iVAE_list[m].use_aux}")

                            model_MISA = MISA(weights=initial_weights, # TODO MGPCA weights "mgpca"
                                                index=index, 
                                                subspace=subspace, 
                                                eta=eta, 
                                                beta=beta, 
                                                lam=lam, 
                                                input_dim=input_dim, 
                                                output_dim=output_dim, 
                                                seed=seed, 
                                                device=device,
                                                model=model_iVAE_list)
                                                
                            # update iVAE and MISA model weights
                            # run iVAE per modality
                            np.random.seed(7)
                            rand_seq = np.random.randint(0, 1000, size=n_epochs*mi_misa)

                            for e in range(n_epochs):
                                print('Epoch: {}'.format(e))
                                # loop MISA through segments
                                # remove the mean of segment because MISA loss assumes zero mean
                                # randomize segment order
                                for it in range(mi_misa):
                                    np.random.seed(rand_seq[e*mi_misa+it])
                                    segment_shuffled = np.arange(n_segments)
                                    np.random.shuffle(segment_shuffled)

                                    for seg in segment_shuffled:
                                        if experiment == "sim":
                                            y_seg = y[seg*n:(seg+1)*n]
                                            x_seg = x[seg*n:(seg+1)*n,:,:]
                                        elif experiment == "img":
                                            ind = np.where(y[:,seg]==1)[0]
                                            y_seg = y[ind,:]
                                            x_seg = x[ind,:,:]

                                        x_seg_dm = x_seg - np.mean(x_seg, axis=0) # remove mean of segment
                                        
                                        ds = ConditionalDataset(x_seg_dm.astype(np.float32), y_seg.astype(np.float32), device)
                                        train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_misa, **loader_params)
                                        test_loader = DataLoader(ds, shuffle=False, batch_size=len(ds), **loader_params)

                                        model_MISA, final_MISI = MISA_wrapper_(data_loader=train_loader,
                                                            test_data_loader=test_loader,
                                                            epochs=1,
                                                            lr=lr_misa,
                                                            device=device,
                                                            ckpt_file=ckpt_file,
                                                            model_MISA=model_MISA)

                                for m in range(n_modality):
                                    model_MISA.input_model[m].set_aux(True)
                                    print(f"model_MISA.input_model[{m}].use_aux = {model_MISA.input_model[m].use_aux}")
                                    res_iVAE, model_MISA.input_model[m], params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, n_layers=n_layers, hidden_dim=data_dim * 2,
                                                        cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=False, model=model_MISA.input_model[m])
                                    model_MISA.input_model[m].set_aux(False)
                                    print(f"model_MISA.input_model[{m}].use_aux = {model_MISA.input_model[m].use_aux}")
                                    
                                    # store results every epoch_interval epochs
                                    if e % epoch_interval == 0:
                                        res_ivae = res_iVAE.detach().numpy()
                                        
                                        if experiment == 'sim':
                                            res_corr[l][e].append(mean_corr_coef(res_ivae, s[:,:,m]))
                                            res_corr[l][e].append(mean_corr_coef_per_segment(res_ivae, s[:,:,m], y))
                                            print(res_corr[l][e])
                                            res_recovered_source[l][e].append(res_ivae)
                                            if e//epoch_interval == epoch_last and m == n_modality - 1: # last epoch, last modality
                                                res_ground_truth_source[l].append(s)
                                                res_ivae_stack = np.dstack(res_recovered_source[l][e])
                                                mse_ps, mcc_ps, Rr_ps, mse_pm, mcc_pm, Rr_pm, mse, mcc, Rr = MMSE(res_ivae_stack, s, y)
                                                metric = { 'mse_ps': mse_ps, 'mcc_ps': mcc_ps, 'Rr_ps': Rr_ps, 'mse_pm': mse_pm, 'mcc_pm': mcc_pm, 'Rr_pm': Rr_pm, 'mse': mse, 'mcc': mcc, 'Rr': Rr }
                                                res_metric[l].append(metric)
                                        
                                        elif experiment == 'img':
                                            res_recovered_source[l][e].append(res_ivae)

                        elif method.lower() == 'ivae':
                            # intiate iVAE model for each modality
                            model_iVAE_list = []
                            
                            for m in range(n_modality):
                                ckpt_file = os.path.join(args.run, f'{experiment}_ivae_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_modality{m+1}_epoch{n_epochs}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                                ds = ConditionalDataset(x[:,:,m].astype(np.float32), y.astype(np.float32), device)
                                train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_ivae, **loader_params)
                                data_dim, latent_dim, aux_dim = ds.get_dims() # data_dim: 10, latent_dim: 10, aux_dim: 20
                                
                                model_iVAE = iVAE(latent_dim, 
                                                data_dim, 
                                                aux_dim, 
                                                activation='lrelu', 
                                                device=device, 
                                                n_layers=l, 
                                                hidden_dim=data_dim * 2,
                                                method=method.lower())
                                
                                for e in range(n_epochs):
                                    print('Epoch: {}'.format(e))
                                    res_iVAE, model_iVAE, params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, n_layers=l, 
                                                                hidden_dim=data_dim * 2, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                                ckpt_file=ckpt_file, seed=seed, model=model_iVAE)
                                    
                                    if e % epoch_interval == 0:
                                        res_ivae = res_iVAE.detach().numpy()
                                        
                                        if experiment == 'sim':
                                            res_corr[l][e].append(mean_corr_coef(res_ivae, s[:,:,m]))
                                            res_corr[l][e].append(mean_corr_coef_per_segment(res_ivae, s[:,:,m], y))
                                            print(res_corr[l][e])
                                            res_recovered_source[l][e].append(res_ivae)
                                        
                                            if e//epoch_interval == epoch_last and m == n_modality - 1: # last epoch, last modality
                                                res_ground_truth_source[l].append(s)
                                                res_ivae_stack = np.dstack(res_recovered_source[l][e])
                                                mse_ps, mcc_ps, Rr_ps, mse_pm, mcc_pm, Rr_pm, mse, mcc, Rr = MMSE(res_ivae_stack, s, y)
                                                metric = { 'mse_ps': mse_ps, 'mcc_ps': mcc_ps, 'Rr_ps': Rr_ps, 'mse_pm': mse_pm, 'mcc_pm': mcc_pm, 'Rr_pm': Rr_pm, 'mse': mse, 'mcc': mcc, 'Rr': Rr }
                                                res_metric[l].append(metric)
                                        
                                        elif experiment == 'img':
                                            res_recovered_source[l][e].append(res_ivae)
                                
                                model_iVAE_list.append(model_iVAE)
                                
                        elif method.lower() == 'misa':
                            ckpt_file = os.path.join(args.run, f'{experiment}_misa_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_epoch{n_epochs}_maxiter{mi_misa}_lrmisa{round(lr_misa, 5)}.pt')
                            
                            model_MISA = MISA(weights=initial_weights,
                                index=index, 
                                subspace=subspace, 
                                eta=eta, 
                                beta=beta, 
                                lam=lam, 
                                input_dim=input_dim, 
                                output_dim=output_dim, 
                                seed=seed, 
                                device=device)

                            # update iVAE and MISA model weights
                            # run iVAE per modality
                            np.random.seed(7)
                            segment_shuffled = np.arange(n_segments)
                            np.random.shuffle(segment_shuffled)

                            if experiment == "sim":
                                res_MISA = np.zeros_like(s)
                            else:
                                res_MISA = np.zeros((x.shape[0], data_dim, n_modality))

                            for e in range(n_epochs):
                                print('Epoch: {}'.format(e))
                                # loop MISA through segments
                                # remove the mean of segment because MISA loss assumes zero mean
                                # randomize segment order
                                for seg in segment_shuffled:
                                    if experiment == "sim":
                                        x_seg = x[seg*n:(seg+1)*n,:,:]
                                    elif experiment == "img":
                                        ind = np.where(y[:,seg]==1)[0]
                                        x_seg = x[ind,:,:]
                                    x_seg_dm = x_seg - np.mean(x_seg, axis=0) # remove mean of segment
                                    # a list of datasets, each dataset dimension is sample x source
                                    ds = Dataset(data_in=x_seg_dm, device=device)
                                    train_loader = DataLoader(dataset=ds, batch_size=batch_size_misa, shuffle=True)
                                    test_loader = DataLoader(dataset=ds, batch_size=len(ds), shuffle=False)

                                    model_MISA, final_MISI = MISA_wrapper_(data_loader=train_loader,
                                                        test_data_loader=test_loader,
                                                        epochs=mi_misa,
                                                        lr=lr_misa,
                                                        device=device,
                                                        ckpt_file=ckpt_file,
                                                        model_MISA=model_MISA)

                                    if e % epoch_interval == 0:
                                        for m in range(n_modality):
                                            if experiment == "sim":
                                                res_MISA[seg*n:(seg+1)*n,:,m] = model_MISA.output[m].detach().numpy()
                                            elif experiment == "img":
                                                res_MISA[ind,:,m] = model_MISA.output[m].detach().numpy()
                                
                                if e % epoch_interval == 0:
                                    for m in range(n_modality):
                                        if experiment == 'sim':
                                            res_corr[l][e].append(mean_corr_coef(res_MISA[:,:,m], s[:,:,m]))
                                            res_corr[l][e].append(mean_corr_coef_per_segment(res_MISA[:,:,m], s[:,:,m], y))
                                            print(res_corr[l][e])
                                            res_recovered_source[l][e].append(res_MISA[:,:,m])
                                            if e//epoch_interval == epoch_last and m == n_modality - 1: # last epoch, last modality
                                                res_ground_truth_source[l].append(s)
                                                mse_ps, mcc_ps, Rr_ps, mse_pm, mcc_pm, Rr_pm, mse, mcc, Rr = MMSE(res_MISA, s, y)
                                                metric = { 'mse_ps': mse_ps, 'mcc_ps': mcc_ps, 'Rr_ps': Rr_ps, 'mse_pm': mse_pm, 'mcc_pm': mcc_pm, 'Rr_pm': Rr_pm, 'mse': mse, 'mcc': mcc, 'Rr': Rr }
                                                res_metric[l].append(metric)
                                        elif experiment == 'img':
                                            res_recovered_source[l][e].append(res_MISA[:,:,m])
                                
    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': n_segments,
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results
