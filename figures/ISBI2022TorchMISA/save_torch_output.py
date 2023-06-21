import os
import torch
import numpy as np

seed_list = [7, 14, 21]
dataset_list = [2, 12, 32, 100]
source_list = [12, 32, 100]
sample_list = [64, 256, 1024, 4096, 16384, 32768]
w_list = ['wpca', 'w0', 'w1']

datadir = '/Users/xli77/Documents/MISA-pytorch/run/checkpoints/sim-siva'

# isi = np.zeros([len(dataset_list), len(source_list), len(sample_list), len(seed_list), len(w_list)])
# train_loss = np.zeros([len(dataset_list), len(source_list), len(sample_list), len(seed_list), len(w_list)])
test_loss = np.zeros([len(dataset_list), len(source_list), len(sample_list), len(seed_list), len(w_list)])

for i, n_dataset in enumerate(dataset_list):
    for j, n_source in enumerate(source_list):
        if (n_dataset==32 and n_source==100) or (n_dataset==100 and n_source==32) or (n_dataset==100 and n_source==100):
            continue
        for k, n_sample in enumerate(sample_list):
            if (n_source > n_sample) or (n_dataset > n_sample):
                continue
            for m, w in enumerate(w_list):
                for n, seed in enumerate(seed_list):
                    datapath=os.path.join(datadir, f'misa_mat_sim-siva_dataset{n_dataset}_source{n_source}_sample{n_sample}_seed{seed}_{w}_s0.pt')
                    import pdb; pdb.set_trace()
                    try:
                        # isi[i,j,k,m,n]=torch.load(datapath)['training_MISI'][-1]
                        # train_loss[i,j,k,m,n]=torch.load(datapath)['training_loss'][-1]
                        test_loss[i,j,k,m,n]=float(torch.load(datapath)['test_loss'][0].numpy())
                    except:
                        print(f"missing dataset{n_dataset}_source{n_source}_sample{n_sample}_seed{seed}_{w}")
                        pass

# np.save(os.path.join(datadir, 'misa_mat_sim-siva_isi_all.npy'), isi)
np.save(os.path.join(datadir, 'misa_mat_sim-siva_loss_all.npy'), test_loss)