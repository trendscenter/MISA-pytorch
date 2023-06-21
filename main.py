import os
import sys
import argparse
import pickle
import torch
import numpy as np
import yaml
from runners.generic_runner import run_misa
from runners.simulation_runner import run_ivae_exp

def parse_sim():
    
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    optional = parser._action_groups.pop()
    
    required.add_argument('-d', '--data', type=str, default='MAT', help='Dataset to run experiments. Should be MAT, FUSION, or FMRI')
    required.add_argument('-f', '--filename', type=str, default='sim-siva.mat', help='Dataset filename')
    required.add_argument('-w', '--weights', type=str, default='w0', help='Name of weighted matrix W in the dataset')
    required.add_argument('-c', '--config', type=str, default='sim-siva.yaml', help='Path to the config file')
    required.add_argument('-r', '--run', type=str, default='run/', help='Path for saving running related data.')
    required.add_argument('-t', '--test', action='store_true', help='Whether to evaluate the models from checkpoints')
    
    # required.add_argument('--dataset', type=str, default='TCL', help='Dataset to run experiments. Should be TCL or IMCA')
    required.add_argument('-m', '--method', type=str, default='icebeem',
                        help='Method to employ. Should be TCL, iVAE or ICE-BeeM')
    required.add_argument('-n', '--n_sims', type=int, default=10, help='Number of simulations to run')

    optional.add_argument('-a', '--a_exist', action='store_true', help='Whether the dataset includes ground truth A matrix')
    optional.add_argument('--n_epochs', type=int, default=2000, help='Number of epochs')
    optional.add_argument('--ivae_lr', type=float, default=0.001, help='iVAE learning rate')
    optional.add_argument('--ivae_max_iter_per_epoch', type=int, default=10, help='Number of maximum iterations per epoch')
    optional.add_argument('--data_path', default='neuroimaging_data/mmiva_ivae.mat', type=str, help='Path of dataset')
    optional.add_argument('--n_sources', type=int, default=30, help='Number of sources')
    optional.add_argument('--n_obs_per_seg', type=int, default=200, help='Number of observations per segment')
    optional.add_argument('--n_segments', type=int, default=14, help='Number of segments')

    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def make_dirs_simulations(args):
    os.makedirs(args.run, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.config.split('.')[0])
    os.makedirs(args.checkpoints, exist_ok=True)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



if __name__ == '__main__':
    args = parse_sim()
    print('\n\n\nRunning {} experiments'.format(args.data))
    # make checkpoint and log folders
    make_dirs_simulations(args)

    if args.data.lower() in ['mat', 'fusion', 'fmri']:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if args.method.lower() in ['ivae', 'misa', 'diva']:
            r = run_ivae_exp(args, new_config)
        else:
            r = run_misa(args, new_config)
            for k, v in r.items():
                if type(v) == list:
                    vcpu=[]
                    if isinstance(v[0], (np.ndarray, np.generic)):
                        vcpu = v
                    else:
                        for i, j in enumerate(v[0]):
                            vcpu.append(j.detach().cpu().numpy())
                    r[k] = vcpu
        
        # save results
        # runner loops over many seeds, so the saved file contains results from multiple runs
        if args.test:
            fname = os.path.join(args.run, 'res_' + args.filename.split('.')[0] + '_' + args.weights + '_test.p')
        elif args.method.lower() in ['ivae', 'misa', 'diva']:
            fname = os.path.join(args.run, f'res_{args.method.lower()}_source{args.n_sources}_obs{args.n_obs_per_seg}_seg{args.n_segments}_epoch{args.n_epochs}_bsmisa{new_config.misa.batch_size}_bsivae{new_config.ivae.batch_size}_lrivae{args.ivae_lr}_maxiter{args.ivae_max_iter_per_epoch}.p')
        else:
            fname = os.path.join(args.run, 'res_' + args.filename.split('.')[0] + '_' + args.weights + '.p')

        pickle.dump(r, open(fname, "wb"))

    else:
        raise ValueError('Unsupported data {}'.format(args.data))
