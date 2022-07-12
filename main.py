import argparse
import os
import pickle
import torch
import yaml
from runners.generic_runner import run_misa


def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default='MAT', help='Dataset to run experiments. Should be MAT, FUSION, or FMRI')
    parser.add_argument('--config', type=str, default='sim-iva-3x16.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')

    parser.add_argument('--test', action='store_true', help='Whether to evaluate the models from checkpoints')

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
        
        r = run_misa(args, new_config)
        
        # save results
        # runner loops over many seeds, so the saved file contains results from multiple runs
        fname = os.path.join(args.run, 'res_' + args.config.split('.')[0] + '.p')
        pickle.dump(r, open(fname, "wb"))

    else:
        raise ValueError('Unsupported data {}'.format(args.data))
