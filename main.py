import os
import sys
import argparse
import pickle
import torch
import numpy as np
import yaml
from runners.generic_runner import run_misa
import datetime
import csv

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
    required.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for model training')
    optional.add_argument('-a', '--a_exist', action='store_true', help='Whether the dataset includes ground truth A matrix')
    required.add_argument('-b1', '--beta1', type=float, default=0.8, help='Beta1 parameter for Adam optimizer')
    required.add_argument('-b2', '--beta2', type=float, default=0.9, help='Beta2 parameter for Adam optimizer')
    required.add_argument('-bs', '--batch_size', type=int, default=500, help='Batch size for training')
    required.add_argument('-e', '--experimenter', type=str, default='', help='Name of the experimenter')
    required.add_argument('-ff', '--adam_params', type=int, default=0, help='Whether fused=true or false and foreach=true or false')
    required.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    required.add_argument('-s', '--seed', type=int, default=1, help='Random seed for reproducibility')
    required.add_argument('-g', '--gpu', type=str, default='A100', help='Defines which GPU to use on the server')


    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

def make_dirs_simulations(args):
    os.makedirs(args.run, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.config.split('/')[-1].split('.')[0])
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

def append_csv(csv_path, experimenter, date_time, batch_size, learning_rate, weights, seed, patience, beta1, beta2, fused, foreach, run_time, epochs_completed, gpu, dataset_filename, running_data):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if file_exists:
            writer.writerows([[experimenter, date_time, batch_size, learning_rate, weights, seed, patience, beta1, beta2, fused, foreach, run_time, epochs_completed, gpu, dataset_filename, running_data]])
        else:
            writer.writerows([["experimenter","date_time", "batch_size", "learning_rate", "weights", "seed", "patience", "beta1", "beta2", "fused", "foreach", "run_time", "epochs_completed", "GPU", "dataset_filename", "running_data"]], 
                [[experimenter, date_time, batch_size, learning_rate, weights, seed, patience, beta1, beta2, fused, foreach, run_time, epochs_completed, gpu, dataset_filename, running_data]])


if __name__ == '__main__':
    csv_path = "/data/users1/cmccurdy5/MISA-pytorch/results/MathPath2024/misa_results.csv"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_T%H-%M-%S")
    start_time = datetime.datetime.now()

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
        print("Finished running misa")
        for k, v in r.items():
            if type(v) == list:
                vcpu=[]
                if isinstance(v[0], (np.ndarray, np.generic) ):
                    vcpu = v
                else:
                    for i, j in enumerate(v[0]):
                        vcpu.append(j.detach().cpu().numpy())
                r[k] = vcpu
                print("Finished updating r")
        append_csv(csv_path, args.experimenter, date_time, args.batch_size, args.learning_rate, args.weights, args.seed, args.patience, args.beta1, args.beta2, r.get("fused"), r.get("foreach"), r.get('run_time'), r.get('epochs_completed'), args.gpu, args.filename, args.run)
        # save results
        # runner loops over many seeds, so the saved file contains results from multiple runs
        if args.test:
            fname = os.path.join(args.run, 'res_' + args.filename.split('.')[0] + '_' + args.weights + '_test.p')
        else:
            fname = os.path.join(args.run, 'res_' + args.filename.split('.')[0] + '_' + args.weights + '.p')

        pickle.dump(r, open(fname, "wb"))
        print("Finished saving data")
    else:
        raise ValueError('Unsupported data {}'.format(args.data))
