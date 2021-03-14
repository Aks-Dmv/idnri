from synthetic_sim import ChargedParticlesSim
import time
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=1000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=1000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=1000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5100,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=5100,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=3,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--save-dir', type=str, default='data/charges',
                    help='Where to save the data')

args = parser.parse_args()

if args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 50 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all


print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} test simulations".format(args.num_test))
loc_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                  args.length_test,
                                                  args.sample_freq)

# [num_samples, num_timesteps, num_dims, num_atoms]
num_atoms = loc_train.shape[3]

loc_max = loc_train.max()
loc_min = loc_train.min()
vel_max = vel_train.max()
vel_min = vel_train.min()

# Normalize to [-1, 1]
loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

# Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
loc_train = np.transpose(loc_train, [0, 3, 1, 2])
vel_train = np.transpose(vel_train, [0, 3, 1, 2])
feat_train = np.concatenate([loc_train, vel_train], axis=3)
edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
edges_train = np.array((edges_train + 3) / 2, dtype=np.int64)

loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
edges_valid = np.array((edges_valid + 3) / 2, dtype=np.int64)

loc_test = np.transpose(loc_test, [0, 3, 1, 2])
vel_test = np.transpose(vel_test, [0, 3, 1, 2])
feat_test = np.concatenate([loc_test, vel_test], axis=3)
edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
edges_test = np.array((edges_test + 3) / 2, dtype=np.int64)

feat_train = np.transpose(feat_train, [0, 2, 1, 3])
feat_valid = np.transpose(feat_valid, [0, 2, 1, 3])
feat_test = np.transpose(feat_test, [0, 2, 1, 3])

feat_train = torch.FloatTensor(feat_train)
edges_train = torch.LongTensor(edges_train)
feat_valid = torch.FloatTensor(feat_valid)
edges_valid = torch.LongTensor(edges_valid)
feat_test = torch.FloatTensor(feat_test)
edges_test = torch.LongTensor(edges_test)

# Exclude self edges
off_diag_idx = np.ravel_multi_index(
    np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    [num_atoms, num_atoms])
edges_train = edges_train[:, off_diag_idx]
edges_valid = edges_valid[:, off_diag_idx]
edges_test = edges_test[:, off_diag_idx]

train_data = torch.FloatTensor(feat_train)
val_data = torch.FloatTensor(feat_valid)
test_data = torch.FloatTensor(feat_test)
torch.save(train_data, args.save_dir + "/" + 'train_feats')
torch.save(val_data, args.save_dir + "/" + 'val_feats')
torch.save(test_data, args.save_dir + "/" + 'test_feats')

train_edges = torch.FloatTensor(edges_train.reshape(train_data.shape[0], train_data.shape[1], -1).float())
val_edges = torch.FloatTensor(edges_valid.reshape(val_data.shape[0], val_data.shape[1], -1).float())
test_edges = torch.FloatTensor(edges_test.reshape(test_data.shape[0], test_data.shape[1], -1).float())
torch.save(train_edges, args.save_dir + "/" + 'train_edges')
torch.save(val_edges, args.save_dir + "/" + 'val_edges')
torch.save(test_edges, args.save_dir + "/" + 'test_edges')
